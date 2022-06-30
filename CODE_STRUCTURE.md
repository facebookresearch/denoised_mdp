# Code Structure Overview


Code structure in this repository is designed to be adaptable to general model-based reinforcement learning (MBRL) algorithms which perform the following steps in a loop:
1. Experience collection,
2. Model fitting,
3. Policy optimization.

These three parts consist of the main algorithmic components. In addition, we have code for handling argument-parsing, pre-emption, logging, checkpointing, etc.


The main entry point is [`main()` in `main.py`](./main.py#L511), which performs
+ [argument parsing](./main.py#L513-514),
+ [pre-emption mechanism registration](./main.py#L516-525),
+ [basic logging of the configurations](./main.py#L529-543),
+ [random seeding](./main.py#L545-548),
+ [model-based training algorithm](./main.py#L550-573).


## A Common Pattern: Configurable Objects

Throughout the codebase, we heavily use a pattern to define functions/classes configurable by Hydra, where each such function/class is accompanied with an `attrs` config class (sometimes as an inner class).

Such config classes only serve for the purpose of argument parsing, default value specification, and value checking. They are **not** part of the algorithmic core code that defines models or implements any algorithm. Feel free to skip them when reading the code, unless you are interested in our [Argument Parsing](#argument-parsing) mechanism.

<details>
<summary>
For the interested readers, click to see an example showcasing these configs are defined.
</summary>

<div style="margin-left:15px">

We will explain using the example of the reward model from [denoised_mdp/agents/denoised_mdp.py](./denoised_mdp/agents/denoised_mdp.py#L81-104):

```py
def reward_model_parser(dense_activation_fn, hidden_size, stddev, *,
                        transition_model: TransitionModel):
    return RewardModel(
        transition_model.x_belief_size,
        transition_model.x_state_size,
        transition_model.y_belief_size,
        transition_model.y_state_size,
        hidden_size,
        stddev=stddev,
        activation_function=dense_activation_fn,
    )


@attrs.define(kw_only=True, auto_attribs=True)
class RewardModelConfig:
    _target_: str = attrs.Factory(lambda: f"{reward_model_parser.__module__}.{reward_model_parser.__qualname__}")
    _partial_: bool = True

    class InstantiatedT(Protocol):  # for typing
        def __call__(self, *, transition_model: TransitionModel) -> RewardModel: ...

    dense_activation_fn: ActivationKind = ActivationKind.elu
    hidden_size: int = attrs.field(default=400, validator=attrs.validators.gt(0))
    stddev: float = attrs.field(default=1, validator=attrs.validators.gt(0))
```

**`reward_model_parser` is the function being configured**, with a config specified as the `attrs` class, `RewardModelConfig`. An `attrs` class is just like a `dataclass` but with more functionalities like value validation (such as `validator=attrs.validators.gt(0)` used here).

Here we have three configurable arguments, `dense_activation_fn`, `hidden_size`, and `stddev`, which are defined as three fields of `RewardModelConfig`, each with specified types, (optional) default values, and (optional) validators to make sure that given values are valid.

Additionally,
+ `_target_` field tells Hydra that `reward_model_parser` is the configured function associated with this `RewardModelConfig`,
+ `_partial_` field tells Hydra whether the parsed object should be
  + (if `_partial_=True`) a `functools.partial` that combines the target function `reward_model_parser` with values of configured arguments, or
  + (if `_partial_=False`) the result of calling the target function `reward_model_parser` with values of configured arguments.

Since `_partial_=True`, the parsed object will be a callable with signalture:
```py
def parsed_reward_model(*, transition_model: TransitionModel) -> RewardModel: ...
```
To support better type checking & hints, we also manually write out this type as `RewardModelConfig.InstantiatedT`, which will be used as a type annotation in the final parsed full config (see [Argument Parsing](#argument-parsing)).

To summarize, the fields of a config class should consist of the following:
+ All configurable arguments (optionally with default values and validators);
+ `InstantiatedT`, a type annotation of the parsed object type;
+ `_target_`, specifying which function/class is being configurated;
+ (optionally) `_partial_`, specifying the way to apply values of configured aguments to the target (default is `False`).

</div>
</details>

## Algorithmic Code

### Model-Based Training in `ModelTrainer.fit(...)`

The main model-based training loop is handled by [`ModelTrainer.fit(...)`](./main.py#L390-508). The function
1. Prefills replay buffer via `ModelTrainer.fill_with_noise`, if prefilling isn't yet done.
2. Creates an iterator of experiences from model+policy interacting with environment, via `env_interact_with_model` (see [Environment Interface and Interaction](#environment-interface-and-interaction)).
3. Repeatedly fetches experience from this iterator (the `while`-loop [here](./main.py#L458-504)):
   1. Train model and policy on reply-buffer data every `train_interval` iterations;
      + Implemented at `ModelTrainer.train(...)`, calling into the [model learning](#model-learning) and [policy learning](#policy-learning) objects;
   2. Test policy performance every `test_interval` iterations;
   3. Save checkpoints every `checkpoint_interval` iterations;
   4. Append data to replay buffer.

### Environment Interface and Interaction

All environments conform to the [`AutoResetEnvBase`](./denoised_mdp/envs/abc.py#L166) interface.
+ It is mostly a extension to standard `gym.Env` with automatic resetting.
+ Notably, its `env.reset()` returns a tuple `(observation, info)` rather than just the observation for standard `gym.Env`. `info` objects from `env.reset()` and `env.step(...)` are also restricted to a specific class `AutoResetEnvBase.Info`, which contains useful data, including number of actual environment steps taken (which could be more than 1 due to `action_repeat` and `episode_length`) and the observation before an auotmatic reset.

+ To interact with and collect experiences from such environments, we provide [`denoised_mdp.envs.interaction.env_interact_random_actor`](./denoised_mdp/envs/interaction.py#L192) and [`denoised_mdp.envs.interaction.env_interact_with_model`](./denoised_mdp/envs/interaction.py#L280). They are generators of ineraction data [`EnvInteractData`](./denoised_mdp/envs/interaction.py#L49), which provides rich information of each interaction.

More details can be found in [this note](./denoised_mdp/envs/README.md).

### Replay Buffer

The replay buffer is implemented via the class [`ExperienceReplay`](./denoised_mdp/memory.py#L302).

+ It stores sequences of `(action, reward, next_observation_nonfirststep, next_observation)`, where
  + `next_observation_nonfirststep` shows whether `next_observation` is the observation from a environment reset (if so, this will be `False`).
  + environment reset is also stored in such a tuple, where `reward` is 0, and `action` is a fixed tensor specified when creating the replay buffer (usually just all zeros, following Dreamer).

+ Adding data to this buffer is done via `replay_buffer.append_reset(...)` and `replay_buffer.append_step(...)`.

+ Sampling random segments is done via `replay_buffer.sample(...)`.

+ `ExperienceReplay` is essentially a wrapper over an `Accessor` object, which is designed to store and sample sequences of arbitary tuples (or dictionaries) of `torch.Tensor`s. It handles storing, continuously writing dataset to a temporary directory on disk, loading from disk with a checksum-like verification, and efficient random batch sampling.
  <details>
  <summary>Expand for a description of <code>Accessor</code> implementation</summary>

  + Upon creating such an `Accessor` object, a pool of saver threads are also initialized, which keep writing data to disk. This is necessary for support pre-emption because writing upon preemption becomes too slow when dataset grows large.
  + [`BaseAccessor`](./denoised_mdp/memory.py#L65) implements most functionalities (including sampling and saving/loading), except for
    + `BaseAccessor.get_complete_data(self, idx: int)` for getting a complete sequence at index `idx`, and
    + `BaseAccessor._extend_complete_data(self, data)` for storing a complete sequence.
  + We use the `ListAccessor` implementation of `BaseAccessor`, which simply store sequences in a list. But one can easily add more custom `Accessor` variants if needed, by implementing the above two methods.
</details>

+ `ExperienceReplay` can quickly save all content to a checkpoint directory when needed (e.g., pre-emption). This is done by waiting on the `Accessor`'s saving and moving data from the temporary directory to the target directory.

### Agent

[`AgentBase`](./denoised_mdp/agents/base.py#L34) defines the interface for a model-based agent. Notably, it should at least a transition model and a policy (i.e., actor model). The agent is assumed to
+ be trained via some reconstruction-based objective, via [`AgentBase.train_reconstruct`](./denoised_mdp/agents/base.py#L65), which outputs [a `AgentBaseTrainOutput` object](./denoised_mdp/agents/base.py#L57);
+ support latent-space rollout, via [`AgentBase.imagine_ahead_noiseless`](./denoised_mdp/agents/base.py#L73), which outputs [a `ImagineOutput` object](./denoised_mdp/agents/base.py#L68);
+ update its latent representation based on observation, via [`AgentBase.posterior_rsample_one_step`](./denoised_mdp/agents/base.py#L81);
+ be able to act from the latent state, via [`AgentBase.act`](./denoised_mdp/agents/base.py#L93).

We provide [`DenoisedMDP`](denoised_mdp/agents/denoised_mdp.py#L181) agent.

### Model Learning

In general, any reconstruction-based algorithm can be implemented in the [`BaseModelLearning`](./denoised_mdp/agents/learning/model.py#L172) form, exposing `BaseModelLearning.train_step(...)` that should
1. run agents on a given batch of segments,
2. generatre [a `AgentBaseTrainOutput` object](./denoised_mdp/agents/base.py#L57) from [`AgentBase.train_reconstruct`](./denoised_mdp/agents/base.py#L65),
3. compute and update the model parameters with a dictionary of losses.

Our Denoised MDP objective is an extension of the variational model-fitting objective. It is implemented as [`VariationalModelLearning`](./denoised_mdp/agents/learning/model.py#L182), with ability to configurate different KL terms for different latent spaces.

In `ModelTrainer.fit(...)` loop, `BaseModelLearning.train_step` gets called in [`ModelTrainer.train(...)`](./main.py#L236-239).

### Policy Learning

Policy learning algorithms are implemented as subclasses of [`BasePolicyLearning`](./denoised_mdp/agents/learning/policy.py#L76), which is also a really general interface.

We provide two implementations:
1. [`DynamicsBackpropagateActorCritic`](./denoised_mdp/agents/learning/policy.py#L109): Dreamer-style latent imagination and backpropagation of $\lambda$-return estimates to actor parameters;
2. [`SoftActorCritic`](./denoised_mdp/agents/learning/policy.py#L285): Soft Actor-Critic (SAC) on the Denoised MDP transition data (converted from raw transition data via the learned posterior encoder model).

In particular `SoftActorCritic` is implemented with the [`GeneralRLLearning`](./denoised_mdp/agents/learning/policy.py#L235) class, which handles the conversion to Denoised MDP transitions. More genereal-purpose policy optimization algorithms can be easily implemented as well with `GeneralRLLearning`.

In [`./denoised_mdp/agents/learning/policy.py`](./denoised_mdp/agents/learning/policy.py) we also provide many utility funtions, including value/Q-function creation, Polyak update, etc., for easy implementation of other policy optimization algorithms.

In `ModelTrainer.fit(...)` loop, `BaseModelLearning.train_step` gets called in [`ModelTrainer.train(...)`](./main.py#L246).


## Argument Parsing

`to_config_and_instantiate()` defined in [`config.py`](./config.py) handles all config parsing by calling into Hydra. It returns a 2-tuple `(config, instantiated_config)`. Both are type-annotated `attrs` classes (possibly with nested such classes) that specify the entire code behavior. The main difference is in how they handled [configurable objects](#a-common-pattern-configurable-objects):
+ [`Config`](./config.py#L48) contains only configs of those objects.
+ [`InstantiatedConfig`](./config.py#L148) is created directly from `Config`, and converts the configs to parsed versions of these objects.

For the example shown in [expanded section of Configurable Objects](#a-common-pattern-configurable-objects), where `reward_model_parser` function is accompanied with config class `RewardModelConfig`.
+ `Config` will contain a `RewardModelConfig` object.
+ `InstantiatedConfig`  will contain a variant of `reward_model_parser` that is aware of the config values (a `functools.partial` function binding it with the values in this case).

Hence, `Config` contains only simple primitives, can be converted directly to a YAML, and is useful for logging and writing to disk. `InstantiatedConfig` contains objects/components that are used for actually running the algorithm.

## Pre-Emption Mechanism

`SIGUSR1` is our mechanism to pre-empt. Unpon SIGUSR1, we catch the signal, start checkpointing (incl. replay buffer), and finally kill the training job. When we start training, the code first checks if the output directory contains some resumable state, and resume from there if so.

Please see our [note in code](config.py#L153-166) for details.
