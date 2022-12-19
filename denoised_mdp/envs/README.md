# Environments

The environments used in this repository follows mostly an extension of the typical OpenAI Gym API, except for a few differences:
+ All environments are subclasses of `EnvBase` defined in [`./abc.py`](./abc.py), which supports a bit more functionalities than the Gym API:
  + `env.reset()` returns `(observation, info)`
  + `info` returned by `env.reset()` and `env.step()` follow `EnvBase.Info`
  + Explicit observation format choices (must be images), defined in `EnvBase.ObsOutputKind`
  + Various properties (e.g., `action_repeat`, `batch_shape`) and auxiliary methods.
+ We generally use environments that automatically resets (i.e., resetting when done), because in planning the last observation is useless. However, that observation is still useful for fitting the model. We represent such environments via the base class [`AutoResetEnvBase`](./abc.py), which extends `EnvBase` and handles such intricacies.
  + `AutoResetEnvBase` generally only needs the first `.reset()` call. Afterwards, if a `.step(...)` leads to `done=True`, an internal `.reset()` is automatically performed, returning the first observation of the new trajectory.
  + [`utils.make_batched_auto_reset_env(...)`](./utils.py#L20) is a helper for creating (possibly batched) `AutoResetEnvBase` instances.
  + `AutoResetEnvBase` instances yield info objects that follow `AutoResetEnvBase.Info`, which extends `EnvBase.Info` by including the last observation before resetting. This is helpful when filling in replay buffers.

Environments are created in [`EnvKind.create(...)`](./__init__.py#L22), which uses the specified environment kind to call into specific creation functions.

## Interacting with Environments

There are generally two modes of interaction:
1. Random actor (for prefilling replay buffers)
2. Learned actor + exploration noise (for most of training)

We use a unified interface to implement both strategies at [`./interaction.py`](./interaction.py). Generally, the entry points are [`env_interact_random_actor`](./interaction.py#L195) and [`env_interact_with_model`](./interaction.py#L283) which instantiates generators of [`EnvInteractData`](./interaction.py#L52), a named tuple that contains all necessary information about a step taken. Futhermore, they are implemented as applying two different [`Interactor`](./interaction.py#L102) classes to the same base interaction function [`env_interact`](./interaction.py#L116). Thus new interaction modes can be easily added by simply defining new [`Interactor`](./interaction.py#L102) classes.

Interested readers can read the source file for more details.
