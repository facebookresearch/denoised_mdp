# Environments

The environments used in this repository follows mostly an extension of the typical OpenAI Gym API, except for a few differences:
+ All environments are subclasses of `EnvBase` defined in [`./abc.py`](./abc.py), which supports a bit more functionalities than the Gym API:
  + `env.reset()` returns `(observation, info)`
  + `info` returned by `env.reset()` and `env.step()` follow `EnvBase.Info`
  + Explicit one of three return format choices, defined in `EnvBase.ObsOutputKind`
  + Various properties (e.g., `action_repeat`, `batch_shape`) and auxiliary methods.
+ We generally use environments that automatically resets (i.e., resetting when done), because in planning the last observation is useless. However, that observation is still useful for fitting the model. We represent such environments via the base class [`AutoResetEnvBase`](./abc.py), which extends `EnvBase` and handles such intricacies.
  + `AutoResetEnvBase` generally only needs the first `.reset()` call. Afterwards, if a `.step(...)` leads to `done=True`, an internal `.reset()` is automatically performed, returning the first observation of the new trajectory.
  + [`utils.make_batched_auto_reset_env(...)`](./utils.py) is a helper for creating (possibly batched) `AutoResetEnvBase` instances.
  + `AutoResetEnvBase` instances yield info objects that follow `AutoResetEnvBase.Info`, which extends `EnvBase.Info` by including the last observation before resetting. This is helpful when filling in replay buffers.

## Interacting with Environments

There are generally two modes of interaction:
1. Random actor (for prefilling replay buffers)
2. Learned actor + exploration noise (for most of training)

We use a unified interface to implement both strategies at [`./interaction.py`](./interaction.py). Generally, the entry points are `env_interact_random_actor` and `env_interact_with_model` which instantiates generators of `InteractData`, a named tuple that contains all necessary information about a step taken.

Interested readers can read the source file for more details.
