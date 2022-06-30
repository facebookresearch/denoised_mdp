# DeepMind Control Suite with Distractors

Contents under this folder are modified from [the public repository of Deep Bisimulation for Control (DBC)](https://github.com/facebookresearch/deep_bisim4control), published under CC-by-NC 4.0 license.

Notable modifications are:
+ [`./natural_imgsource.py`](./natural_imgsource.py) has better RNG management, and a typed API.
+ [`./wrappers.py`](./wrappers.py) supports more noise types (e.g., sensor noise, camera jittering).
+ [`./local_dm_control_suite`](./local_dm_control_suite/) implements sensor noise via [`NoisyMuJoCoPhysics`](./local_dm_control_suite/common/__init__.py).

We thank authors of DBC for releasing their code.
