# Denoised MDPs: Learning World Models Better Than The World Itself

**[Tongzhou Wang](https://ssnl.github.io/), [Simon S. Du](https://simonshaoleidu.com/), [Antonio Torralba](https://web.mit.edu/torralba/www/), [Phillip Isola](https://web.mit.edu/phillipi/), [Amy Zhang](https://amyzhang.github.io/), [Yuandong Tian](https://yuandong-tian.com/)**

We provide a PyTorch implementation of [Denoised MDPs: Learning World Models Better Than The World Itself](https://ssnl.github.io/denoised_mdp), published in ICML 2022.

+ [arXiv](https://arxiv.org/abs/2206.15477)
+ [Project Page](https://ssnl.github.io/denoised_mdp)

(We also provide a PyTorch implementation of [Dreamer](https://arxiv.org/abs/1912.01603) that is carefully written and verified to reproduce results. See [here](#reproducing-dreamer) for usages.)

The raw real world is noisy. How can reinforcement learning agent successfully learn with such raw data, where signals can be strongly entangled with noises? Denoised MDP characterizes information into four distinct types, based on controllability and relation with rewards, and proposes to extract a state representation space containing only information both **controllable** and **reward-relevant**. Under this view, several prior works can be seen as insufficiently removing noisy information.

To properly extract *only* the useful signal, Denoised MDP considers novel factorized MDP transition structures, where **signal** representation and **noise** representation are separated into distinct latent spaces. The state abstraction (i.e., representation learning) problem is turned into a regularized model fitting problem: fitting the factorized forward model to collected trajectories, while requiring the **signal** latents to be minimally informative of the raw observations.

The resulting variational formulation (derivation in paper) successfully disentangles a variety of noise types (and also noiseless settings), outperforming baseline methods that often can only do well for certain particular noise types.

## Visualizations

For environments with distinct types of noises, we visualize latent factorization idenfitied by Denoised MDP, and other baseline methods. **Only** Denoised MDP successfully disentangle signal from noises across **all environments**.

+ **Task:** Press green button to shift TV hue to green ([a RoboDesk variant](https://github.com/SsnL/robodesk)). <br>
  **True Signal:** Robot joint position, TV green-ness, Green light on desk. <br>
  **True Noise:** Lighting, Camera, TV content, Imperfect sensor.

  https://user-images.githubusercontent.com/5674597/173155667-d4bcc7af-1f12-4ba3-a733-ef9d5f631c96.mp4
<!--
+ **Task:** Move the half cheetah robot forward.  <br>
  **True Signal:** Robot joint position. <br>
  **True Noise:** None.

  https://user-images.githubusercontent.com/5674597/173155678-4c9bf104-1e15-48e0-882e-b0d097c383a8.mp4 -->

<!--
+ **Task:** Make the walker robot move forward while standing up. <br>
  **True Signal:** Robot joint position. <br>
  **True Noise:** Background.

  https://user-images.githubusercontent.com/5674597/173155689-cfe0a504-44e5-4d48-b519-196f4d2a4554.mp4 -->

+ **Task:** Make reacher robot touch the target red object. <br>
  **True Signal:** Robot joint position, Target location. <br>
  **True Noise:** Background.

  https://user-images.githubusercontent.com/5674597/173155695-d85f5866-0bbe-4451-9898-d032d3bcf51c.mp4

+ **Task:** Make the walker robot move forward when sensor readings are noisily affected by background images.  <br>
  **True Signal:** Robot joint position. <br>
  **True Noise:** Background, Imperfect sensor.

  https://user-images.githubusercontent.com/5674597/173155705-02f098f7-dca8-4022-995a-57b1aa854935.mp4

+ **Task:** Move half cheetah robot forward when camera is shaky.  <br>
  **True Signal:** Robot joint position. <br>
  **True Noise:** Background, Camera.

  https://user-images.githubusercontent.com/5674597/173155710-285d0088-b3a3-42e7-9127-a177e4f6b955.mp4



## Requirements

The code has been tested on
+ CUDA 11 with NVIDIA RTX Titan, NVIDIA 2080Ti, and NVIDIA Titan XP,
+ `mujoco=2.2.0` with `egl` renderer.

Software dependencices (also in [`requirements.txt`](./requirements.txt)):

```
torch>=1.9.0
tqdm
numpy>=1.17.0
PIL
tensorboardX>=2.5
attrs>=21.4.0
hydra-core==1.2.0
omegaconf==2.2.1
mujoco
dm_control
```

## Environments

The code supports the following environments:

| `kind` | `spec` | Description |
| :-------- | :---------- | :-- |
| `robodesk` | `${TASK_NAME}` or <br> `${TASK_NAME}_noisy` <br><br> (e.g., `tv_green_hue_noisy`) | RoboDesk environment (`96x96` resolution) with a diverse set of distractors (when using `${TASK_NAME}_noisy` variant). The distractors are implemented and descriped in details at [this RoboDesk fork](https://github.com/SsnL/robodesk). |
| `dmc`      | `${DOMAIN_NAME}_${TASK_NAME}_${VARIANT}` <br> with `VARIANT` being one of `[noiseless, video_background, video_background_noisy_sensor, video_background_camera_jitter]` <br><br> (e.g., `cheetah_run_video_background_camera_jitter`)  | DeepMind Control (DMC) environment (`64x64` resolution) with four possible variants, representing different types of noises. |

In the paper, we used the following 13 environments, all run with 1000 max episode length and action repeat of 2:

| `kind` | `spec` |
| :-------- | :---------- |
| `robodesk` | `tv_green_hue_noisy` |
| `dmc`      | `${DOMAIN_NAME}_${TASK_NAME} in [cheetah_run, walker_walk, reacher_easy]` each with all 4 `VARIANT` options |

**NOTE:** All noisy environments require the `driving_car` class of the Kinetics-400 training dataset. Some instructions for downloading the dataset can be found [here](https://github.com/Showmax/kinetics-downloader). After downloading, you may either place it under `~kinetics/070618/400` (so that the videos are `~kinetics/070618/400/train/driving_car/*.mp4`) or specify `KINETICS_DIR` environment variatble (so that the videos are `${KINETICS_DIR}/train/driving_car/*.mp4`).

## Training and Evaluation

```sh
env CUDA_VISIBLE_DEVICES=0 \              # GPU ID for training
    EGL_DEVICE_ID=0 \                     # GPU ID for rendering
    KINETICS_DIR=/path/to/kinectics/ \    # Videos for noisy env
    python main.py \
        env.kind=robodesk \               # Env kind
        env.spec=tv_green_hue_noisy \     # Env spec
        learning.model_learning.kl.alpha=2 \       # alpha, weight of the KL terms
        learning.model_learning.kl.beta_y=0.125 \  # beta, smaller => stronger regularization
        learning.model_learning.kl.beta_z=0.125 \  # beta, smaller => stronger regularization
        seed=12 \                         # Seed
        output_folder=subdir/for/output/  # [Optional] subdirectory under `./results`
                                          # for storing outputs. If not given, a folder name will
                                          # be automatically constructed with information from
                                          # given config
```

Hyperparameter choice (see also Appendix A.2 for more details):
+ `alpha` parameter is selected proportional to the size of observation. For DMC (`64x64x3` observations), we use `alpha=1`. For RoboDesk (`96x96x3` observations), we use `alpha=2`.
+ `beta` parameter (for `y` and `z` component) controls the regularization strength, and should be set in `(0, 1)`. Noisier environments benefit from a smaller value.


Default behaviors:
+ Train the *Figure 2b* Denoised MDP variant over `10^6` environment steps, with `5000` steps prefilling the replay buffer, and then training for `100` iterations for every `100` steps.
+ Optimize policy bybackpropagation through dynamics (Dreamer-style). One can switch to Soft Actor-Critic via specifying `learning/policy_learning=sac` (note the `/` rather than `.`).
+ Evaluate for `10` episodes every `10000` steps.
+ Visualize for `3` episodes (both full reconstruction and with noise latent fixed) every `20000` steps.

We use [Hydra](https://hydra.cc/) to handle argument specification. You can use Hydra's overriding syntax to specify all sorts of config options. See [`config.py`](./config.py) for the complete set of options. Additionally, one can also check the `config.yaml` file generated in output directory for all options.

### Reproducing `Dreamer`
When `y` and `z` latent spaces are completely turned off (i.e., empty), the code essentially is Dreamer. This can be done by setting
```sh
learning.model.transition.x.belief_size=200 \  # give `x` the dimensionality specified in Dreamer paper
learning.model.transition.x.state_size=30 \
learning.model.transition.y.belief_size=0 \
learning.model.transition.y.state_size=0 \
learning.model.transition.z.belief_size=0 \
learning.model.transition.z.state_size=0 \
```

## Code Structure

To facilitate easier parsing and usage of this repository, we provide a detailed note on how our code is structured [here](./CODE_STRUCTURE.md).

## Pre-emption

Upon receiving `SIGUSR1`, the provided code starts writing all necessary states (including replay buffer) into a folder under the output directory (usually taking up to 10 minutes), and exits naturally afterwards. When the code is run with the same output directory, it continues from that state (and deletes the saved state). This may be particularly useful if you are running on a shared cluster.

## Citation

Tongzhou Wang, Simon S. Du, Antonio Torralba, Phillip Isola, Amy Zhang, Yuandong Tian. "Denoised MDPs: Learning World Models Better Than The World Itself" International Conference on Machine Learning. 2022.

```
@inproceedings{wang2022denoisedmdps,
  title={Denoised MDPs: Learning World Models Better Than The World Itself},
  author={Wang, Tongzhou and Du, Simon S. and Torralba, Antonio and Isola, Phillip and Zhang, Amy and Tian, Yuandong},
  booktitle={International Conference on Machine Learning},
  organization={PMLR},
  year={2022}
}
```

If you find the RoboDesk distractor options (see [this repository](https://github.com/SsnL/robodesk) for more options and details) useful for your research, please also cite the following:
<details>
 <summary>Click to show RoboDesk distractor <code>bibtex</code>!</summary>

```
@misc{wang2022robodeskdistractor,
  author = {Tongzhou Wang},
  title = {RoboDesk with A Diverse Set of Distractors},
  year = {2022},
  howpublished = {\url{https://github.com/SsnL/robodesk}},
}

@misc{kannan2021robodesk,
  author = {Harini Kannan and Danijar Hafner and Chelsea Finn and Dumitru Erhan},
  title = {RoboDesk: A Multi-Task Reinforcement Learning Benchmark},
  year = {2021},
  howpublished = {\url{https://github.com/google-research/robodesk}},
}
```
</details>

## Questions

For questions about the code provided in this repository, please open an GitHub issue.

For questions about the paper, please contact Tongzhou Wang (`tongzhou _AT_ mit _DOT_ edu`).

## License
This repo is under CC BY-NC 4.0. Please check [LICENSE](./LICENSE) file.
