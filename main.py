# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import *

import os

os.environ['MUJOCO_GL'] = 'egl'

import sys

import glob
import itertools
import signal
import logging
import socket
from collections import defaultdict

import hydra
from omegaconf import DictConfig, OmegaConf

from tqdm.auto import tqdm
import numpy as np
import torch
import torch.backends.cudnn
from torch import nn, optim
from PIL import Image

from denoised_mdp.envs import (
    AutoResetEnvBase, split_seed, env_interact_random_actor, env_interact_with_model,
)
from denoised_mdp.envs.interaction import EnvInteractData, ModelActorState
from denoised_mdp.memory import ExperienceReplay
from denoised_mdp.agents import (
    BaseModelLearning,
    BasePolicyLearning,
    DynamicsBackpropagateActorCritic,
    DenoisedMDP,
)
from denoised_mdp.io import write_video, make_grid
from denoised_mdp import utils
from tensorboardX import SummaryWriter


from config import InstantiatedConfig, to_config_and_instantiate


class ModelTrainer(object):
    cfg: InstantiatedConfig

    def __init__(self, cfg: InstantiatedConfig, summary_writer: SummaryWriter, data_collect_env_seed: np.random.SeedSequence,
                 replay_buffer_seed: np.random.SeedSequence, test_env_seed: np.random.SeedSequence) -> None:
        super().__init__()
        self.cfg = cfg
        self.summary_writer = summary_writer
        # env
        self.data_collect_env: AutoResetEnvBase = cfg.env(seed=data_collect_env_seed, batch_shape=(), for_storage=True)
        self.test_env_seed = test_env_seed
        self.test_env: AutoResetEnvBase = cfg.env(seed=test_env_seed, batch_shape=(cfg.learning.test.num_episodes,),
                                                  for_storage=False)
        self.world_model: DenoisedMDP = cfg.learning.model(env=self.data_collect_env).to(cfg.device)

        # optimizers
        def optimizer_ctor(parameters: Iterable[nn.Parameter], lr: float) -> torch.optim.Optimizer:
            return optim.Adam(
                parameters,
                lr=lr,
                betas=(0.9, 0.999),
                eps=cfg.learning.optimization.adam_eps,
            )

        # losses
        self.model_learning: BaseModelLearning = cfg.learning.model_learning(
            world_model=self.world_model,
            optimizer_ctor=optimizer_ctor,
        )
        self.policy_learning: BasePolicyLearning = cfg.learning.policy_learning(
            world_model=self.world_model,
            optimizer_ctor=optimizer_ctor,
        )
        # replay buffer
        self.replay_buffer: ExperienceReplay = cfg.learning.experience_replay(
            self.data_collect_env, self.world_model.action_from_init_latent_state(device='cpu'),
            seed=replay_buffer_seed)
        # metrics
        self.train_metrics: DefaultDict[str, List[Union[float, np.ndarray]]] = defaultdict(list)
        self.train_metrics.update(
            num_env_steps=[],
            train_rewards=[],
        )
        self.test_metrics: Dict[str, List[Union[float, np.ndarray]]] = dict(
            num_env_steps=[],
            test_rewards=[],
        )
        # env collect init
        self.num_env_steps = 0
        self.train_info: Dict[str, Dict[str, Union[int, float]]] = dict(
            last_train=dict(
                num_env_steps=-np.inf,
                total_complete_episodes_reward=0,
                total_complete_episodes=0,
                current_episode_reward=0,
            ),
            last_test=dict(
                num_env_steps=-np.inf,
            ),
            last_visualize=dict(
                num_env_steps=-np.inf,
            ),
            last_checkpoint=dict(
                num_env_steps=-np.inf,
            ),
        )
        if self.resume_from_saved_state_if_exists():
            logging.info('Resume from previous checkpoint!')

    @property
    def can_save_state(self) -> bool:
        return not self.replay_buffer.has_incomplete_episode  # replay buffer can not save incomplete episodes

    def state_dict(self):
        return dict(
            train_info=self.train_info,
            num_env_steps=self.num_env_steps,
            train_metrics=self.train_metrics,
            test_metrics=self.test_metrics,
            data_collect_env_random_state=self.data_collect_env.get_random_state(),
            world_model=self.world_model.state_dict(),
            model_learning=self.model_learning.state_dict(),
            policy_learning=self.policy_learning.state_dict(),
        )

    def save_resumable_state_if_possible(self) -> bool:  # returns whether successful
        num_env_steps_repr = str(self.num_env_steps).zfill(len(str(self.cfg.learning.exploration.total_steps)))

        target_dir = os.path.join(self.cfg.preempt_ckpt_dir, num_env_steps_repr)

        if self.can_save_state and not os.path.exists(target_dir):

            logging.info('')
            target_tmp_dir = os.path.join(self.cfg.tmp_preempt_ckpt_dir, num_env_steps_repr)
            logging.info(f'Start saving to TEMPORARY checkpoint dir... :\n\t{self.cfg.fmtdir(target_tmp_dir)}')

            self.replay_buffer.save_assert_all_complete(
                move_to_dir=os.path.join(target_tmp_dir, 'replay_buffer'))
            logging.info(f'Saved replay buffer to TEMPORARY checkpoint dir:\n\t{self.cfg.fmtdir(target_tmp_dir)}')

            torch.save(self.state_dict(), os.path.join(target_tmp_dir, 'checkpoint.pth'))
            logging.info(f'Saved others to TEMPORARY checkpoint dir:\n\t{self.cfg.fmtdir(target_tmp_dir)}')

            logging.info(f'Finished saving to TEMPORARY checkpoint dir:\n\t{self.cfg.fmtdir(target_tmp_dir)}')

            logging.info(f'Move to checkpoint dir... :\n\t{self.cfg.fmtdir(target_dir)}')
            os.makedirs(self.cfg.preempt_ckpt_dir, exist_ok=True)
            os.rename(target_tmp_dir, target_dir)
            logging.info(f'Moved to checkpoint dir:\n\t{self.cfg.fmtdir(target_dir)}')
            logging.info('')
            return True
        return False

    def resume_from_saved_state_if_exists(self, remove_after_loading: bool = True) -> bool:  # returns if resuming
        saved_dirs = glob.glob(os.path.join(glob.escape(self.cfg.preempt_ckpt_dir), '*'))
        if len(saved_dirs) == 0:
            return False
        latest: Tuple[int, Optional[str]] = (-1, None)
        for dir in saved_dirs:
            num_env_steps = int(os.path.basename(dir))
            if num_env_steps >= latest[0]:
                latest = num_env_steps, dir
        assert latest[1] is not None
        logging.info('')
        load_dir: str = latest[1]
        logging.info(f'Start loading from {self.cfg.fmtdir(load_dir)}')

        self.replay_buffer.load_from_folder(os.path.join(load_dir, 'replay_buffer'))
        checkpoint: Dict[str, Any] = torch.load(os.path.join(load_dir, 'checkpoint.pth'), map_location='cpu')
        logging.info(f'Loaded replay buffer from: {self.cfg.fmtdir(load_dir)}')

        self.train_info = checkpoint['train_info']
        self.num_env_steps = checkpoint['num_env_steps']
        self.train_metrics = checkpoint['train_metrics']
        self.test_metrics = checkpoint['test_metrics']
        self.data_collect_env.set_random_state(checkpoint['data_collect_env_random_state'])

        if 'model_optimizer' in checkpoint.keys():
            # BC old
            checkpoint['model_learning'] = dict(
                module=checkpoint['model_learning_loss'],
                optimizers=dict(
                    model=checkpoint['model_optimizer'],
                ),
            )
            assert isinstance(self.policy_learning, DynamicsBackpropagateActorCritic)
            policy_learning_module_state_dict: Dict[str, torch.Tensor] = checkpoint['actor_critic_loss']
            for k in list(checkpoint['world_model'].keys()):
                if k.startswith('value_model.'):
                    policy_learning_module_state_dict[k] = checkpoint['world_model'].pop(k)
            checkpoint['policy_learning'] = dict(
                module=policy_learning_module_state_dict,
                optimizers=dict(
                    actor=checkpoint['actor_optimizer'],
                    value=checkpoint['value_optimizer'],
                ),
            )

        self.world_model.load_state_dict(checkpoint['world_model'])
        self.model_learning.load_state_dict(checkpoint['model_learning'])
        self.policy_learning.load_state_dict(checkpoint['policy_learning'])
        logging.info(f'Loaded others from: {self.cfg.fmtdir(load_dir)}')

        logging.info(f'Loaded from {self.cfg.fmtdir(load_dir)}')
        assert self.replay_buffer.num_steps == self.num_env_steps

        if remove_after_loading and utils.rm_if_exists(load_dir, maybe_dir=True):
            logging.info(f'Deleted {self.cfg.fmtdir(load_dir)}')

        logging.info('')
        return True

    def train(self, num_iterations) -> Dict[str, np.ndarray]:
        losses: DefaultDict[str, List[float]] = defaultdict(list)
        for ii in tqdm(range(num_iterations), desc='train steps', disable=(num_iterations < 50)):
            update_model = (ii % self.cfg.learning.optimization.model_every) == 0
            update_pi = (ii % self.cfg.learning.optimization.policy_every) == 0
            if not update_model and not update_pi:
                continue

            # Draw sequence chunks {(o_t, a_t, r_t+1, terminal_t+1)} ~ D uniformly (including terminal flags)
            data = self.replay_buffer.sample(self.cfg.learning.batch_size, self.cfg.learning.chunk_length)
            data = data.to(self.cfg.device)

            # model
            train_out, model_loss_terms = self.model_learning.train_step(
                data, self.world_model,
                grad_update=update_model,
            )
            for k, l in model_loss_terms.items():
                losses[f"model/{k}"].append(float(l))

            # policy
            if not update_pi:
                continue
            policy_loss_terms = self.policy_learning.train_step(data, train_out, self.world_model)
            for k, l in policy_loss_terms.items():
                losses[f"policy/{k}"].append(float(l))

        return {k: np.asarray(v, dtype=np.float32) for k, v in losses.items()}

    def fill_with_noise(self, num_steps):
        if num_steps == 0:
            return

        assert len(self.data_collect_env.batch_shape) == 0

        interaction = env_interact_random_actor(
            self.data_collect_env, num_steps, tqdm_desc='Prefill with noise')

        for interact_data in interaction:
            if bool(interact_data.is_first_step):
                self.replay_buffer.append_reset(
                    observation=interact_data.observation,
                )

            self.replay_buffer.append_step(
                action=interact_data.action,
                reward=interact_data.reward,
                next_observation=(  # Don't put comma at the end of next line...
                    interact_data.observation_before_reset if interact_data.done else interact_data.next_observation
                ),
                done=interact_data.done,
            )

            self.num_env_steps += interact_data.num_new_steps.sum().item()
        self.replay_buffer.mark_previous_episode_as_complete_if_needed()
        assert interact_data.num_steps == num_steps

    def test(self, visualize_file_suffix: Optional[str], visualize_num_episodes: int) -> np.ndarray:
        torch.cuda.empty_cache()
        first_episode_total_rewards: np.ndarray = np.zeros(self.test_env.batch_shape, dtype=np.float32)
        assert len(self.test_env.batch_shape) == 1

        do_visualize = visualize_file_suffix is not None and visualize_num_episodes > 0

        def obs_to_image(obs: torch.Tensor):
            assert do_visualize
            if obs.shape[-3] == 1:
                return obs.expand(*obs.shape[:-3], 3, *obs.shape[-2:])
            else:
                assert obs.shape[-3] % 3 == 0
                return obs.reshape(*obs.shape[:-3], 3, -1, obs.shape[-1])

        def save_image(uint8_image: torch.Tensor, fp: str):
            im = Image.fromarray(uint8_image.permute(1, 2, 0).numpy())
            im.save(fp)

        video_frames: List[torch.Tensor] = []
        video_frames_noise: List[torch.Tensor] = []
        viz_saved_t0: Any = None

        @torch.no_grad()
        def collect_frames(interact_data: 'EnvInteractData[ModelActorState]'):
            # Collect real vs. predicted frames for video
            nonlocal viz_saved_t0
            posterior_latent_state: DenoisedMDP.LatentState = \
                interact_data.state_after_step.flat_model_latent_state_before_next_observation

            posterior_latent_state = posterior_latent_state.narrow(0, 0, visualize_num_episodes)

            reconstruction: torch.Tensor = self.world_model.observation_model(
                posterior_latent_state).mean
            video_frames.append(
                make_grid(
                    torch.cat(
                        [
                            obs_to_image(interact_data.observation[:visualize_num_episodes]),
                            obs_to_image(reconstruction.cpu()),
                        ],
                        dim=-1,
                    ).add_(0.5),
                    nrow=self.test_env.batch_shape[-1],
                ).mul_(255).clamp_(0, 255).to(torch.uint8)
            )

            if not self.world_model.transition_model.only_x:
                if viz_saved_t0 is None:
                    viz_saved_t0 = posterior_latent_state  # save t0 latents

                assert isinstance(viz_saved_t0, DenoisedMDP.LatentState)
                reconstruction: torch.Tensor = self.world_model.observation_model(
                    posterior_latent_state.replace(
                        y=viz_saved_t0,
                        z=viz_saved_t0,
                    ),
                ).mean
                video_frames_noise.append(
                    make_grid(
                        torch.cat(
                            [
                                obs_to_image(interact_data.observation[:visualize_num_episodes]),
                                obs_to_image(reconstruction.cpu()),
                            ],
                            dim=-1,
                        ).add_(0.5),
                        nrow=self.test_env.batch_shape[-1],
                    ).mul_(255).clamp_(0, 255).to(torch.uint8)
                )

        @torch.no_grad()
        def output_video():
            vis_dir = os.path.join(self.cfg.output_dir, 'visualization')
            os.makedirs(vis_dir, exist_ok=True)
            write_video(video_frames, f"test_episode_{visualize_file_suffix}", vis_dir)  # Lossy compression
            save_image(
                torch.as_tensor(video_frames[-1]),
                os.path.join(vis_dir, f"test_episode_{visualize_file_suffix}.png"),
            )
            if len(video_frames_noise):
                write_video(video_frames_noise, f"test_episode_noise_{visualize_file_suffix}", vis_dir)  # Lossy compression
                save_image(
                    torch.as_tensor(video_frames_noise[-1]),
                    os.path.join(vis_dir, f"test_episode_noise_{visualize_file_suffix}.png"),
                )
            logging.info('Saved visualization.')

        # actual testing begins

        self.test_env.seed(self.test_env_seed)
        interaction = env_interact_with_model(self.test_env, self.world_model, self.test_env.max_episode_length,
                                              train=False, tqdm_desc='Test')

        interact_data: 'EnvInteractData[ModelActorState]'
        for interact_data in interaction:
            assert len(interact_data.batch_shape) == 1
            first_episode_total_rewards += np.asarray(
                interact_data.reward * (interact_data.num_episodes == 0), dtype=np.float32)

            if do_visualize:
                collect_frames(interact_data)

        if do_visualize:
            output_video()

        assert torch.logical_or(interact_data.num_episodes > 0, interact_data.done).all()
        torch.cuda.empty_cache()
        return first_episode_total_rewards

    def fit(self):
        # prefill
        self.fill_with_noise(max(0, self.cfg.learning.exploration.prefill_steps - self.num_env_steps))
        # train-test
        # this will be a huge loop over all collected training data, where we update metrics whenever we train
        # or test
        explore_actor_kwargs = dict(
            explore=True,
            action_noise_stddev=self.cfg.learning.exploration.action_noise,
        )
        train_data_iter = env_interact_with_model(
            self.data_collect_env, self.world_model, self.cfg.learning.exploration.total_steps - self.num_env_steps,
            actor_kwargs=explore_actor_kwargs, train=False, tqdm_desc=None)

        def train():
            # Wrapper fof `self.train` with additional logging and metric tracking
            logging.info(f'num_env_steps={self.num_env_steps}: train')
            metrics = self.train(self.cfg.learning.optimization.train_iterations)

            self.train_metrics['num_env_steps'].append(self.num_env_steps)
            if self.train_info['last_train']['total_complete_episodes'] > 0:
                self.train_metrics['train_rewards'].append(
                    self.train_info['last_train']['total_complete_episodes_reward'] / self.train_info['last_train']['total_complete_episodes'])
            else:
                self.train_metrics['train_rewards'].append(0)
            self.summary_writer.add_scalar("train_reward", self.train_metrics["train_rewards"][-1], self.num_env_steps)
            for k, v in metrics.items():
                self.train_metrics[k].append(v)
                self.summary_writer.add_scalar(k, v[-1], self.num_env_steps)

            self.train_info['last_train'].update(
                num_env_steps=self.num_env_steps,
                total_train_reward=0,
                total_complete_episodes_reward=0,
                total_complete_episodes=0,
                current_episode_reward=self.train_info['last_train']['current_episode_reward'],
            )

        def test(force_visualize=False):
            # Wrapper fof `self.test` with additional logging and metric tracking
            logging.info(f'num_env_steps={self.num_env_steps}: test')
            do_visualize = force_visualize
            do_visualize |= (self.num_env_steps - self.train_info['last_visualize']['num_env_steps']) >= self.cfg.learning.test.visualize_interval
            test_rewards = self.test(
                str(self.num_env_steps).zfill(len(str(self.cfg.learning.exploration.total_steps))) if do_visualize else None,
                visualize_num_episodes=self.cfg.learning.test.visualize_num_episodes,
            )

            self.train_info['last_test']['num_env_steps'] = self.num_env_steps
            if do_visualize:
                self.train_info['last_visualize']['num_env_steps'] = self.num_env_steps
            self.test_metrics['num_env_steps'].append(self.num_env_steps)
            log_info = dict(
                num_env_steps=self.num_env_steps,
            )
            self.test_metrics['test_rewards'].append(test_rewards)
            test_reward = test_rewards.mean()
            self.summary_writer.add_scalar(f'test_reward', test_reward, self.num_env_steps)
            logging.info(f'num_env_steps={self.num_env_steps}'.ljust(30) + f'  test_reward={float(test_reward):4g}')

        def save():
            # Saving latest model
            logging.info(f'num_steps={self.num_env_steps}: checkpoint')
            self.train_info['last_checkpoint']['num_env_steps'] = self.num_env_steps
            step_suffix = str(self.num_env_steps).zfill(len(str(self.cfg.learning.exploration.total_steps)))
            torch.save(self.state_dict(), os.path.join(self.cfg.output_dir, f'checkpoint_{step_suffix}.pth'))

        test_update_to_date = save_up_to_date = False
        while self.num_env_steps < self.cfg.learning.exploration.total_steps:
            test_update_to_date = save_up_to_date = False

            if self.cfg.received_SIGUSR1 and self.save_resumable_state_if_possible():
                sys.exit(signal.SIGINT)

            if (self.num_env_steps - self.train_info['last_train']['num_env_steps']) >= self.cfg.learning.optimization.train_interval:
                train()

            if (self.num_env_steps - self.train_info['last_test']['num_env_steps']) >= self.cfg.learning.test.test_interval:
                test()
                test_update_to_date = True

            if (self.num_env_steps - self.train_info['last_checkpoint']['num_env_steps']) >= self.cfg.learning.checkpoint_interval:
                save()
                save_up_to_date = True

            interact_data = next(train_data_iter)
            assert len(interact_data.batch_shape) == 0

            if bool(interact_data.is_first_step):
                self.replay_buffer.append_reset(
                    observation=interact_data.observation,
                )

            self.replay_buffer.append_step(
                action=interact_data.action,
                reward=interact_data.reward,
                next_observation=(  # Don't put comma at the end of next line...
                    interact_data.observation_before_reset if interact_data.done else interact_data.next_observation
                ),
                done=interact_data.done,
            )

            total_num_new_steps = interact_data.num_new_steps.item()
            self.num_env_steps += total_num_new_steps

            self.train_info['last_train']['current_episode_reward'] += interact_data.reward
            if interact_data.done:
                self.train_info['last_train']['total_complete_episodes_reward'] += self.train_info['last_train']['current_episode_reward']
                self.train_info['last_train']['total_complete_episodes'] += 1
                self.train_info['last_train']['current_episode_reward'] = 0

        if not test_update_to_date:
            test(force_visualize=True)
        if not save_up_to_date:
            save()
        # See NOTE [ Pre-emption ]
        open(self.cfg.job_complete_file, 'a').close()
        utils.rm_if_exists(self.cfg.tmp_preempt_ckpt_dir, maybe_dir=True)
        utils.rm_if_exists(self.cfg.preempt_ckpt_dir, maybe_dir=True)


@hydra.main(version_base=None, config_name="config")
def main(dict_cfg: DictConfig) -> None:
    # parse
    _, cfg = to_config_and_instantiate(dict_cfg)

    # NOTE [ Pre-emption ]
    def handle_SIGUSR1_set_flag(_, __):
        logging.warning('Signal received: SIGUSR1, prepare to checkpoint')
        cfg.received_SIGUSR1 = True
        if os.path.isfile(cfg.job_complete_file):
            logging.warning('Already complete, exiting')
            sys.exit(0)

    signal.signal(signal.SIGUSR1, handle_SIGUSR1_set_flag)
    logging.info('Signal handler installed')

    torch.backends.cudnn.benchmark = True

    # Log config
    logging.info('')
    logging.info(OmegaConf.to_yaml(cfg.config))
    logging.info('')
    logging.info(f'Running on {socket.getfqdn()}:')
    logging.info(f'\t{"PID":<30}{os.getpid()}')
    for var in ['CUDA_VISIBLE_DEVICES', 'EGL_DEVICE_ID']:
        logging.info(f'\t{var:<30}{os.environ.get(var, None)}')
    logging.info('')
    logging.info(f'Base Git directory {cfg.base_git_dir}')
    logging.info(f'Output directory {cfg.output_dir}')
    logging.info('')

    with open(os.path.join(cfg.output_dir, 'config.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg.config))

    # Seeding
    torch_seed, np_seed, data_collect_env_seed, replay_buffer_seed = split_seed(cast(int, cfg.seed), 4)
    np.random.seed(np.random.Generator(np.random.PCG64(np_seed)).integers(1 << 31))
    torch.manual_seed(np.random.Generator(np.random.PCG64(torch_seed)).integers(1 << 31))

    # Trainer
    writer = SummaryWriter(cfg.output_dir)
    trainer = ModelTrainer(cfg, writer, data_collect_env_seed, replay_buffer_seed, cfg.test_seed)

    logging.info('World Model:\n\t' + str(trainer.world_model).replace('\n', '\n\t') + '\n')
    logging.info('Model Learning:\n\t' + str(trainer.model_learning).replace('\n', '\n\t') + '\n')
    logging.info('Policy Learning:\n\t' + str(trainer.policy_learning).replace('\n', '\n\t') + '\n')
    logging.info('Number of parameters:')
    logging.info(f'\t model                {sum(p.numel() for p in trainer.world_model.model_learning_parameters())}')
    logging.info(f'\t actor                {sum(p.numel() for p in trainer.world_model.actor_model.parameters())}')
    logging.info(f'\t model_learning       {sum(p.numel() for p in trainer.model_learning.parameters())}')
    logging.info(f'\t pi_learning          {sum(p.numel() for p in trainer.policy_learning.parameters())}')
    total_params = sum(
        p.numel() for p in itertools.chain(
            trainer.world_model.parameters(),
            trainer.policy_learning.parameters(),
            trainer.model_learning.parameters(),
        )
    )
    logging.info('\t ' + '-' * 35)
    logging.info(f'\t TOTAL                {total_params}')
    logging.info('')

    trainer.fit()


if __name__ == '__main__':
    main()
