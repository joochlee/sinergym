import logging

import gymnasium as gym
import numpy as np

from gymnasium.wrappers import TransformAction

# import sinergym
# from sinergym.utils.logger import TerminalLogger
# from sinergym.utils.wrappers import (
#     CSVLogger,
#     LoggerWrapper,
#     NormalizeAction,
#     NormalizeObservation,
# )

import sinergym
from sinergym.utils.callbacks import *
from sinergym.utils.constants import *
from sinergym.utils.logger import WandBOutputFormat
from sinergym.utils.rewards import *
from sinergym.utils.wrappers import *


from datetime import datetime

from stable_baselines3 import *
from stable_baselines3.common.callbacks import CallbackList, BaseCallback
from stable_baselines3.common.logger import HumanOutputFormat
from stable_baselines3.common.logger import Logger as SB3Logger
from stable_baselines3.common.monitor import Monitor

from torch.utils.tensorboard import SummaryWriter


def transform_action(action):
    continuous1 = action[0]
    continuous2 = action[1]
    discrete = int(np.round(action[2]))
    discrete = np.clip(discrete, 0, 1)  # 0 또는 1
    return np.array([continuous1, continuous2, discrete], dtype=np.float32)


# Logger
terminal_logger = TerminalLogger()
logger = terminal_logger.getLogger(
    name='MAIN',
    level=logging.INFO
)

# Environment ID
environment = 'Eplus-CompassCAV-normal-continuous-stochastic-v1'

# Training episodes
episodes = 10

# Name of the experiment
experiment_date = datetime.today().strftime('%Y-%m-%d_%H:%M')
experiment_name = 'SB3_PPO-' + environment + \
    '-episodes-' + str(episodes)
experiment_name += '_' + experiment_date


env = gym.make(environment, env_name=experiment_name)
eval_env = gym.make(environment, env_name=experiment_name + '_EVALUATION')


# Create environment and apply wrappers for normalization and logging
# env = gym.make('Eplus-CompassCAV-normal-continuous-stochastic-v1')

env = TransformAction(env, transform_action, env.action_space)
env = NormalizeAction(env)
env = NormalizeObservation(env)
env = LoggerWrapper(env)
env = CSVLogger(env)
env = Monitor(env)

eval_env = TransformAction(eval_env, transform_action, eval_env.action_space)
eval_env = NormalizeObservation(eval_env)
eval_env = NormalizeAction(eval_env)
eval_env = LoggerWrapper(eval_env)
eval_env = CSVLogger(eval_env)
eval_env = Monitor(eval_env)


class SinergymTBCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.ep_rewards = deque(maxlen=100)

        # self._energy_buffer = []

    def _on_training_start(self) -> None:
        self.writer = SummaryWriter(log_dir = self.logger.dir)
        return super()._on_training_start()

    def _on_step(self) -> bool:
        # infos = self.locals.get("infos", [])
        # if infos:
        #     info = infos[-1]
        #     # 예시 키: Sinergym 환경/리포트에 맞춰 실제 키로 교체하세요
        #     energy = info.get("electricity_power")  # kW 등
        #     discomfort = info.get("discomfort")     # 0~1 스칼라 등
        #     if energy is not None:
        #         self._energy_buffer.append(energy)
        #         # step 단위 로그
        #         self.logger.record("custom/energy_step", float(energy))
        #     if discomfort is not None:
        #         self.logger.record("custom/discomfort_step", float(discomfort))
        info = self.locals.get('infos')[-1]
        self.writer.add_scalar( f"result/reward", info.get('reward'), self.num_timesteps)
        
        infos = self.locals.get("infos")
        if infos is not None:
            for info in infos:
                # Monitor 래퍼가 episode 종료 시 info["episode"] 추가
                if "episode" in info.keys():
                    ep_r = info["episode"]["r"]  # 해당 episode의 총 보상
                    self.ep_rewards.append(ep_r)

                    # rolling 평균 계산
                    mean_r = sum(self.ep_rewards) / len(self.ep_rewards)
                    self.writer.add_scalar("custom/ep_rew_mean", mean_r, self.num_timesteps)

        step_rewards = self.locals.get('rewards')
        self.writer.add_scalar( f"result/step_reward", step_rewards, self.num_timesteps)

        # if (self.num_timesteps % 4) == 0:
        # self.logger.record('result/reward', float(info.get('reward')))

        # print(f'======> reward : {info.get("reward")}\n')
        # print(
        #     f'======> n_step : {
        #         self.model.n_steps}, n_envs: {
        #         self.model.n_envs}\n')

        return True

    # def _on_rollout_end(self) -> None:
    #     if self._energy_buffer:
    #         self.logger.record("custom/energy_rollout_mean",
    #                            float(np.mean(self._energy_buffer)))
    #         self._energy_buffer.clear()
    def _on_training_end(self) -> None:
        self.writer.flush()
        self.writer.close()


# In this case, all the hyperparameters are the default ones
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log='./tb_logs')


callbacks = []

# Set up Evaluation logging and saving best model
eval_callback = LoggerEvalCallback(
    eval_env=eval_env,
    train_env=env,
    n_eval_episodes=1,
    eval_freq_episodes=2,
    deterministic=True)

callbacks.append(eval_callback)
callbacks.append(SinergymTBCallback())
callback = CallbackList(callbacks)

timesteps = episodes * (env.get_wrapper_attr('timestep_per_episode') - 1)

# Execute 1 episode
# episodes = 1

model.learn(
    total_timesteps=timesteps,
    callback=callback,
    log_interval=100,
    tb_log_name='ppo_cav')


model.save(env.get_wrapper_attr('workspace_path') + '/model')

# for i in range(episodes):

#     # Reset the environment to start a new episode
#     obs, info = env.reset()

#     rewards = []
#     truncated = terminated = False
#     current_month = 0

#     while not (terminated or truncated):

#         # Random action selection
#         a = env.action_space.sample()

#         # Perform action and receive env information
#         obs, reward, terminated, truncated, info = env.step(a)

#         rewards.append(reward)

#         # Display results every simulated month
#         if info['month'] != current_month:
#             current_month = info['month']
#             logger.info('Reward: {}'.format(sum(rewards)))
#             logger.info('Info: {}'.format(info))

#     logger.info('Episode {} - Mean reward: {} - Cumulative Reward: {}'.format(i,
#   np.mean(rewards), sum(rewards)))
env.close()
