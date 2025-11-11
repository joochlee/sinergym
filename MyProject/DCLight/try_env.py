import logging

import gymnasium as gym
from gymnasium import spaces, Wrapper, ActionWrapper
from gymnasium.wrappers import TransformAction
import numpy as np

import sinergym
from sinergym.utils.logger import TerminalLogger
from sinergym.utils.wrappers import (
    CSVLogger,
    LoggerWrapper,
    NormalizeAction,
    NormalizeObservation,
)


# class DiscreteAction(gym.Env):
#     def __init__(self):
#         super().__init__()
#         self.action_space = spaces.Dict({
#             "direction": spaces.Discrete(3),
#             "throttle": spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
#         })
#         self.observation_space = spaces.Box(-np.inf,
# np.inf, shape=(4,), dtype=np.float32)

#     def step(self, action):
#         direction = action["direction"]
#         throttle = action["throttle"]
#         # 환경 로직...
#         return np.zeros(4, dtype=np.float32), 0.0, False, False, {}

#     def reset(self, seed=None, options=None):
#         return np.zeros(4, dtype=np.float32), {}


# class ActionFlattenWrapper(Wrapper):
#     def __init__(self, env):
#         super().__init__(env)
#         # Discrete(3) + Box(-1,1) → Box([0,-1], [2,1])
#         low = np.array([0, -1.0], dtype=np.float32)
#         high = np.array([2, 1.0], dtype=np.float32)
#         self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

#     def step(self, action):
#         # action: [direction, throttle] (Box로 들어옴)
#         direction = int(round(action[0]))        # 0~2 사이 정수로 변환
#         throttle = np.array([action[1]], dtype=np.float32)
#         dict_action = {"direction": direction, "throttle": throttle}
#         return self.env.step(dict_action)


# class DiscreteAction(ActionWrapper):
#     """HAV operation on/off를 환경에 전달하기 전에 discretize 시킴

#     Args:
#         gym (_type_): 원
#     """

#     def __init__(self, env):
#         super().__init__(env)

#     def action(self, action):
#         continuous1 = action[0]
#         continuous2 = action[1]
#         discrete = int(np.round(action[2]))
#         discrete = np.clip(discrete, 0, 1)  # 0 또는 1
# return np.array([continuous1, continuous2, discrete], dtype=np.float32)


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

# Create environment and apply wrappers for normalization and logging
env = gym.make('Eplus-DCLight-normal-continuous-stochastic-v1')
# env = DiscreteAction(env)

env = TransformAction(env, transform_action, env.action_space)
env = NormalizeAction(env)
env = NormalizeObservation(env)
env = LoggerWrapper(env)
env = CSVLogger(env)

# CAV on/off 값을 위한 wrapper 필요


# Execute 1 episode
episodes = 1
for i in range(episodes):

    # Reset the environment to start a new episode
    obs, info = env.reset()

    rewards = []
    truncated = terminated = False
    current_month = 0

    while not (terminated or truncated):

        # Random action selection
        a = env.action_space.sample()

        # Perform action and receive env information
        obs, reward, terminated, truncated, info = env.step(a)

        rewards.append(reward)

        # Display results every simulated month
        if info['month'] != current_month:
            current_month = info['month']
            logger.info('Reward: {}'.format(sum(rewards)))
            logger.info('Info: {}'.format(info))
            logger.info(f'obs:{obs}')

    logger.info('Episode {} - Mean reward: {} - Cumulative Reward: {}'.format(i,
                                                                              np.mean(rewards), sum(rewards)))
env.close()
