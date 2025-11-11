import logging
from collections import deque

import gymnasium as gym
import numpy as np

from gymnasium.wrappers import TransformAction, TimeLimit
from gymnasium import spaces
from torch.onnx.symbolic_opset9 import to


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

from random import choice
# from sinergym.utils.wrappers import NormalizeObservation



from datetime import datetime

from stable_baselines3 import *
from stable_baselines3.common.callbacks import CallbackList, BaseCallback
from stable_baselines3.common.logger import HumanOutputFormat
from stable_baselines3.common.logger import Logger as SB3Logger
from stable_baselines3.common.monitor import Monitor


from stable_baselines3.common.env_util import make_vec_env


from torch.utils.tensorboard import SummaryWriter
import torch


# -----------------------------------------------------------------------------
# 사용자 선호도 추가용 wrapper
# ----------------------------------------------------------------------------- 
# 사용자 선호 정의 (벡터 + 가중치)
PREFERENCE_MAP = {
   "economical": {
      "vec": np.array([1.0, 0.0, 0.0]),
      "reward_weight": 0.8  # energy_weight
   },
   "balanced": {
      "vec": np.array([0.0, 1.0, 0.0]),
      "reward_weight": 0.5
   },
   "comfort": {
      "vec": np.array([0.0, 0.0, 1.0]),
      "reward_weight": 0.2
   }
}

class PreferenceWrapper(gym.Wrapper):
   def __init__(self, env, preference_type="balanced"):
      super().__init__(env)
      assert preference_type in PREFERENCE_MAP

      self.preference_type = preference_type
      self.preference_vec = PREFERENCE_MAP[preference_type]["vec"]
      self.w_energy = PREFERENCE_MAP[preference_type]["reward_weight"]

      # 상태 공간 확장
      orig_obs_space = self.observation_space
      self.observation_space = spaces.Box(
         low=np.concatenate([orig_obs_space.low, np.zeros_like(self.preference_vec)]),
         high=np.concatenate([orig_obs_space.high, np.ones_like(self.preference_vec)]),
         dtype=np.float64
      )

   def reset(self, **kwargs):
      obs, info = self.env.reset(**kwargs)
      # 한 에피소드가 끝나면 "선호도"를 랜덤값으로 변경해서 다시 학습함
      # self.preference_type = choice(list(PREFERENCE_MAP))
      # self.preference_vec = PREFERENCE_MAP[self.preference_type]['vec']
      # self.w_energy = PREFERENCE_MAP[self.preference_type]['reward_weight']
      # energy weight를 선호도에 맞게 업데이트
      # self.unwrapped.reward_fn.W_energy = self.w_energy

      # 변경된 "선호도"를 상태에 업데이트함
      obs = np.concatenate([obs, self.preference_vec])
      return obs, info

   def set_pref(self, preference_type : str = 'balanced'):
      """사용자선호도를 지정된 값으로 변경함 (reward_weight (== energy_weight) 변경)

      Args:
          preference_type (str): 선호도 값
      """
      self.preference_type = preference_type
      self.preference_vec = PREFERENCE_MAP[self.preference_type]['vec']
      self.w_energy = PREFERENCE_MAP[self.preference_type]['reward_weight']
      self.unwrapped.reward_fn.W_energy = self.w_energy

   def step(self, action):
      obs, reward, done, truncated, info = self.env.step(action)
      # obs, reward, done, truncated, info = self.unwrapped.step(action)

      # ----- 원래 reward와 raw info 기반의 새로운 reward 계산 -----
      # Sinergym의 info에 따라 적절히 조정 필요 (예시는 아래 가정 기반)
      # energy = info.get('electricity_demand', 0.0)
      # discomfort = info.get('comfort_penalty', 0.0)

      # shaped_reward = - (self.w_energy * energy + self.w_discomfort * discomfort)

      obs = np.concatenate([obs, self.preference_vec])
      return obs, reward, done, truncated, info


def transform_action(action):
   """
   액션을 변환하는 함수
   연속값 2개와 이산값 1개를 처리하여 적절한 형태로 변환
   """
   continuous1 = action[0]  # 첫 번째 연속 액션
   continuous2 = action[1]  # 두 번째 연속 액션
   discrete = int(np.round(action[2]))  # 이산 액션을 정수로 변환
   discrete = np.clip(discrete, 0, 1)  # 0 또는 1로 제한
   return np.array([continuous1, continuous2, discrete], dtype=np.float32)


def find_wrapper(env, wrapper_class):
   """원하는 wrapper를 찾는 함수

   Args:
      env (gym.Wrapper): wrapper
      wrapper_class (any): 찾고자 하는 wrapper type

Returns:
      any: 찾은 wrapper
   """
   while hasattr(env, 'env'):
      if isinstance(env, wrapper_class):
         return env
      env = env.env
   return None

# 로거 설정
terminal_logger = TerminalLogger()
logger = terminal_logger.getLogger(
   name='MAIN',
   level=logging.INFO
)

# 환경 설정
environment = 'Eplus-CompassCAV-normal-continuous-stochastic-v1'  # Sinergym 환경 ID
episodes = 1  # 실행 에피소드 수

# 실험 이름 생성 (날짜/시간 포함)
experiment_date = datetime.today().strftime('%Y-%m-%d_%H:%M')
experiment_name = 'SB3_PPO-' + environment + \
   '-episodes-' + str(episodes)
experiment_name += '_' + experiment_date


# 훈련용 환경 생성
env = gym.make(environment, env_name=experiment_name)
# 평가용 환경 생성
# eval_env = gym.make(environment, env_name=experiment_name + '_EVALUATION')

print(f'\n===> workspace_path \n{env.get_wrapper_attr('workspace_path')}\n')


# Create environment and apply wrappers for normalization and logging
# env = gym.make('Eplus-CompassCAV-normal-continuous-stochastic-v1')

# 훈련 환경에 래퍼 적용
env = TransformAction(env, transform_action, env.action_space)  # 액션 변환
env = NormalizeAction(env)  # 액션 정규화
env = NormalizeObservation(env)  # 관찰값 정규화
env = PreferenceWrapper(env)  # 선호도

# env = LoggerWrapper(env)  # 로깅 래퍼
# env = CSVLogger(env)  # CSV 로깅
# env = Monitor(env)  # 모니터링

# 평가 환경에 래퍼 적용
# eval_env = TransformAction(eval_env, transform_action, eval_env.action_space)
# eval_env = NormalizeObservation(eval_env)
# eval_env = NormalizeAction(eval_env)
# eval_env = LoggerWrapper(eval_env)
# eval_env = CSVLogger(eval_env)
# eval_env = Monitor(eval_env)



# -----------------------------------------------------------------------------
# GPU 사용 가능 여부 확인
# ----------------------------------------------------------------------------- 
# PPO with MlpPolicy는 GPU보다 CPU가 더 빠름, 아래 경고메시지 참고 (by jclee)
# /usr/local/lib/python3.12/dist-packages/stable_baselines3/common/on_policy_algorithm.py:150: UserWarning: You are trying to run PPO on the GPU, but it is primarily intended to run on the CPU when not using a CNN policy (you are using ActorCriticPolicy which should be a MlpPolicy). See https://github.com/DLR-RM/stable-baselines3/issues/1245 for more info. You can pass `device='cpu'` or `export CUDA_VISIBLE_DEVICES=` to force using the CPU.Note: The model will train, but the GPU utilization will be poor and the training might take longer than on CPU.
device = 'cpu'
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f"사용 중인 디바이스: {device}")
# if torch.cuda.is_available():
#    print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
#    print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
# else:
#    print("GPU를 사용할 수 없습니다. CPU로 실행됩니다.")



# ----------------------------------------------------------------------------- 
# PPO 모델 생성 (기본 하이퍼 파라미터사용)
# ----------------------------------------------------------------------------- 
# model = PPO('MlpPolicy', env, verbose=1, tensorboard_log='./tb_logs')

# PPO 모델 생성 (GPU 사용 설정 및 최적화된 하이퍼파라미터)
# model = PPO(
#    'MlpPolicy', 
#    env, 
#    verbose=1, 
#    tensorboard_log='./tb_logs',
#    device=device,  # GPU 사용 설정
#    learning_rate=3e-4,
#    n_steps=2048,  # GPU 사용 시 더 큰 배치 크기 권장
#    batch_size=64,
#    n_epochs=10,
#    gamma=0.99,
#    gae_lambda=0.95,
#    clip_range=0.2,
#    ent_coef=0.0,
#    vf_coef=0.5,
#    max_grad_norm=0.5,
#    policy_kwargs={
#       'net_arch': [dict(pi=[256, 256], vf=[256, 256])]  # GPU 사용 시 더 큰 네트워크
#    } if device == 'cuda' else None
# )

# model = PPO(
#    'MlpPolicy', 
#    env, 
#    verbose=1, 
#    tensorboard_log='./tb_logs',
#    device=device,  # GPU 사용 설정
#    # learning_rate=3e-4,
#    n_steps=2048,  # 더 큰 rollout로 안정적인 학습
#    batch_size=256,
#    n_epochs=10,
#    # gamma=0.99,
#    # gae_lambda=0.95,
#    # clip_range=0.2,
#    # ent_coef=0.0,
#    # vf_coef=0.5,
#    # max_grad_norm=0.5,
#    # policy_kwargs={
#    #    'net_arch': [dict(pi=[256, 256], vf=[256, 256])]  # GPU 사용 시 더 큰 네트워크
#    # } if device == 'cuda' else None
# )

# MODEL_PATH = './trained_model/model_ew_0.7.zip'
# MODEL_PATH = './trained_model/model_ew_0.3.zip'
MODEL_PATH = './trained_model/model_dynamic_pref.zip'
LOG_DIR = './tb_logs/evaluation'

# save된 학습모델 로딩
model = PPO.load(MODEL_PATH, device=device)

# --- TensorBoard writer 설정 ---
run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, run_id))


timesteps = episodes * (env.get_wrapper_attr('timestep_per_episode') - 1)
print(f'\n===> 1ep 당 timesteps: \n{timesteps}\n')

# TimeLimit 래퍼로 최대 스텝 제한 (예: 500 스텝)
# env = TimeLimit(env, max_episode_steps=timesteps)

# 100 스텝마다 total_power_demand의 rolling average를 구함
q_len = 10
mean_total_power_demand_rolling = deque(maxlen=q_len)  # 최근 100개 스텝만 저장


# 한 에피소드 (1년) 동안 테스트
pref_env = find_wrapper(env, PreferenceWrapper)
if pref_env:
   # pref_env.set_pref('balanced')
   pref_env.set_pref('comfort')
   # pref_env.set_pref('economical')
   print(f'\n===> 선호도 초기설정 : \n{pref_env.w_energy}\n')
else:
   print(f'\n===> Error! PreferenceWrapper not found! \n')
   exit(-1)

# env.set_pref('comfort')

for episode in range(episodes):
   obs, info = env.reset()

   print(f'\n===> obs \n{obs}\n')
   print(f'\n===> info \n{info}\n')

   done = False
   truncated = False
   total_reward = 0.0
   steps = 0

   power_demand_0 = 0.0
   power_demand_1 = 0.0
   power_demand_2 = 0.0

   while not (done or truncated or (timesteps < steps)):
      action, _states = model.predict(obs, deterministic=True)

      # print(f'\n===> obs \n{obs}\n')

      obs, reward, done, truncated, info = env.step(action)

      # print(f'\n===> info \n{info}\n')

      total_reward += reward
      writer.add_scalar('eval/step_reward', reward, steps)
      writer.add_scalar('eval/total_power_demand', info['total_power_demand'], steps)
      writer.add_scalar('eval/total_temperature_violation', info['total_temperature_violation'], steps)
      
      # total_power_demand_sum = sum(self.ep_total_power_demand) if self.ep_total_power_demand else 0.0
      mean_total_power_demand_rolling.append(info['total_power_demand'])
         
      if not (steps % q_len):
         # Rolling 평균 계산 (최근 100개 스텝)
         if len(mean_total_power_demand_rolling) > 0:
            rolling_mean = sum(mean_total_power_demand_rolling) / len(mean_total_power_demand_rolling)
            writer.add_scalar("eval/mean_total_power_demand_rolling", rolling_mean, steps)

      if not (steps % 5000):
         print(f'\n===> reward_weight \n{info['reward_weight']}\n')

      # -----------------------------------------

      # if steps == 10000:
      #    # pref_env.set_pref('comfort')
      #    pref_env.set_pref('economical')
      
      # env.set_pref('economical')
      
      if steps < 10000:
         power_demand_0 += info['total_power_demand']
      else:
         power_demand_1 += info['total_power_demand']

      # -----------------------------------------


      # if steps == 10000:
      #    env.set_pref('comfort')
      # elif steps == 45000:
      #    env.set_pref('economical')

      # if steps < 10000:
      #    power_demand_0 += info['total_power_demand']
      # elif steps < 45000:
      #    power_demand_1 += info['total_power_demand']
      # else:
      #    power_demand_2 += info['total_power_demand']

      # -----------------------------------------

      # if not (steps % 17500):
      #    if steps / 17500 == 1:
      #       env.set_pref('economical')
      #    elif steps / 17500 == 2:
      #       env.set_pref('comfort')

      # if steps < 17500:
      #    power_demand_0 += info['total_power_demand']
      # elif steps < 35000:
      #    power_demand_1 += info['total_power_demand']
      # else:
      #    power_demand_2 += info['total_power_demand']

      steps += 1

   print(f"[Episode {episode}] Total reward: {total_reward}, Steps: {steps}")

   print(f'\n===> power_demand_0 \n{power_demand_0}\n')
   print(f'\n===> power_demand_1 \n{power_demand_1}\n')
   print(f'\n===> power_demand_2 \n{power_demand_2}\n')

   # TensorBoard 기록
   writer.add_scalar('eval/Total_Reward', total_reward, episode)
   writer.add_scalar('eval/Episode_Length', steps, episode)

# --- 종료 ---
writer.close()
env.close()


# # 콜백 리스트 초기화
# callbacks = []

# # 평가 콜백 설정 (모델 저장 및 평가 로깅)
# eval_callback = LoggerEvalCallback(
#    eval_env=eval_env,  # 평가 환경
#    train_env=env,  # 훈련 환경
#    n_eval_episodes=1,  # 평가 에피소드 수
#    eval_freq_episodes=2,  # 평가 주기 (2 에피소드마다)
#    deterministic=True)  # 결정적 정책 사용


# # 콜백들을 리스트에 추가
# # callbacks.append(eval_callback)
# callbacks.append(SinergymTBCallback())
# # callbacks.append(rich_cb)

# callback = CallbackList(callbacks)  # 콜백 리스트 생성

# # 총 훈련 타임스텝 계산
# # (에피소드 수 × (에피소드당 타임스텝 - 1))
# timesteps = episodes * (env.get_wrapper_attr('timestep_per_episode') - 1)

# # print(f'\n===> env \n{env}\n')

# print('===> What is all the variables names which conform my Gymnasium observation?: {}'.format(
#    env.get_wrapper_attr('observation_variables')))
# print('===> What is all the variables names which conform my Gymnasium action?: {}'.format(
#    env.get_wrapper_attr('action_variables')))
# print('===> Is there an episode executing?: {}'.format(
#    env.get_wrapper_attr('is_running')))
# print('===> What is the episode run period?: {}'.format(
#    env.get_wrapper_attr('runperiod')))
# print('===> Episode length in seconds?: {}'.format(
#    env.get_wrapper_attr('episode_length')))
# episode_length = env.get_wrapper_attr('timestep_per_episode')
# print('===> How many steps have an episode?: {}'.format(episode_length))
# print(f'===> Episode length: {episode_length} steps')
# print(f'===> Current n_steps: 4096')
# print(f'===> Recommendation: n_steps should be < {episode_length} for proper episode tracking')
# print('===> How often do steps changed?: {}'.format(
#    env.get_wrapper_attr('step_size')))
# print('===> What are the available zones?: {}'.format(
#    env.get_wrapper_attr('zone_names')))
# print('===> Where is the output path of Sinergym environment?: {}'.format(
#    env.get_wrapper_attr('workspace_path')))
# print('===> Where is the building used?: {}'.format(
#    env.get_wrapper_attr('building_path')))
# print('===> Where is the weather used?: {}'.format(
#    env.get_wrapper_attr('weather_path')))
# print('===> Is the action space discrete?: {}'.format(
#    env.get_wrapper_attr('is_discrete')))


# # -----------------------------------------------------------------------------
# # 모델 훈련 실행
# # ----------------------------------------------------------------------------- 
# model.learn(
#    total_timesteps=timesteps,  # 총 훈련 타임스텝
#    callback=callback,  # 콜백 함수들
#    log_interval=100,  # 로그 출력 주기
#    tb_log_name='cav_ppo_dymanic_preference')  # TensorBoard 로그 이름


# # -----------------------------------------------------------------------------
# # 훈련된 모델 저장
# # ----------------------------------------------------------------------------- 
# model.save(env.get_wrapper_attr('workspace_path') + '/model')

# # 주석 처리된 수동 에피소드 실행 코드
# # (현재는 PPO 모델이 자동으로 에피소드를 실행하므로 불필요)
# # for i in range(episodes):
# #     obs, info = env.reset()

# #     rewards = []
# #     truncated = terminated = False
# #     current_month = 0

# #     while not (terminated or truncated):

# #         # Random action selection
# #         a = env.action_space.sample()

# #         # Perform action and receive env information
# #         obs, reward, terminated, truncated, info = env.step(a)

# #         rewards.append(reward)

# #         # Display results every simulated month
# #         if info['month'] != current_month:
# #             current_month = info['month']
# #             logger.info('Reward: {}'.format(sum(rewards)))
# #             logger.info('Info: {}'.format(info))

# #     logger.info('Episode {} - Mean reward: {} - Cumulative Reward: {}'.format(i,
# #         np.mean(rewards), sum(rewards)))

# # 환경 종료
# env.close()
