import logging
from collections import deque

import random

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
from sinergym.utils.logger import WandBOutputFormat, TerminalLogger, CustomFormatter
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
# 로거 설정
# ----------------------------------------------------------------------------- 
# _terminal_logger = TerminalLogger()


custom_formatter = CustomFormatter()
# LOG_FORMAT을 함수명/파일명 포함 포맷으로 오버라이드
custom_formatter.log_format = "[%(name)s] (%(levelname)s) [%(filename)s::%(funcName)s] : %(message)s"
custom_formatter.FORMATS = {
    logging.DEBUG:    custom_formatter.grey     + custom_formatter.log_format + custom_formatter.reset,
    logging.INFO:     custom_formatter.grey     + custom_formatter.log_format + custom_formatter.reset,
    logging.WARNING:  custom_formatter.yellow   + custom_formatter.log_format + custom_formatter.reset,
    logging.ERROR:    custom_formatter.red      + custom_formatter.log_format + custom_formatter.reset,
    logging.CRITICAL: custom_formatter.bold_red + custom_formatter.log_format + custom_formatter.reset,
}

logger = TerminalLogger().getLogger(
    name='MAIN',
    level=logging.INFO,
    formatter=custom_formatter
)


# logger = TerminalLogger().getLogger(
#    name='MAIN',
#    level=logging.INFO
# )



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



# -----------------------------------------------------------------------------
# continuous action => discrete action 변환 wrapper
# ----------------------------------------------------------------------------- 
class ActionAdjustmentWrapper(gym.ActionWrapper):
   """
   환경에 넘겨주기 직전에 action을 수정
   """
   def __init__(self, env):
      super().__init__(env)

   def action(self, action):
      # 여기서의 'action'은 이미 바깥쪽 래퍼에서 복원된 '물리 값'입니다.
      # 예: 특정 임계값을 넘지 못하게 제한(Clipping)하거나 보정치를 더함

      # adjusted_action = action + 5.0  # 예시: 모든 액션에 5를 더함
      # return adjusted_action

      # XXX 06:00 ~ 19:00 이외에는 강제로 OFF 처리
      if 6 > self.env.get_wrapper_attr("last_info")["hour"] or self.env.get_wrapper_attr("last_info")["hour"] > 19:
         action[2] = 0
         # print(f'\n===> HVAC OFF Time! \n{action}, Hour: {self.env.get_wrapper_attr("last_info")["hour"]}\n')

      return action


# -----------------------------------------------------------------------------
# 강화학습 모델에 전달되는 observation space의 확장 및 reward 재계산용 wrapper
# ----------------------------------------------------------------------------- 
class ObsRewardWrapper(gym.Wrapper):
   def __init__(self, env):
      super().__init__(env)

      # 학습에 사용할 SoC 의 시작값을 설정함
      self.current_soc = 1.0

      # 상태 공간 확장 : 
      # - 기존 : (lights_electricity_rate)
      # - 확장 : (lights_electricity_rate, ess_soc, grid_usage, dim_level)
      orig_obs_space = self.observation_space

      logger.info(f'\n===> orig_obs_space \n{orig_obs_space}\n')

      # self.observation_space = spaces.Box(
      self.unwrapped.observation_space = spaces.Box(
         low=np.concatenate([orig_obs_space.low, [0.0, 0.0, 0.0]]),
         high=np.concatenate([orig_obs_space.high, [1.0, 20000.0, 1.0]]),
         dtype=np.float64
      )


   def _update_soc(self, light_rate: float, ess_total: float) -> None:
      """
      매 20분마다 조명이 소모하는 전력을 받아서 현재 SoC를 업데이트함
      - 방전 에너지 [Wh] = light_rate [W] × (20/60) [h]
      
      Args:
         light_rate : 스마트조명 방전 순간전력 [W]
         ess_total  : 배터리 총 용량 [Wh]
      """
      # print(f'\n===> initial self.current_soc \n{self.current_soc}\n')

      discharge_wh = light_rate * (20 / 60)   # W → Wh 변환 (20분 기준)
      new_soc = self.current_soc - (discharge_wh / ess_total)

      # 새로운 SoC (0.0 ~ 1.0 클리핑)
      # return float(np.clip(new_soc, 0.0, 1.0))
      self.current_soc = float(np.clip(new_soc, 0.2, 1.0))

      # print(f'\n===> light_rate : {light_rate}, discharge_wh : {discharge_wh}, new_soc : {new_soc}, self.current_soc : {self.current_soc}\n')

   def _reset_start_soc(self):
      """학습에 사용할 SoC의 시작값을 재설정한다
      """
      # self.current_soc = random.uniform(0.8, 1.0)
      self.current_soc = 1.0


   def reset(self, **kwargs):
      obs, info = self.env.reset(**kwargs)
      # # 한 에피소드가 끝나면 "선호도"를 랜덤값으로 변경해서 다시 학습함
      # self.preference_type = choice(list(PREFERENCE_MAP))
      # self.preference_vec = PREFERENCE_MAP[self.preference_type]['vec']
      # self.w_energy = PREFERENCE_MAP[self.preference_type]['reward_weight']
      # # energy weight를 선호도에 맞게 업데이트
      # self.unwrapped.reward_fn.W_energy = self.w_energy
      
      # 새로운 에피소드에서 사용할 SoC의 시작값을 업데이트함
      self._reset_start_soc()

      # 변경된 "선호도"를 상태에 업데이트함
      obs = np.concatenate([obs, [self.current_soc, 0.0, 0.0]])

      # print(f'\n===> obs \n{obs}\n')
      # TODO: step()에서 수행하던 것처럼 info에 정보를 추가해줄 것
      return obs, info

   def step(self, action):
      # # 한 에피소드가 진행될때도 랜덤하게 "선호도"를 변경해서 학습함
      # self.preference_type = choice(list(PREFERENCE_MAP))
      # self.preference_vec = PREFERENCE_MAP[self.preference_type]['vec']
      # self.w_energy = PREFERENCE_MAP[self.preference_type]['reward_weight']
      # # energy weight를 선호도에 맞게 업데이트
      # self.unwrapped.reward_fn.W_energy = self.w_energy

      # 현재 SoC를 reward계산에 활용할 수 있도록 업데이트해준다
      # XXX: 중요 : SoC를 업데이트하기 전의 값을 알려주어야 한다
      self.unwrapped.reward_fn.set_ess_soc(self.current_soc)

      obs, reward, done, truncated, info = self.env.step(action)
     
      # print(f'\n===> before obs \n{obs}\n')
      # print(f'\n===> before reward \n{reward}\n')

      # 조명이 소모한 전력을 ESS 배터리의 잔여용량에서 제하고, 배터리의 현재 SoC를 업데이트해준다
      self._update_soc(info['total_power_demand'], info['ess_total'])
      # 만약 배터리가 소진되었으면 학습을 종료함
      if self.current_soc == 0.2:
         done = True
      
      # 계통에서 사용하고 있는 전력량
      grid_usage = 0

      # 스마트조명의 현재 밝기 저장
      dim_level = info['dim_level']


      # XXX min SoC 와 grid usage 는 페널티를 먹어도 에이전트가 어떤행동을 해도 이를 개선할 수가 없으므로 일단 제외하자
      shaped_reward = reward 
      # shaped_reward = reward - info['ess_soc_weight']*max(0, info['min_soc'] - self.current_soc) \
      #    - info['grid_weight']*grid_usage*info['lambda_energy']

      # ----- 원래 reward와 raw info 기반의 새로운 reward 계산 -----
      # Sinergym의 info에 따라 적절히 조정 필요 (예시는 아래 가정 기반)
      # energy = info.get('electricity_demand', 0.0)
      # discomfort = info.get('comfort_penalty', 0.0)

      # shaped_reward = - (self.w_energy * energy + self.w_discomfort * discomfort)

      obs = np.concatenate([obs, [self.current_soc, grid_usage, dim_level]])

      # print(f'\n===> after obs \n{obs}\n')
      logger.info(f'\n===> self.current_soc \n{self.current_soc}\n')
      logger.info(f'\n===> action \n{action}\n')

      # XXX info 에 현재 SoC값을 저장해 준다. 일반적으로 info 는 ObservationNormalization wrapper에 의해서 normalize되지 않는다.
      info['ess_soc'] = self.current_soc

      return obs, shaped_reward, done, truncated, info



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

'''
# 로거 설정
terminal_logger = TerminalLogger()
logger = terminal_logger.getLogger(
   name='MAIN',
   level=logging.INFO
)
'''




# -----------------------------------------------------------------------------
# 에이전트 클래스
# ----------------------------------------------------------------------------- 
class Agent:
   def __init__(
      self, 
      environment_id: str, 
      experiment_algorithm: str,
      experiment_detail: str,
      episodes: int,
      experiment_date: str
   ):
      """
      에이전트 초기화

      Args:
         environment_id (str): Gym Env ID
         experiment_algorithm (str): RL 알고리즘명
         experiment_detail (str): RL 설정 세부사항
         episodes (int): 에피소드 횟수
         experiment_date (str): 수행날짜
      """

      # 출력폴더이름 조합
      self.experiment_name = environment_id + '-' \
                        + experiment_algorithm + '-' \
                        + experiment_detail + '[ ' + str(episodes) + ' ]-' \
                        + experiment_date

      # ----------------------------------------------------------------------------- 
      # Env 생성
      # ----------------------------------------------------------------------------- 
      # `env_name`은 EplusEnv 객체의 생성자에게 전달되는 파라미터임. 이 값은 sinergym 출력폴더명으로 이용된다.
      self.env = gym.make(environment_id, env_name = self.experiment_name)
      self.episodes = episodes
      
      # ----------------------------------------------------------------------------- 
      # 로거 설정
      # ----------------------------------------------------------------------------- 
      # 아래코드는 전역변수로 뺐음 (2026-03-06 07:45:26)
      # _terminal_logger = TerminalLogger()
      # self.logger = _terminal_logger.getLogger(
      #    name='MAIN',
      #    level=logging.INFO
      # )
      logger.info(f'\n===> workspace_path \n{self.env.get_wrapper_attr('workspace_path')}\n')

      # ----------------------------------------------------------------------------- 
      # 총 step 수
      # ----------------------------------------------------------------------------- 
      self.total_timesteps = episodes * (self.env.get_wrapper_attr('timestep_per_episode') - 1)
      logger.info(f'\n===> 1ep 당 timesteps: \n{self.total_timesteps}\n')

      # ----------------------------------------------------------------------------- 
      # 필요한 wrapper 적용
      # ----------------------------------------------------------------------------- 
      # self.env = ActionAdjustmentWrapper(self.env)
      # self.env = TransformAction(self.env, transform_action, self.env.action_space)  # 액션 변환
      # self.env = NormalizeAction(self.env)  # 액션 정규화
      # self.env = NormalizeObservation(self.env)  # 관찰값 정규화
      # self.env = LoggerWrapper(self.env)  # 관찰값 정규화
      # self.env = CSVLogger(self.env)
      # # self.env = PreferenceWrapper(self.env)  # 선호도

      # ----------------------------------------------------------------------------- 
      # GPU 사용여부 확인 :
      # PPO with MlpPolicy는 GPU보다 CPU가 더 빠름, 아래 경고메시지 참고 (by jclee)
      # /usr/local/lib/python3.12/dist-packages/stable_baselines3/common/on_policy_algorithm.py:150: UserWarning: You are trying to run PPO on the GPU, but it is primarily intended to run on the CPU when not using a CNN policy (you are using ActorCriticPolicy which should be a MlpPolicy). See https://github.com/DLR-RM/stable-baselines3/issues/1245 for more info. You can pass `device='cpu'` or `export CUDA_VISIBLE_DEVICES=` to force using the CPU.Note: The model will train, but the GPU utilization will be poor and the training might take longer than on CPU.
      # ----------------------------------------------------------------------------- 
      self.device = 'cpu'
      # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
      # print(f"사용 중인 디바이스: {self.device}")
      # if torch.cuda.is_available():
      #    print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
      #    print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
      # else:
      #    print("GPU를 사용할 수 없습니다. CPU로 실행됩니다.")


   def _add_wrappers(self, wrappers_list: list):
      """wrappers_list 에 포함된 wrapper 들을 순서대로 env에 추가해 준다

      Args:
         wrappers_list (list): env wrapper 객체타입 및 파라미터
         e.g.) wrappers_list: [(gym.wrappers.FlattenObservation, {}), 
                              (gym.wrappers.TimeLimit, {"max_episode_steps": 500})]
      """
      
      for wrapper, kwargs in wrappers_list:
         if wrapper is NormalizeAction:
            self.normalize_action_wrapper = wrapper(self.env, **kwargs)
         elif wrapper is NormalizeObservation:
            self.normalize_observation_wrapper = wrapper(self.env, **kwargs)
         self.env = wrapper(self.env, **kwargs)



   def load_model(self, model_path: str, log_dir: str):
      """
      학습된 모델 로딩

      Args:
          model_path (str): SB3 학습된 모델 경로
          log_dir (str): tensorboard 로깅 폴더 경로
      """

      # save된 학습모델 로딩
      self.model = PPO.load(model_path, device = self.device)
      # TensorBoard writer 설정
      self.writer = SummaryWriter(log_dir = os.path.join(log_dir, self.experiment_name))


   def run(self):
      """
      학습된 기존 모델을 읽어들여서 1 episode만큼 실행함
      """
      for episode in range(self.episodes):
         # ----------------------------------------------------------------------------- 
         # Env reset
         # ----------------------------------------------------------------------------- 
         logger.warning(f'=========================== start new episode ==========================')

         obs, info = self.env.reset()

         logger.info(f'\n===> obs after reset \n{obs}\n')
         logger.info(f'\n===> info after reset \n{info}\n')

         done = False
         truncated = False
         total_reward = 0.0
         steps = 0

         '''
         power_demand_0 = 0.0
         power_demand_1 = 0.0
         power_demand_2 = 0.0
         '''
         
         # 50 스텝마다 total_power_demand의 rolling average를 구함
         q_len = 50
         mean_total_power_demand_rolling = deque(maxlen=q_len)  # 최근 50개 스텝만 저장

         while not (done or truncated or (self.total_timesteps < steps)):
            # ----------------------------------------------------------------------------- 
            # 현재 상태에서 다음 action 추정
            # ----------------------------------------------------------------------------- 
            action, _states = self.model.predict(obs, deterministic=True)
            logger.info(f'\n===> obs after predict \n{obs}\n')
            logger.info(f'\n===> action after predict \n{self.normalize_action_wrapper.reverting_action(action)}, {type(action)}\n')

            # ----------------------------------------------------------------------------- 
            # 선택한 action에 따른 Env 상태 업데이트
            # ----------------------------------------------------------------------------- 
            obs, reward, done, truncated, info = self.env.step(action)
            logger.info(f'\n===> info after step \n{info}\n')


            # ----------------------------------------------------------------------------- 
            # tensorboard 로깅
            # ----------------------------------------------------------------------------- 

            self.writer.add_scalar("perf/ess_soc", info.get('ess_soc'), steps)
            # self.writer.add_scalar("perf/ess_soc", obs[-3], steps)
            # self.writer.add_scalar("perf/grid_usage", obs[-2], steps)

            # TensorBoard에 현재 스텝의 보상값을 기록 (카테고리: result/reward, 값: reward, x축: timesteps)
            reward_value = info.get('reward', 0.0)
            self.writer.add_scalar("perf/step_reward", reward_value, steps)
            
            ess_value = info.get('ess_term', 0.0)
            self.writer.add_scalar("perf/ess_term", ess_value, steps)

            dim_value = info.get('dim_term', 0.0)
            self.writer.add_scalar("perf/dim_term", dim_value, steps)

            dim_level_value = info.get('dim_level', 0.0)
            self.writer.add_scalar("perf/dim_level", dim_level_value, steps)

            total_power_demand_value = info.get('total_power_demand', 0.0)
            self.writer.add_scalar("perf/total_power_demand", total_power_demand_value, steps)





            # total_reward += reward
            # self.writer.add_scalar('eval/step_reward', reward, steps)
            # self.writer.add_scalar('eval/total_power_demand', info['total_power_demand'], steps)
            # self.writer.add_scalar('eval/total_temperature_violation', info['total_temperature_violation'], steps)
            
            # # total_power_demand_sum = sum(self.ep_total_power_demand) if self.ep_total_power_demand else 0.0
            # mean_total_power_demand_rolling.append(info['total_power_demand'])
               
            # if not (steps % q_len):
            #    # Rolling 평균 계산 (최근 100개 스텝)
            #    if len(mean_total_power_demand_rolling) > 0:
            #       rolling_mean = sum(mean_total_power_demand_rolling) / len(mean_total_power_demand_rolling)
            #       self.writer.add_scalar("eval/mean_total_power_demand_rolling", rolling_mean, steps)

            # if not (steps % 5000):
            #    logger.info(f'\n===> reward_weight \n{info['reward_weight']}\n')





            # -----------------------------------------

            # if steps == 10000:
            #    # pref_env.set_pref('comfort')
            #    pref_env.set_pref('economical')
            
            # if steps < 10000:
            #    power_demand_0 += info['total_power_demand']
            # else:
            #    power_demand_1 += info['total_power_demand']

            # -----------------------------------------


            '''
            if steps == 8600:
               env.set_pref('economical')
               # env.set_pref('comfort')
            elif steps == 14000:
               env.set_pref('comfort')
            '''

            # if steps < 8600:
            #    power_demand_0 += info['total_power_demand']
            # elif steps < 14000:
            #    power_demand_1 += info['total_power_demand']
            # else:
            #    power_demand_2 += info['total_power_demand']

            # -----------------------------------------

            # if not (steps % 8600):
            #    if steps / 8600 == 1:
            #       env.set_pref('economical')
            #    elif steps / 8600 == 2:
            #       env.set_pref('comfort')

            # if steps < 8600:
            #    power_demand_0 += info['total_power_demand']
            # elif steps < 17200:
            #    power_demand_1 += info['total_power_demand']
            # else:
            #    power_demand_2 += info['total_power_demand']

            steps += 1

         logger.info(f"[Episode {episode}] Total reward: {total_reward}, Steps: {steps}")

         '''
         print(f'\n===> power_demand_0 \n{power_demand_0}\n')
         print(f'\n===> power_demand_1 \n{power_demand_1}\n')
         print(f'\n===> power_demand_2 \n{power_demand_2}\n')
         '''

         # TensorBoard 기록
         self.writer.add_scalar('eval/Total_Reward', total_reward, episode)
         self.writer.add_scalar('eval/Episode_Length', steps, episode)

      # --- 종료 ---
      self.writer.close()
      self.env.close()



if __name__ == '__main__':
   # -----------------------------------------------------------------------------
   # 에이전트 설정 파라미터
   # ----------------------------------------------------------------------------- 
   env_config_params = {
      'environment_id' : 'Eplus-DCLight-normal-continuous-stochastic-v1',
      'episodes' : 50, 
      'experiment_date' : datetime.today().strftime('%Y-%m-%d_%H:%M'),
      'experiment_algorithm' : '[ SB3_PPO ]',
      # 'experiment_detail' : '[ Evaluation GreenLevel ]',
      'experiment_detail' : '[ Evaluation YellowLevel ]',
   }

   load_model_params = {
      # 'model_path' : './trained_model/model_green_level.zip',
      'model_path' : './trained_model/model_yellow_level.zip',
      'log_dir' : './tb_logs/trained_model_evaluation'
      # 'log_dir' : './tb_logs'
   }

   # Agent 객체 생상
   agent = Agent(**env_config_params)
   
   # self.env = ActionAdjustmentWrapper(self.env)
   # self.env = TransformAction(self.env, transform_action, self.env.action_space)  # 액션 변환
   # self.env = NormalizeAction(self.env)  # 액션 정규화
   # self.env = NormalizeObservation(self.env)  # 관찰값 정규화
   # self.env = LoggerWrapper(self.env)  # 관찰값 정규화
   # self.env = CSVLogger(self.env)

   # env 에 wrapper 들 추가
   env_wrappers_list = [ (ObsRewardWrapper, {}), 
                        (NormalizeAction, {}),
                        (NormalizeObservation, {}) ]
   agent._add_wrappers(env_wrappers_list)

   agent.load_model(**load_model_params)

   agent.run()


