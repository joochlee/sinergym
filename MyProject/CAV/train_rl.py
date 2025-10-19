import logging

import gymnasium as gym
import numpy as np

from gymnasium.wrappers import TransformAction
from gymnasium import spaces


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
import torch


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


# 로거 설정
terminal_logger = TerminalLogger()
logger = terminal_logger.getLogger(
   name='MAIN',
   level=logging.INFO
)

# 환경 설정
environment = 'Eplus-CompassCAV-normal-continuous-stochastic-v1'  # Sinergym 환경 ID
episodes = 1  # 훈련 에피소드 수

# 실험 이름 생성 (날짜/시간 포함)
experiment_date = datetime.today().strftime('%Y-%m-%d_%H:%M')
experiment_name = 'SB3_PPO-' + environment + \
   '-episodes-' + str(episodes)
experiment_name += '_' + experiment_date


# 훈련용 환경 생성
env = gym.make(environment, env_name=experiment_name)
# 평가용 환경 생성
eval_env = gym.make(environment, env_name=experiment_name + '_EVALUATION')

print(f'\n===> workspace_path \n{env.get_wrapper_attr('workspace_path')}\n')


# Create environment and apply wrappers for normalization and logging
# env = gym.make('Eplus-CompassCAV-normal-continuous-stochastic-v1')

# 훈련 환경에 래퍼 적용
env = TransformAction(env, transform_action, env.action_space)  # 액션 변환
env = NormalizeAction(env)  # 액션 정규화
env = NormalizeObservation(env)  # 관찰값 정규화
env = LoggerWrapper(env)  # 로깅 래퍼
env = CSVLogger(env)  # CSV 로깅
env = Monitor(env)  # 모니터링

# 평가 환경에 래퍼 적용
eval_env = TransformAction(eval_env, transform_action, eval_env.action_space)
eval_env = NormalizeObservation(eval_env)
eval_env = NormalizeAction(eval_env)
eval_env = LoggerWrapper(eval_env)
eval_env = CSVLogger(eval_env)
eval_env = Monitor(eval_env)


class SinergymTBCallback(BaseCallback):
   def __init__(self, verbose=0):
      super().__init__(verbose)
      self.ep_rewards = deque(maxlen=100)  # 최근 100개 에피소드 보상 저장
      self.step_count = 0  # 콜백 호출 횟수 추적

      # self._energy_buffer = []  # 에너지 버퍼 (현재 사용하지 않음)

   def _on_training_start(self) -> None:
      """훈련 시작 시 TensorBoard writer 초기화"""
      self.writer = SummaryWriter(log_dir = self.logger.dir)
      return super()._on_training_start()

   def _on_step(self) -> bool:
      """
      각 스텝마다 호출되는 함수
      보상값과 에피소드 정보를 TensorBoard에 기록
      """
      # 주석 처리된 에너지/불편함 로깅 코드
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
      
      # 환경에서 받은 정보 중 마지막 정보를 가져옴
      info = self.locals.get('infos')[-1]


      # TensorBoard에 현재 스텝의 보상값을 기록 (카테고리: result/reward, 값: reward, x축: timesteps)
      reward_value = info.get('reward', 0.0)
      self.writer.add_scalar("perf/step_reward", reward_value, self.num_timesteps)
      
      # 매 10 스텝마다 강제로 flush (즉시 반영)
      if self.num_timesteps % 10 == 0:
         self.writer.flush()
         
      # 디버깅: 매 100 스텝마다 로그 출력
      # self.step_count += 1
      # if self.num_timesteps % 100 == 0:
      #     print(f"Step {self.num_timesteps}: Reward = {reward_value:.4f}, Callback calls = {self.step_count}")
      
      # 에피소드 정보 처리
      infos = self.locals.get("infos")
      if infos is not None:
         for info in infos:
            # Monitor 래퍼가 episode 종료 시 info["episode"] 추가
            if "episode" in info.keys():
               print(f'\n===> self.locals.info \n{info}\n')
               ep_r = info["episode"]["r"]  # 해당 episode의 총 보상
               ep_length = info["episode"]["l"]  # 에피소드 길이
               self.ep_rewards.append(ep_r)

               # rolling 평균 계산 (최근 100개 에피소드)
               mean_r = sum(self.ep_rewards) / len(self.ep_rewards)
               print(f'\n🎯 EPISODE COMPLETED! 🎯')
               print(f'Episode #{len(self.ep_rewards)}: Reward = {ep_r:.2f}, Length = {ep_length}')
               print(f'Rolling Average: {mean_r:.2f} (over {len(self.ep_rewards)} episodes)')
               print(f"📊 Recent episodes: {list(self.ep_rewards)[-5:]}")  # 최근 5개 에피소드
               print(f'Current Timestep: {self.num_timesteps}')
               self.writer.add_scalar("perf/ep_rew_mean_rolling", mean_r, self.num_timesteps)

      # 스텝별 보상값 기록
      # step_rewards = self.locals.get('rewards')
      # self.writer.add_scalar("result/step_reward", step_rewards, self.num_timesteps)

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
      """훈련 종료 시 TensorBoard writer 정리"""
      self.writer.flush()  # 버퍼에 남은 데이터 모두 기록
      self.writer.close()  # writer 종료



# from collections import deque
# import numpy as np
# import torch as th
# from torch.utils.tensorboard import SummaryWriter
# from stable_baselines3.common.callbacks import BaseCallback
# from gymnasium import spaces


class RichPPOLogger(BaseCallback):
   """
   PPO 학습에서 추가 지표를 TensorBoard로 기록하는 고급 콜백 클래스
   
   기록하는 메트릭들:
   - perf/step_reward_mean: 스텝별 보상 평균 (학습 진행 추적)
   - perf/ep_rew_mean_rolling: 최근 N개 에피소드의 평균 보상 (성능 개선 추적)
   - advantage/mean, std, min, max: rollout 기준 advantage 통계 (학습 안정성)
   - policy/entropy_est_preupdate: 정책 엔트로피 추정값 (탐험 정도)
   - value/v_mean_rollout: value 함수 예측 평균 (가치 추정 품질)
   - action/mean, std: 액션 통계 (정책 행동 패턴)
   - action/near_bound_frac: 경계 근처 액션 비율 (정책 다양성)
   - action/freq/a{idx}: 이산 액션의 각 행동 빈도 (정책 분포)

   주의사항:
   - KL divergence, clip fraction, 손실 등은 SB3가 자동으로 'train/*' 태그로 기록
   - 스텝 단위 로깅은 I/O 부하가 크므로 step_log_every로 간격 조절 필요
   - episode_window는 rolling 평균 계산에 사용되는 에피소드 수
   """

   def __init__(
      self,
      episode_window: int = 100,      # rolling 평균 계산에 사용할 에피소드 수
      step_log_every: int = 10,       # 스텝 단위 기록 간격 (성능에 영향)
      near_bound_eps: float = 1e-3,   # Box 액션에서 경계 부근 판정 임계값
      verbose: int = 0,               # 로그 출력 레벨
   ):
      """
      RichPPOLogger 초기화
      
      Args:
          episode_window: rolling 평균 계산에 사용할 최근 에피소드 수
          step_log_every: 스텝별 로깅 간격 (너무 작으면 성능 저하)
          near_bound_eps: 연속 액션에서 경계 근처 판정을 위한 임계값
          verbose: 로그 출력 레벨
      """
      super().__init__(verbose)
      
      # 로깅 설정
      self.episode_window = episode_window    # rolling 평균 윈도우 크기
      self.step_log_every = step_log_every    # 스텝 로깅 간격
      self.near_bound_eps = near_bound_eps    # 경계 근처 판정 임계값

      # TensorBoard writer (훈련 시작 시 초기화)
      self.writer = None
      
      # 액션 스페이스 관련 변수들
      self.is_discrete = False        # 이산 액션 여부
      self.is_box = False            # 연속 액션 여부
      self.n_actions = None          # 이산 액션의 경우 액션 수
      self.act_low = None            # 연속 액션의 하한값
      self.act_high = None           # 연속 액션의 상한값

      # 에피소드 보상 추적 변수들
      self.curr_ep_returns = None    # 현재 에피소드의 누적 보상 (환경별)
      self.recent_ep_returns = None  # 최근 에피소드들의 보상 (rolling 평균용)
      self.action_counts = None      # 이산 액션의 경우 각 액션 선택 횟수

   # ==== Life‑cycle hooks ====

   def _on_training_start(self) -> None:
      """
      훈련 시작 시 호출되는 초기화 함수
      - TensorBoard writer 설정
      - 액션 스페이스 분석 및 설정
      - 에피소드 보상 추적 변수 초기화
      """
      # SB3 내부 로거 디렉터리에 맞춰 SummaryWriter 생성 (tb_log_name과 동일 폴더)
      self.writer = SummaryWriter(log_dir=self.logger.dir)

      # 액션 스페이스 타입 판정 및 설정
      aspace = self.training_env.action_space
      self.is_discrete = isinstance(aspace, spaces.Discrete)  # 이산 액션 여부
      self.is_box = isinstance(aspace, spaces.Box)           # 연속 액션 여부
      
      if self.is_discrete:
         # 이산 액션의 경우: 액션 수와 카운터 배열 설정
         self.n_actions = aspace.n
         self.action_counts = np.zeros(self.n_actions, dtype=np.int64)
      elif self.is_box:
         # 연속 액션의 경우: 액션 범위 설정
         self.act_low = np.array(aspace.low)   # 액션 하한값
         self.act_high = np.array(aspace.high) # 액션 상한값

      # 에피소드 보상 추적을 위한 변수 초기화
      n_envs = self.training_env.num_envs                    # 병렬 환경 수
      self.curr_ep_returns = np.zeros(n_envs, dtype=np.float32)  # 현재 에피소드 누적 보상
      self.recent_ep_returns = deque(maxlen=self.episode_window)  # 최근 에피소드 보상 (rolling 평균용)

   def _on_step(self) -> bool:
      """
      매 스텝마다 호출되는 함수 (rollout 중에만)
      - 스텝별 보상과 액션 통계 수집
      - 에피소드 완료 감지 및 rolling 평균 계산
      - TensorBoard에 주기적으로 메트릭 기록
      """
      # PPO rollout에서 수집된 데이터 가져오기
      rewards = np.array(self.locals.get("rewards"))  # 현재 스텝의 보상들
      dones   = np.array(self.locals.get("dones"))    # 에피소드 종료 여부
      actions = self.locals.get("actions")            # 선택된 액션들

      # ----- 스텝별 보상 누적 및 에피소드 완료 감지 -----
      self.curr_ep_returns += rewards  # 각 환경별로 에피소드 보상 누적
      
      # 에피소드가 완료된 환경들 처리
      for i, done in enumerate(dones):
         if done:
               # 완료된 에피소드의 총 보상을 rolling 평균 리스트에 추가
               self.recent_ep_returns.append(self.curr_ep_returns[i])
               self.curr_ep_returns[i] = 0.0  # 해당 환경의 보상 초기화

      # ----- 주기적 메트릭 기록 (성능 최적화를 위해 step_log_every 간격으로) -----
      if self.num_timesteps % self.step_log_every == 0:
         # 스텝별 보상 평균 기록 (학습 진행 상황 추적)
         self.writer.add_scalar("perf/step_reward_mean", float(rewards.mean()), self.num_timesteps)
         
         # rolling 평균 에피소드 보상 기록 (성능 개선 추적)
         if len(self.recent_ep_returns) > 0:
               rolling_mean = float(np.mean(self.recent_ep_returns))
               if self.num_timesteps % 1000 == 0:  # 1000 스텝마다 디버깅 출력
                  print(f"📊 Rolling Average: {rolling_mean:.2f} (over {len(self.recent_ep_returns)} episodes)")
                  print(f"📊 Recent episodes: {list(self.recent_ep_returns)[-5:]}")  # 최근 5개 에피소드
               self.writer.add_scalar(
                  "perf/ep_rew_mean_rolling",
                  rolling_mean,
                  self.num_timesteps,
               )

      # ----- 액션 통계 수집 및 기록 (스텝 기반) -----
      if actions is not None and self.num_timesteps % self.step_log_every == 0:
         a = np.asarray(actions)  # 액션을 numpy 배열로 변환
         
         if self.is_discrete:
            # 이산 액션의 경우: 액션 분포 분석
            flat = a.reshape(-1)  # 1차원으로 평탄화
            
            # 각 액션 선택 횟수 누적 (rollout 끝에서 빈도로 기록)
            for act in flat:
               if 0 <= int(act) < self.n_actions:
                  self.action_counts[int(act)] += 1
            
            # 액션 통계 기록 (평균, 표준편차)
            self.writer.add_scalar("action/mean", float(flat.mean()), self.num_timesteps)
            self.writer.add_scalar("action/std", float(flat.std()), self.num_timesteps)
            
         elif self.is_box:
            # 연속 액션의 경우: 액션 분포 및 경계 근처 비율 분석
            self.writer.add_scalar("action/mean", float(a.mean()), self.num_timesteps)  # 액션 평균
            self.writer.add_scalar("action/std", float(a.std()), self.num_timesteps)   # 액션 표준편차
            
            # 경계 근처 액션 비율 계산 (정책 다양성 지표)
            near_low  = np.isclose(a, self.act_low,  atol=self.near_bound_eps).sum()   # 하한 근처
            near_high = np.isclose(a, self.act_high, atol=self.near_bound_eps).sum()   # 상한 근처
            total = a.size
            self.writer.add_scalar(
               "action/near_bound_frac",
               float((near_low + near_high) / max(total, 1)),
               self.num_timesteps,
            )

      return True

   def _on_rollout_end(self) -> None:
      """
      Rollout 수집이 완료된 직후 호출 (학습 업데이트 직전)
      - rollout_buffer의 데이터를 이용해 고급 메트릭 계산 및 기록
      - advantage, 정책 엔트로피, value 함수, 액션 분포 등 분석
      """
      rb = self.model.rollout_buffer  # PPO의 rollout buffer 참조

      # ----- Advantage 통계 분석 (학습 안정성 지표) -----
      adv = rb.advantages  # GAE로 계산된 advantage 값들
      if isinstance(adv, torch.Tensor):
         adv = adv.detach().cpu().numpy()  # GPU 텐서를 CPU numpy로 변환
      adv_flat = adv.reshape(-1)  # 1차원으로 평탄화
      
      # Advantage 통계 기록 (학습 안정성 모니터링)
      self.writer.add_scalar("advantage/mean", float(adv_flat.mean()), self.num_timesteps)  # 평균
      self.writer.add_scalar("advantage/std",  float(adv_flat.std()),  self.num_timesteps)  # 표준편차
      self.writer.add_scalar("advantage/min",  float(adv_flat.min()),  self.num_timesteps)  # 최솟값
      self.writer.add_scalar("advantage/max",  float(adv_flat.max()),  self.num_timesteps)  # 최댓값

      # ----- 정책 엔트로피 추정 (학습 전 상태) -----
      # 정책 엔트로피: E[-log π(a|s)] ≈ -(rollout에 저장된 old_log_prob의 평균)
      # 높은 엔트로피 = 더 많은 탐험, 낮은 엔트로피 = 더 결정적인 정책
      old_lp = rb.log_probs  # rollout 중 저장된 로그 확률
      if isinstance(old_lp, torch.Tensor):
         old_lp = old_lp.detach().cpu().numpy()  # GPU 텐서를 CPU numpy로 변환
      entropy_est = float(-old_lp.mean())  # 엔트로피 추정값 계산
      self.writer.add_scalar("policy/entropy_est_preupdate", entropy_est, self.num_timesteps)

      # ----- Value 함수 예측 품질 분석 -----
      vals = rb.values  # rollout 중 저장된 value 함수 예측값들
      if isinstance(vals, torch.Tensor):
         vals = vals.detach().cpu().numpy()  # GPU 텐서를 CPU numpy로 변환
      self.writer.add_scalar("value/v_mean_rollout", float(vals.mean()), self.num_timesteps)  # value 예측 평균

      # ----- 액션 분포 분석 (rollout 기반) -----
      acts = rb.actions  # rollout 중 저장된 모든 액션들
      if isinstance(acts, torch.Tensor):
         acts = acts.detach().cpu().numpy()  # GPU 텐서를 CPU numpy로 변환

      if self.is_discrete:
         # 이산 액션의 경우: 각 액션 선택 빈도 분석
         total = self.action_counts.sum()  # 총 액션 선택 횟수
         if total > 0:
               # 각 액션별 선택 빈도를 TensorBoard에 기록
               for a_idx in range(self.n_actions):
                  freq = float(self.action_counts[a_idx] / total)
                  self.writer.add_scalar(f"action/freq/a{a_idx}", freq, self.num_timesteps)
         # 다음 rollout을 위해 액션 카운터 리셋
         self.action_counts[:] = 0

      elif self.is_box:
         # 연속 액션의 경우: rollout 기반 액션 분포 분석
         # 경계 근처 액션 비율 계산 (정책 다양성 지표)
         near_low  = np.isclose(acts, self.act_low,  atol=self.near_bound_eps).sum()   # 하한 근처
         near_high = np.isclose(acts, self.act_high, atol=self.near_bound_eps).sum()   # 상한 근처
         total = acts.size
         self.writer.add_scalar(
               "action/near_bound_frac_rollout",
               float((near_low + near_high) / max(total, 1)),
               self.num_timesteps,
         )
         # rollout 기반 액션 통계
         self.writer.add_scalar("action/rollout_mean", float(acts.mean()), self.num_timesteps)  # 평균
         self.writer.add_scalar("action/rollout_std",  float(acts.std()),  self.num_timesteps)  # 표준편차

      # 참고: KL divergence, clip fraction, 손실 등은 SB3가 자동으로 'train/*' 태그로 기록

   def _on_training_end(self) -> None:
      """
      훈련 종료 시 호출되는 정리 함수
      - TensorBoard writer의 버퍼를 모두 기록하고 종료
      - 메모리 누수 방지
      """
      if self.writer is not None:
         self.writer.flush()  # 버퍼에 남은 모든 데이터를 디스크에 기록
         self.writer.close()  # TensorBoard writer 종료






# GPU 사용 가능 여부 확인
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"사용 중인 디바이스: {device}")
if torch.cuda.is_available():
   print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
   print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
   print("GPU를 사용할 수 없습니다. CPU로 실행됩니다.")


# PPO 모델 생성 (기본 하이퍼 파라미터사용)
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
model = PPO(
   'MlpPolicy', 
   env, 
   verbose=1, 
   tensorboard_log='./tb_logs',
   device=device,  # GPU 사용 설정
   # learning_rate=3e-4,
   n_steps=2048,  # 더 큰 rollout로 안정적인 학습
   batch_size=256,
   n_epochs=10,
   # gamma=0.99,
   # gae_lambda=0.95,
   # clip_range=0.2,
   # ent_coef=0.0,
   # vf_coef=0.5,
   # max_grad_norm=0.5,
   # policy_kwargs={
   #    'net_arch': [dict(pi=[256, 256], vf=[256, 256])]  # GPU 사용 시 더 큰 네트워크
   # } if device == 'cuda' else None
)


# 콜백 리스트 초기화
callbacks = []

# 평가 콜백 설정 (모델 저장 및 평가 로깅)
eval_callback = LoggerEvalCallback(
   eval_env=eval_env,  # 평가 환경
   train_env=env,  # 훈련 환경
   n_eval_episodes=1,  # 평가 에피소드 수
   eval_freq_episodes=2,  # 평가 주기 (2 에피소드마다)
   deterministic=True)  # 결정적 정책 사용


# 4) 통합 로거 콜백
rich_cb = RichPPOLogger(
    episode_window=100,  # 더 큰 윈도우로 안정적인 평균
    step_log_every=10,  # 스텝 로그 간격 (너무 작으면 느려질 수 있음)
)



# 콜백들을 리스트에 추가
# callbacks.append(eval_callback)
callbacks.append(SinergymTBCallback())
# callbacks.append(rich_cb)

callback = CallbackList(callbacks)  # 콜백 리스트 생성

# 총 훈련 타임스텝 계산
# (에피소드 수 × (에피소드당 타임스텝 - 1))
timesteps = episodes * (env.get_wrapper_attr('timestep_per_episode') - 1)

# print(f'\n===> env \n{env}\n')

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


# 모델 훈련 실행
model.learn(
   total_timesteps=timesteps,  # 총 훈련 타임스텝
   callback=callback,  # 콜백 함수들
   log_interval=100,  # 로그 출력 주기
   tb_log_name='ppo_cav')  # TensorBoard 로그 이름

# 훈련된 모델 저장
model.save(env.get_wrapper_attr('workspace_path') + '/model')

# 주석 처리된 수동 에피소드 실행 코드
# (현재는 PPO 모델이 자동으로 에피소드를 실행하므로 불필요)
# for i in range(episodes):
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
#         np.mean(rewards), sum(rewards)))

# 환경 종료
env.close()
