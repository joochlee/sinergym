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
episodes = 5  # 훈련 에피소드 수

# 실험 이름 생성 (날짜/시간 포함)
experiment_date = datetime.today().strftime('%Y-%m-%d_%H:%M')
experiment_name = 'SB3_PPO-' + environment + \
    '-episodes-' + str(episodes)
experiment_name += '_' + experiment_date


# 훈련용 환경 생성
env = gym.make(environment, env_name=experiment_name)
# 평가용 환경 생성
eval_env = gym.make(environment, env_name=experiment_name + '_EVALUATION')

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

        # print(f'\n===> self.locals.info \n{info}\n')

        # TensorBoard에 현재 스텝의 보상값을 기록 (카테고리: result/reward, 값: reward, x축: timesteps)
        self.writer.add_scalar("result/reward", info.get('reward'), self.num_timesteps)
        # 에피소드 정보 처리
        infos = self.locals.get("infos")
        if infos is not None:
            for info in infos:
                # Monitor 래퍼가 episode 종료 시 info["episode"] 추가
                if "episode" in info.keys():
                    ep_r = info["episode"]["r"]  # 해당 episode의 총 보상
                    self.ep_rewards.append(ep_r)

                    # rolling 평균 계산 (최근 100개 에피소드)
                    mean_r = sum(self.ep_rewards) / len(self.ep_rewards)
                    print(f'\n===> sum(self.ep_rewards) \n{sum(self.ep_rewards)}\n')
                    print(f'\n===> len(self.ep_rewards) \n{len(self.ep_rewards)}\n')
                    self.writer.add_scalar("custom/ep_rew_mean", mean_r, self.num_timesteps)

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
model = PPO(
    'MlpPolicy', 
    env, 
    verbose=1, 
    tensorboard_log='./tb_logs',
    device=device,  # GPU 사용 설정
    learning_rate=3e-4,
    n_steps=2048,  # GPU 사용 시 더 큰 배치 크기 권장
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs={
        'net_arch': [dict(pi=[256, 256], vf=[256, 256])]  # GPU 사용 시 더 큰 네트워크
    } if device == 'cuda' else None
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

# 콜백들을 리스트에 추가
callbacks.append(eval_callback)
callbacks.append(SinergymTBCallback())
callback = CallbackList(callbacks)  # 콜백 리스트 생성

# 총 훈련 타임스텝 계산
# (에피소드 수 × (에피소드당 타임스텝 - 1))
timesteps = episodes * (env.get_wrapper_attr('timestep_per_episode') - 1)

print(f'\n===> env \n{env}\n')

print('===> What is all the variables names which conform my Gymnasium observation?: {}'.format(
    env.get_wrapper_attr('observation_variables')))
print('===> What is all the variables names which conform my Gymnasium action?: {}'.format(
    env.get_wrapper_attr('action_variables')))
print('===> Is there an episode executing?: {}'.format(
    env.get_wrapper_attr('is_running')))
print('===> What is the episode run period?: {}'.format(
    env.get_wrapper_attr('runperiod')))
print('===> Episode length in seconds?: {}'.format(
    env.get_wrapper_attr('episode_length')))
print('===> How many steps have an episode?: {}'.format(
    env.get_wrapper_attr('timestep_per_episode')))
print('===> How often do steps changed?: {}'.format(
    env.get_wrapper_attr('step_size')))
print('===> What are the available zones?: {}'.format(
    env.get_wrapper_attr('zone_names')))
print('===> Where is the output path of Sinergym environment?: {}'.format(
    env.get_wrapper_attr('workspace_path')))
print('===> Where is the building used?: {}'.format(
    env.get_wrapper_attr('building_path')))
print('===> Where is the weather used?: {}'.format(
    env.get_wrapper_attr('weather_path')))
print('===> Is the action space discrete?: {}'.format(
    env.get_wrapper_attr('is_discrete')))


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
