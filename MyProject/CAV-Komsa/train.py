"""
HVAC RL Training - AHU & FCU Agents (Sinergym + Gymnasium + SB3)
=================================================================
Sinergym은 gymnasium 인터페이스를 완전히 따릅니다.

[gymnasium 인터페이스]
  step()  → (obs, reward, terminated, truncated, info)   # 5개
  reset() → (obs, info)                                  # 튜플

[설계 방식]
  Sinergym 환경을 gymnasium.make()로 생성한 뒤,
  AHU/FCU 각각의 커스텀 로직을 gymnasium.Wrapper로 씌웁니다.
  - AHUWrapper : Sinergym obs에서 AHU 관련 변수만 추출 + AHU 보상 계산
  - FCUWrapper : Sinergym obs에서 FCU 관련 변수만 추출 + FCU 보상 계산
  두 Wrapper는 SharedStateBuffer를 통해 관측값을 교환합니다.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import sinergym                                          # noqa: F401 (등록 트리거)
import os


# ─────────────────────────────────────────────────────────
# Sinergym 환경 ID 및 변수 인덱스 설정
# (실제 사용하는 환경의 observation 변수 순서에 맞게 수정하세요)
# ─────────────────────────────────────────────────────────

SINERGYM_ENV_ID = "Eplus-5zone-hot-continuous-stochastic-v1"

# Sinergym observation 벡터에서 각 변수의 인덱스
# env.observation_space 및 env.variables 로 확인 가능
OBS_IDX = {
    # 시간
    "month":             0,
    "day":               1,
    "hour":              2,
    # 외기
    "oa_drybulb":        3,
    "oa_rh":             4,
    # 환기 (Return Air)
    "ra_temp":           5,
    # AHU
    "sat":               6,    # Supply Air Temperature
    "ahu_cooling_rate":  7,    # AHU 냉각량 (W)
    "damper_pos":        8,    # OA 댐퍼 개도
    # FCU / 각 실 온도
    "room_temp_1":       9,
    "room_temp_2":       10,
    "room_temp_3":       11,
    "room_temp_4":       12,
    "room_temp_5":       13,
    # 각 실 FCU 밸브
    "fcu_valve_1":       14,
    "fcu_valve_2":       15,
    "fcu_valve_3":       16,
    "fcu_valve_4":       17,
    "fcu_valve_5":       18,
}
N_ROOMS = 5
ROOM_TEMP_IDXS  = [OBS_IDX[f"room_temp_{i+1}"] for i in range(N_ROOMS)]
FCU_VALVE_IDXS  = [OBS_IDX[f"fcu_valve_{i+1}"] for i in range(N_ROOMS)]

# Sinergym action 벡터 인덱스
# (AHU SAT setpoint, OA 댐퍼, FCU 밸브 × 5)
ACT_IDX = {
    "sat_setpoint":  0,
    "oa_damper":     1,
    "fcu_valve_1":   2,
    "fcu_valve_2":   3,
    "fcu_valve_3":   4,
    "fcu_valve_4":   5,
    "fcu_valve_5":   6,
}
TOTAL_ACTION_DIM = 7    # Sinergym 환경 전체 action 차원


# ─────────────────────────────────────────────────────────
# 1. 공유 상태 버퍼
# ─────────────────────────────────────────────────────────

class SharedStateBuffer:
    """AHU ↔ FCU Wrapper 간 관측값 교환 채널."""

    def __init__(self, n_rooms: int = N_ROOMS):
        self.n_rooms = n_rooms
        self.reset()

    def reset(self):
        # AHU Wrapper가 쓰고, FCU Wrapper가 읽는 값
        self.sat        = 15.0
        self.ahu_load   = 0.0
        self.damper     = 20.0

        # FCU Wrapper가 쓰고, AHU Wrapper가 읽는 값
        self.room_temps    = np.full(self.n_rooms, 26.0, dtype=np.float32)
        self.fcu_valves    = np.full(self.n_rooms, 0.5, dtype=np.float32)
        self.valve_avg     = 0.5
        self.max_deviation = 0.0

        # LLM directive 기본값
        self.sat_range      = [12.0, 18.0]
        self.room_setpoints = np.full(self.n_rooms, 26.0, dtype=np.float32)
        self.w_ahu = {
            "w_enthalpy":     0.3,
            "w_load":         0.4,
            "w_sat":          0.2,
            "w_fcu_overload": 0.1,
        }
        self.w_fcu = {
            "w_comfort": 0.6,
            "w_energy":  0.2,
            "w_reheat":  0.1,
            "w_vacancy": 0.1,
        }

    def update_from_llm(self, directive: dict):
        if not directive.get("intervention_required", False):
            return
        ahu_d = directive.get("ahu_directive", {})
        fcu_d = directive.get("fcu_directive", {})
        rw    = directive.get("reward_override")

        if ahu_d.get("sat_range"):
            self.sat_range = ahu_d["sat_range"]
        if fcu_d.get("room_setpoints"):
            sp = fcu_d["room_setpoints"]
            self.room_setpoints = np.array([
                sp.get(f"room_{i+1}", self.room_setpoints[i])
                for i in range(self.n_rooms)
            ], dtype=np.float32)
        if rw:
            if rw.get("ahu"):
                self.w_ahu.update(rw["ahu"])
            if rw.get("fcu"):
                self.w_fcu.update(rw["fcu"])


# ─────────────────────────────────────────────────────────
# 2. 엔탈피 계산 유틸
# ─────────────────────────────────────────────────────────

def calc_enthalpy(drybulb_c: float, rh_pct: float) -> float:
    """습공기 엔탈피 근사 계산 (kJ/kg)."""
    T   = drybulb_c
    rh  = rh_pct / 100.0
    Pws = 0.6108 * np.exp(17.27 * T / (T + 237.3))   # kPa (Magnus)
    W   = 0.622 * rh * Pws / (101.325 - rh * Pws)     # 절대습도 kg/kg
    return 1.006 * T + W * (2501.0 + 1.86 * T)        # kJ/kg


# ─────────────────────────────────────────────────────────
# 3. AHU Wrapper
# ─────────────────────────────────────────────────────────

class AHUWrapper(gym.Wrapper):
    """
    Sinergym 환경을 AHU 에이전트 관점으로 래핑합니다.

    - observation: Sinergym obs에서 AHU 관련 변수 추출 + FCU 피드백 추가 (dim=14)
    - action     : [ΔSAT, Δdamper] (dim=2) → Sinergym 전체 action으로 합성
    - reward     : AHU 특화 보상 (엔탈피·부하·SAT범위·FCU과부하)

    step()  → (obs, reward, terminated, truncated, info)   # gymnasium
    reset() → (obs, info)                                  # gymnasium
    """

    def __init__(self, env: gym.Env, shared: SharedStateBuffer):
        super().__init__(env)
        self.shared = shared

        # AHU 에이전트 행동 공간: [ΔSAT, Δdamper]
        self.action_space = spaces.Box(
            low  = np.array([-1.0, -5.0], dtype=np.float32),
            high = np.array([ 1.0,  5.0], dtype=np.float32),
            dtype= np.float32,
        )

        # AHU 에이전트 관측 공간 (dim=14)
        obs_low = np.array([
            -10, -10,  20,    # 외기 건구, RH, 엔탈피
             15,  30,         # 환기 온도, 엔탈피
             12,   0,   0,    # SAT, 냉각부하(정규화), 댐퍼
              0,              # 엔탈피 역전 플래그
             20,   0,   0,    # FCU 평균온도, 평균밸브, 최대이탈
             -1,  -1,         # 시간 인코딩
        ], dtype=np.float32)
        obs_high = np.array([
             45, 100, 150,
             35,  90,
             22,   1, 100,
              1,
             35,   1,   5,
              1,   1,
        ], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )

        # 내부 상태
        self._sat    = 15.0
        self._damper = 20.0
        # Sinergym 전체 action 중 FCU 부분 초기값 (FCU Wrapper가 덮어씀)
        self._fcu_valves = np.full(N_ROOMS, 0.5, dtype=np.float32)

    def _extract_obs(self, sinergym_obs: np.ndarray) -> np.ndarray:
        """Sinergym obs → AHU 에이전트 관측 벡터."""
        oa_db  = sinergym_obs[OBS_IDX["oa_drybulb"]]
        oa_rh  = sinergym_obs[OBS_IDX["oa_rh"]]
        ra_t   = sinergym_obs[OBS_IDX["ra_temp"]]
        h_oa   = calc_enthalpy(oa_db, oa_rh)
        h_ra   = calc_enthalpy(ra_t, 50.0)    # 환기 RH 센서 없으면 50% 가정
        enthalpy_flag = float(h_oa > h_ra)

        hour   = sinergym_obs[OBS_IDX["hour"]]
        angle  = 2 * np.pi * hour / 24.0

        return np.array([
            oa_db, oa_rh, h_oa,
            ra_t, h_ra,
            self._sat,
            self.shared.ahu_load / 60000.0,   # W → 정규화 (60kW 기준)
            self._damper,
            enthalpy_flag,
            float(np.mean(self.shared.room_temps)),
            self.shared.valve_avg,
            self.shared.max_deviation,
            np.sin(angle), np.cos(angle),
        ], dtype=np.float32)

    def _compute_reward(self, sinergym_obs: np.ndarray) -> float:
        w = self.shared.w_ahu

        oa_db = sinergym_obs[OBS_IDX["oa_drybulb"]]
        oa_rh = sinergym_obs[OBS_IDX["oa_rh"]]
        ra_t  = sinergym_obs[OBS_IDX["ra_temp"]]
        h_oa  = calc_enthalpy(oa_db, oa_rh)
        h_ra  = calc_enthalpy(ra_t, 50.0)

        da_ratio   = self._damper / 100.0
        r_enthalpy = (1.0 - da_ratio) if h_oa > h_ra else da_ratio

        r_load = -(self.shared.ahu_load / 60000.0)

        sat_min, sat_max = self.shared.sat_range
        sat_mid  = (sat_min + sat_max) / 2.0
        sat_half = (sat_max - sat_min) / 2.0
        r_sat    = -max(0.0, abs(self._sat - sat_mid) - sat_half * 0.5)

        r_fcu_overload = -max(0.0, self.shared.valve_avg - 0.85)

        return float(
            w["w_enthalpy"]     * r_enthalpy +
            w["w_load"]         * r_load +
            w["w_sat"]          * r_sat +
            w["w_fcu_overload"] * r_fcu_overload
        )

    def _build_sinergym_action(self) -> np.ndarray:
        """AHU 현재값 + FCU 공유값 → Sinergym 전체 action 벡터."""
        action = np.zeros(TOTAL_ACTION_DIM, dtype=np.float32)
        action[ACT_IDX["sat_setpoint"]] = self._sat
        action[ACT_IDX["oa_damper"]]    = self._damper / 100.0  # 0~1 정규화
        for i in range(N_ROOMS):
            action[ACT_IDX[f"fcu_valve_{i+1}"]] = self.shared.fcu_valves[i]
        return action

    # ── gymnasium 스타일: step() → 5개 ──
    def step(self, action: np.ndarray):
        # AHU 행동 적용 (증분)
        sat_min, sat_max = self.shared.sat_range
        self._sat    = float(np.clip(self._sat    + action[0], sat_min, sat_max))
        self._damper = float(np.clip(self._damper + action[1], 10.0, 100.0))

        # 공유 버퍼에 기록 (FCU Wrapper가 읽어감)
        self.shared.sat    = self._sat
        self.shared.damper = self._damper

        # Sinergym step
        sinergym_action              = self._build_sinergym_action()
        sg_obs, _, terminated, truncated, info = self.env.step(sinergym_action)

        # 공유 버퍼 업데이트 (Sinergym info에서 냉각량 읽기)
        self.shared.ahu_load = float(
            info.get("cooling_energy", info.get("total_power", 0.0))
        )

        obs    = self._extract_obs(sg_obs)
        reward = self._compute_reward(sg_obs)

        info["ahu_sat"]    = self._sat
        info["ahu_damper"] = self._damper
        info["ahu_reward"] = reward

        return obs, reward, terminated, truncated, info  # ← 5개 반환

    # ── gymnasium 스타일: reset() → (obs, info) ──
    def reset(self, seed=None, options=None):
        sg_obs, info = self.env.reset(seed=seed, options=options)  # ← (obs, info)
        self._sat    = 15.0
        self._damper = 20.0
        self.shared.reset()
        obs = self._extract_obs(sg_obs)
        return obs, info                                            # ← (obs, info)


# ─────────────────────────────────────────────────────────
# 4. FCU Wrapper
# ─────────────────────────────────────────────────────────

class FCUWrapper(gym.Wrapper):
    """
    Sinergym 환경을 FCU 에이전트 관점으로 래핑합니다.

    - observation: 각 실 온도·편차·밸브 + AHU 피드백 + 시간 인코딩 (dim=19)
    - action     : 각 실 냉수밸브 Δ × 5 (dim=5) → Sinergym 전체 action으로 합성
    - reward     : FCU 특화 보상 (쾌적도·에너지·재열방지·비재실절약)

    step()  → (obs, reward, terminated, truncated, info)   # gymnasium
    reset() → (obs, info)                                  # gymnasium
    """

    def __init__(self, env: gym.Env, shared: SharedStateBuffer):
        super().__init__(env)
        self.shared = shared

        # FCU 에이전트 행동 공간: 5실 밸브 Δ
        self.action_space = spaces.Box(
            low  = np.full(N_ROOMS, -0.1, dtype=np.float32),
            high = np.full(N_ROOMS,  0.1, dtype=np.float32),
            dtype= np.float32,
        )

        # FCU 에이전트 관측 공간 (dim = N_ROOMS*3 + 2 + 2 = 19)
        obs_dim = N_ROOMS * 3 + 4
        self.observation_space = spaces.Box(
            low  = np.full(obs_dim, -50.0, dtype=np.float32),
            high = np.full(obs_dim,  50.0, dtype=np.float32),
            dtype= np.float32,
        )

        self._valves = np.full(N_ROOMS, 0.5, dtype=np.float32)

    def _extract_obs(self, sinergym_obs: np.ndarray) -> np.ndarray:
        """Sinergym obs → FCU 에이전트 관측 벡터."""
        room_temps = sinergym_obs[ROOM_TEMP_IDXS].astype(np.float32)
        deviations = room_temps - self.shared.room_setpoints

        hour  = sinergym_obs[OBS_IDX["hour"]]
        angle = 2 * np.pi * hour / 24.0

        # 공유 버퍼에 실온 기록 (AHU Wrapper가 읽어감)
        self.shared.room_temps = room_temps

        return np.concatenate([
            room_temps,
            deviations,
            self._valves,
            [self.shared.sat,
             self.shared.ahu_load / 60000.0],
            [np.sin(angle), np.cos(angle)],
        ]).astype(np.float32)

    def _compute_reward(self, sinergym_obs: np.ndarray) -> float:
        w          = self.shared.w_fcu
        room_temps = sinergym_obs[ROOM_TEMP_IDXS].astype(np.float32)
        deviations = room_temps - self.shared.room_setpoints
        delta      = 0.5    # dead-band °C

        r_comfort = -float(np.mean(np.maximum(0.0, np.abs(deviations) - delta)))
        r_energy  = -float(np.mean(self._valves))

        sat_low_flag = float(self.shared.sat < 13.5)
        r_reheat     = -sat_low_flag * (1.0 - float(np.mean(self._valves)))

        hour = sinergym_obs[OBS_IDX["hour"]]
        vacancy_mask = np.zeros(N_ROOMS) if (hour < 8 or hour > 19) \
                       else np.ones(N_ROOMS)
        r_vacancy = -float(np.mean((1.0 - vacancy_mask) * self._valves))

        return float(
            w["w_comfort"] * r_comfort +
            w["w_energy"]  * r_energy  +
            w["w_reheat"]  * r_reheat  +
            w["w_vacancy"] * r_vacancy
        )

    def _build_sinergym_action(self) -> np.ndarray:
        """FCU 현재값 + AHU 공유값 → Sinergym 전체 action 벡터."""
        action = np.zeros(TOTAL_ACTION_DIM, dtype=np.float32)
        action[ACT_IDX["sat_setpoint"]] = self.shared.sat
        action[ACT_IDX["oa_damper"]]    = self.shared.damper / 100.0
        for i in range(N_ROOMS):
            action[ACT_IDX[f"fcu_valve_{i+1}"]] = self._valves[i]
        return action

    # ── gymnasium 스타일: step() → 5개 ──
    def step(self, action: np.ndarray):
        # FCU 밸브 증분 적용
        self._valves = np.clip(self._valves + action, 0.0, 1.0).astype(np.float32)

        # 공유 버퍼에 기록 (AHU Wrapper가 읽어감)
        self.shared.fcu_valves    = self._valves.copy()
        self.shared.valve_avg     = float(np.mean(self._valves))
        self.shared.max_deviation = float(
            np.max(np.abs(self.shared.room_temps - self.shared.room_setpoints))
        )

        # Sinergym step
        sinergym_action                      = self._build_sinergym_action()
        sg_obs, _, terminated, truncated, info = self.env.step(sinergym_action)

        obs    = self._extract_obs(sg_obs)
        reward = self._compute_reward(sg_obs)

        info["fcu_valves"]     = self._valves.copy()
        info["room_temps"]     = self.shared.room_temps.copy()
        info["max_deviation"]  = self.shared.max_deviation
        info["fcu_reward"]     = reward

        return obs, reward, terminated, truncated, info  # ← 5개 반환

    # ── gymnasium 스타일: reset() → (obs, info) ──
    def reset(self, seed=None, options=None):
        sg_obs, info = self.env.reset(seed=seed, options=options)  # ← (obs, info)
        self._valves = np.full(N_ROOMS, 0.5, dtype=np.float32)
        obs = self._extract_obs(sg_obs)
        return obs, info                                            # ← (obs, info)


# ─────────────────────────────────────────────────────────
# 5. 커스텀 콜백
# ─────────────────────────────────────────────────────────

class HVACTrainingCallback(BaseCallback):
    def __init__(self, agent_name: str, log_freq: int = 5000, verbose: int = 0):
        super().__init__(verbose)
        self.agent_name = agent_name
        self.log_freq   = log_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            if len(self.model.ep_info_buffer) > 0:
                mean_rew = np.mean([ep["r"] for ep in self.model.ep_info_buffer])
                print(f"[{self.agent_name}] step={self.n_calls:6d} | "
                      f"mean_ep_reward={mean_rew:.3f}")
        return True


# ─────────────────────────────────────────────────────────
# 6. 환경 팩토리
# ─────────────────────────────────────────────────────────

def make_ahu_env(shared: SharedStateBuffer, env_id: str = SINERGYM_ENV_ID):
    """AHU Wrapper가 씌워진 Sinergym 환경 생성."""
    def _init():
        base_env = gym.make(env_id)         # gymnasium.make() 사용
        env      = AHUWrapper(base_env, shared)
        env      = Monitor(env)
        return env
    return _init


def make_fcu_env(shared: SharedStateBuffer, env_id: str = SINERGYM_ENV_ID):
    """FCU Wrapper가 씌워진 Sinergym 환경 생성."""
    def _init():
        base_env = gym.make(env_id)         # gymnasium.make() 사용
        env      = FCUWrapper(base_env, shared)
        env      = Monitor(env)
        return env
    return _init


# ─────────────────────────────────────────────────────────
# 7. 학습 함수
# ─────────────────────────────────────────────────────────

def train_ahu_agent(shared: SharedStateBuffer,
                    env_id: str = SINERGYM_ENV_ID,
                    total_timesteps: int = 200_000,
                    save_dir: str = "./models") -> SAC:
    """1단계: AHU 에이전트 단독 학습."""
    os.makedirs(save_dir, exist_ok=True)

    env = DummyVecEnv([make_ahu_env(shared, env_id)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = DummyVecEnv([make_ahu_env(shared, env_id)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                            training=False, clip_obs=10.0)

    model = SAC(
        policy          = "MlpPolicy",
        env             = env,
        learning_rate   = 3e-4,
        buffer_size     = 100_000,
        learning_starts = 5_000,
        batch_size      = 256,
        tau             = 0.005,
        gamma           = 0.99,
        train_freq      = 1,
        gradient_steps  = 1,
        policy_kwargs   = dict(net_arch=[256, 256]),
        verbose         = 0,
        tensorboard_log = f"{save_dir}/tb_ahu",
    )

    callbacks = [
        HVACTrainingCallback("AHU", log_freq=5000),
        EvalCallback(
            eval_env,
            best_model_save_path = f"{save_dir}/ahu_best",
            log_path             = f"{save_dir}/ahu_eval",
            eval_freq            = 10_000,
            n_eval_episodes      = 3,
            deterministic        = True,
            verbose              = 0,
        ),
        CheckpointCallback(
            save_freq   = 50_000,
            save_path   = f"{save_dir}/ahu_checkpoints",
            name_prefix = "ahu_sac",
        ),
    ]

    print("=" * 55)
    print("1단계: AHU 에이전트 학습 시작")
    print("=" * 55)
    model.learn(total_timesteps=total_timesteps, callback=callbacks,
                progress_bar=True)

    model.save(f"{save_dir}/ahu_final")
    env.save(f"{save_dir}/ahu_vecnorm.pkl")
    print("AHU 에이전트 학습 완료.")
    return model


def train_fcu_agent(shared: SharedStateBuffer,
                    ahu_model: SAC,
                    env_id: str = SINERGYM_ENV_ID,
                    total_timesteps: int = 200_000,
                    save_dir: str = "./models") -> SAC:
    """
    2단계: FCU 에이전트 학습 (AHU 고정).

    FCUWithFrozenAHU: FCU step마다 AHU 에이전트를 결정론적으로 호출해
    공유 버퍼(sat, damper)를 갱신합니다.
    두 에이전트가 같은 Sinergym 인스턴스를 공유하면 EnergyPlus 충돌이
    발생하므로 AHU/FCU 각각 별도의 Sinergym 환경 인스턴스를 사용합니다.
    """
    os.makedirs(save_dir, exist_ok=True)

    # AHU 에이전트 전용 별도 Sinergym 인스턴스
    ahu_base_env = gym.make(env_id)
    ahu_env_raw  = AHUWrapper(ahu_base_env, shared)
    ahu_obs, _   = ahu_env_raw.reset()              # gymnasium: (obs, info)

    class FCUWithFrozenAHU(FCUWrapper):
        """FCU step마다 고정된 AHU 에이전트를 실행해 공유 버퍼 갱신."""

        def step(self, action):
            nonlocal ahu_obs
            # AHU 에이전트 결정론적 실행 → sat, damper → 공유 버퍼 갱신
            ahu_action, _ = ahu_model.predict(ahu_obs, deterministic=True)
            ahu_obs, _, ahu_term, ahu_trunc, _ = ahu_env_raw.step(ahu_action)
            if ahu_term or ahu_trunc:
                ahu_obs, _ = ahu_env_raw.reset()    # gymnasium: (obs, info)

            return super().step(action)

        def reset(self, seed=None, options=None):
            nonlocal ahu_obs
            ahu_obs, _ = ahu_env_raw.reset(seed=seed)  # gymnasium: (obs, info)
            return super().reset(seed=seed, options=options)

    env = DummyVecEnv([
        lambda: Monitor(FCUWithFrozenAHU(gym.make(env_id), shared))
    ])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = DummyVecEnv([make_fcu_env(shared, env_id)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                            training=False, clip_obs=10.0)

    model = SAC(
        policy          = "MlpPolicy",
        env             = env,
        learning_rate   = 3e-4,
        buffer_size     = 100_000,
        learning_starts = 5_000,
        batch_size      = 256,
        tau             = 0.005,
        gamma           = 0.99,
        policy_kwargs   = dict(net_arch=[256, 256]),
        verbose         = 0,
        tensorboard_log = f"{save_dir}/tb_fcu",
    )

    callbacks = [
        HVACTrainingCallback("FCU", log_freq=5000),
        EvalCallback(
            eval_env,
            best_model_save_path = f"{save_dir}/fcu_best",
            log_path             = f"{save_dir}/fcu_eval",
            eval_freq            = 10_000,
            n_eval_episodes      = 3,
            deterministic        = True,
            verbose              = 0,
        ),
    ]

    print("=" * 55)
    print("2단계: FCU 에이전트 학습 시작 (AHU 고정)")
    print("=" * 55)
    model.learn(total_timesteps=total_timesteps, callback=callbacks,
                progress_bar=True)

    model.save(f"{save_dir}/fcu_final")
    env.save(f"{save_dir}/fcu_vecnorm.pkl")
    print("FCU 에이전트 학습 완료.")
    return model


def alternating_finetune(shared: SharedStateBuffer,
                         ahu_model: SAC,
                         fcu_model: SAC,
                         env_id: str = SINERGYM_ENV_ID,
                         n_rounds: int = 5,
                         steps_per_round: int = 50_000,
                         save_dir: str = "./models"):
    """3단계: AHU → FCU → ... 번갈아 파인튜닝."""
    print("=" * 55)
    print("3단계: 번갈아 파인튜닝 시작")
    print("=" * 55)

    for rnd in range(n_rounds):
        print(f"\n[Round {rnd+1}/{n_rounds}]")

        # AHU 파인튜닝
        ahu_env = DummyVecEnv([make_ahu_env(shared, env_id)])
        ahu_env = VecNormalize(ahu_env, norm_obs=True, norm_reward=True,
                               clip_obs=10.0)
        ahu_model.set_env(ahu_env)
        ahu_model.learn(
            total_timesteps     = steps_per_round,
            reset_num_timesteps = False,
            callback            = HVACTrainingCallback("AHU-ft"),
            progress_bar        = False,
        )
        ahu_model.save(f"{save_dir}/ahu_ft_round{rnd+1}")
        print(f"  AHU 파인튜닝 완료 (round {rnd+1})")

        # FCU 파인튜닝
        fcu_env = DummyVecEnv([make_fcu_env(shared, env_id)])
        fcu_env = VecNormalize(fcu_env, norm_obs=True, norm_reward=True,
                               clip_obs=10.0)
        fcu_model.set_env(fcu_env)
        fcu_model.learn(
            total_timesteps     = steps_per_round,
            reset_num_timesteps = False,
            callback            = HVACTrainingCallback("FCU-ft"),
            progress_bar        = False,
        )
        fcu_model.save(f"{save_dir}/fcu_ft_round{rnd+1}")
        print(f"  FCU 파인튜닝 완료 (round {rnd+1})")

    print("\n번갈아 파인튜닝 완료.")


# ─────────────────────────────────────────────────────────
# 8. 추론 (배포 시 사용)
# ─────────────────────────────────────────────────────────

class HVACController:
    """학습 완료된 두 에이전트를 이용한 실시간 제어."""

    def __init__(self, ahu_model_path: str, fcu_model_path: str,
                 ahu_vecnorm_path: str, fcu_vecnorm_path: str,
                 env_id: str = SINERGYM_ENV_ID):
        self.shared = SharedStateBuffer()

        ahu_env_raw = DummyVecEnv([make_ahu_env(self.shared, env_id)])
        fcu_env_raw = DummyVecEnv([make_fcu_env(self.shared, env_id)])

        self.ahu_env = VecNormalize.load(ahu_vecnorm_path, ahu_env_raw)
        self.fcu_env = VecNormalize.load(fcu_vecnorm_path, fcu_env_raw)
        self.ahu_env.training = False
        self.fcu_env.training = False

        self.ahu_model = SAC.load(ahu_model_path, env=self.ahu_env)
        self.fcu_model = SAC.load(fcu_model_path, env=self.fcu_env)

    def apply_llm_directive(self, directive: dict):
        self.shared.update_from_llm(directive)

    def control_step(self, sinergym_obs: np.ndarray) -> dict:
        """
        sinergym_obs: Sinergym 환경의 step/reset 반환 관측값
        Returns: {"sat_delta", "damper_delta", "valve_deltas"}
        """
        # AHU Wrapper obs 추출 및 정규화
        ahu_wrapper = self.ahu_env.envs[0].unwrapped
        raw_ahu_obs = ahu_wrapper._extract_obs(sinergym_obs)[np.newaxis]
        norm_ahu_obs = self.ahu_env.normalize_obs(raw_ahu_obs)
        ahu_act, _   = self.ahu_model.predict(norm_ahu_obs, deterministic=True)

        # FCU Wrapper obs 추출 및 정규화
        fcu_wrapper = self.fcu_env.envs[0].unwrapped
        raw_fcu_obs = fcu_wrapper._extract_obs(sinergym_obs)[np.newaxis]
        norm_fcu_obs = self.fcu_env.normalize_obs(raw_fcu_obs)
        fcu_act, _   = self.fcu_model.predict(norm_fcu_obs, deterministic=True)

        return {
            "sat_delta":    float(ahu_act[0][0]),
            "damper_delta": float(ahu_act[0][1]),
            "valve_deltas": fcu_act[0].tolist(),
        }


# ─────────────────────────────────────────────────────────
# 9. 메인 실행
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    SAVE_DIR        = "./models"
    ENV_ID          = SINERGYM_ENV_ID
    AHU_TIMESTEPS   = 200_000
    FCU_TIMESTEPS   = 200_000
    FINETUNE_ROUNDS = 5
    FINETUNE_STEPS  = 50_000

    # OBS_IDX가 실제 환경과 일치하는지 사전 확인
    probe = gym.make(ENV_ID)
    obs, info = probe.reset()
    print(f"Sinergym obs dim : {obs.shape[0]}")
    print(f"Sinergym act dim : {probe.action_space.shape[0]}")
    print("obs 샘플:", obs)
    probe.close()

    shared = SharedStateBuffer(n_rooms=N_ROOMS)

    ahu_model = train_ahu_agent(
        shared, env_id=ENV_ID,
        total_timesteps=AHU_TIMESTEPS, save_dir=SAVE_DIR
    )

    fcu_model = train_fcu_agent(
        shared, ahu_model, env_id=ENV_ID,
        total_timesteps=FCU_TIMESTEPS, save_dir=SAVE_DIR
    )

    alternating_finetune(
        shared, ahu_model, fcu_model, env_id=ENV_ID,
        n_rounds=FINETUNE_ROUNDS, steps_per_round=FINETUNE_STEPS,
        save_dir=SAVE_DIR
    )

    print("\n전체 학습 파이프라인 완료.")
    print(f"모델 저장 위치: {SAVE_DIR}/")