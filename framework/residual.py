"""
대뇌 개입 채널(residual intervention) — DESIGN.md 운동제어 3층 구조.

동결된 기저 스킬(소뇌) 위에 잔차 정책(대뇌 보정)을 얹는다:
    최종 행동 = 기저정책(관측) + residual_scale * 잔차행동
잔차의 크기에 비례한 "주의 비용(attention cost)"을 보상에서 차감한다
→ 평소에는 잔차≈0(소뇌에 맡김), 정말 필요할 때만 개입하는 패턴이 비용 최소화에서 창발.

사용 예 (거친 지형 걷기를 walk 위에 잔차로 학습):
    base_policy, base_norm = ...  # 동결된 walk 로드
    env = ResidualEnv(Don2Env(mode="walk", ...), base_policy, base_norm,
                      base_obs_dim=153, residual_scale=0.3, attention_cost=0.05)
    PPO("MlpPolicy", env).learn(...)   # 잔차망만 학습됨
"""
import gymnasium as gym
import numpy as np


class ResidualEnv(gym.Wrapper):
    def __init__(self, env, base_policy, base_normalizer, base_obs_dim: int,
                 residual_scale: float = 0.3, attention_cost: float = 0.05):
        super().__init__(env)
        self.base_policy = base_policy          # 동결: predict만 사용, 학습 안 함
        self.base_normalizer = base_normalizer  # 기저 스킬의 VecNormalize (관측 스케일 일치용)
        self.base_obs_dim = base_obs_dim
        self.residual_scale = residual_scale
        self.attention_cost = attention_cost
        self._last_raw_obs = None
        # 잔차 행동공간은 기저와 동일 shape의 [-1,1]
        self.action_space = env.action_space

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_raw_obs = obs
        return obs, info

    def step(self, residual_action):
        raw = self._last_raw_obs[: self.base_obs_dim]
        norm = self.base_normalizer.normalize_obs(raw[None, :])
        base_action, _ = self.base_policy.predict(norm, deterministic=True)
        combined = np.clip(base_action[0] + self.residual_scale * residual_action, -1.0, 1.0)

        obs, reward, term, trunc, info = self.env.step(combined)
        self._last_raw_obs = obs

        # 주의 비용: 대뇌 개입(잔차)의 크기만큼 고통 — 개입은 공짜가 아니다
        attn = self.attention_cost * float(np.sum(np.square(residual_action)))
        reward -= attn
        info["attention_cost"] = attn
        info["residual_norm"] = float(np.linalg.norm(residual_action))
        return obs, reward, term, trunc, info
