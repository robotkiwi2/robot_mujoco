"""
소뇌(cerebellum) v0 — 악보(동결 스킬) 재생 + 성과 추적.

- 레지스트리에서 사용 가능한 스킬(체크포인트 존재)만 로드해 캐시.
- act(skill, obs): 해당 스킬의 정규화 통계로 관측을 스케일링해 행동을 재생 (순전파 1회 = 쌈).
- 성과 EMA: 스킬 실행 중 보상의 지수이동평균 — 가소성 게이트(개선 세션 판단)의 입력.
  v0에서는 기록만 하고, 개선 세션 자체는 plasticity.py에서 구현 예정.
"""
import glob
import os
import re

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from framework.skill_registry import SKILLS, model_dir


def _find_latest(mdir):
    final = os.path.join(mdir, "ppo_final.zip")
    if os.path.exists(final):
        return final, os.path.join(mdir, "vecnormalize_final.pkl")
    cands = glob.glob(os.path.join(mdir, "ppo_*_steps.zip"))
    if not cands:
        raise FileNotFoundError(mdir)
    steps = lambda p: int(re.search(r"ppo_(\d+)_steps", p).group(1))
    best = max(cands, key=steps)
    return best, os.path.join(mdir, f"ppo_vecnormalize_{steps(best)}_steps.pkl")


class Cerebellum:
    def __init__(self, env_factory, skills=None):
        """env_factory: 스킬별 VecNormalize 로드용 더미 env 생성 함수 (env_kwargs를 받음)."""
        self.policies, self.normalizers, self.paths = {}, {}, {}
        self.perf_ema = {}   # 스킬별 성과 EMA (가소성 입력)
        self._ema_alpha = 0.01
        for name in (skills or list(SKILLS)):
            cfg = SKILLS[name]
            try:
                mpath, vpath = _find_latest(model_dir(name))
            except FileNotFoundError:
                continue  # 아직 학습 전인 스킬은 레퍼토리에서 제외
            self.policies[name] = PPO.load(mpath)
            dummy = DummyVecEnv([lambda kw=cfg["env_kwargs"]: env_factory(**kw)])
            norm = VecNormalize.load(vpath, dummy)
            norm.training = False
            self.normalizers[name] = norm
            self.paths[name] = mpath
            self.perf_ema[name] = None

    def available(self):
        return list(self.policies)

    def act(self, skill, obs):
        raw = obs[: SKILLS[skill]["obs_dim"]]
        norm = self.normalizers[skill].normalize_obs(raw[None, :])
        action, _ = self.policies[skill].predict(norm, deterministic=True)
        return action[0]

    def record_reward(self, skill, reward):
        """실행 중 스킬의 성과 기록 — '요즘 이 스킬이 잘 먹히는가'의 근거."""
        prev = self.perf_ema.get(skill)
        self.perf_ema[skill] = reward if prev is None else \
            (1 - self._ema_alpha) * prev + self._ema_alpha * reward
