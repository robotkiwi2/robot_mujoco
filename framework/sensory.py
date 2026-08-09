"""
감각피질(sensory cortex) v0 — 원시 신호를 구조화된 지각(percept)으로.

정책(소뇌)은 평평한 관측 벡터를 그대로 쓰지만, 상위 층(연합령/프로그램/LLM 숙고)은
이름 붙은 지각을 쓴다. 여기서 그 변환을 담당한다.
"""
import mujoco
import numpy as np


class SensoryCortex:
    def __init__(self, env):
        self.env = env
        m = env.model
        self._front_id = env.front_id

    def perceive(self, obs):
        """env의 현재 상태에서 구조화된 지각을 만든다. obs는 정책용 평평한 벡터(그대로 통과)."""
        env, d = self.env, self.env.data
        upright = float(d.xmat[self._front_id].reshape(3, 3)[2, 2])
        p = {
            "t": float(d.time),
            "obs": obs,                      # 소뇌(정책)용 원본 벡터
            "upright": upright,
            "z": float(d.qpos[2]),
            "speed": float(np.linalg.norm(d.qvel[0:2])),
            "yaw_rate": float(d.sensordata[env._gyro_adr + 2]),
        }
        if env.energy:
            p.update({
                "soc": env.energy_state.soc,
                "power_W": env.energy_state.power_W,
                "impact": env.impact_state.impact_norm,
                "damage": env.impact_state.damage,
                "adrenaline": env.hormones.adrenaline,
                "cortisol": env.hormones.cortisol,
                "energy_pain": env.energy_affect.state_pain(env.energy_state.soc)
                               + env.energy_affect.consumption_pain(env.energy_state.power_W),
                "damage_pain": env.impact_affect.damage_pain(env.impact_state.damage),
            })
        # 후각 (nursery 등 냄새 필드가 있는 월드): 채널0 = 충전소/먹이A
        if getattr(env, "scent_field", None) is not None:
            left, right = float(env.scent_nose[0][0]), float(env.scent_nose[1][0])
            p.update({
                "scent_left": left,
                "scent_right": right,
                "scent": 0.5 * (left + right),
                "scent_dg": left - right,   # 양수 = 소스가 왼쪽 → 좌회전해야 함
                "on_charger": env.on_charger,
            })
        return p
