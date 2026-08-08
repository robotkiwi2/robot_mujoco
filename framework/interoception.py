"""
내수용감각(interoception) v0 — 에너지 상태(배터리 SoC)와 전력 소비.

DESIGN.md의 "고통의 입력이 되는 내부상태" 중 1단계로 에너지만 구현.
전력은 로봇 robot_config의 실측 부품 모델로 계산한다:
  P_elec = Σ(|τ_i·ω_i| / η_i) + Σ(서보 idle) + 컴퓨터/센서 기저 전력
추후 확장: 손상(충격 누적), 독성, 피로.
"""
import mujoco
import numpy as np


class EnergyState:
    def __init__(self, model, actuator_model: dict, actuator_specs: dict,
                 battery_capacity_Wh: float, base_power_W: float):
        self.capacity_J = battery_capacity_Wh * 3600.0
        self.base_power_W = base_power_W  # 컴퓨터 + 센서 (액추에이터 외 소비)

        nu = model.nu
        self.eff = np.ones(nu)
        self.idle = np.zeros(nu)
        for i in range(nu):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            spec = actuator_specs[actuator_model[name]]
            self.eff[i] = spec["efficiency"]
            self.idle[i] = spec["idle_power_W"]
        self.idle_total = float(self.idle.sum())

        self.soc = 1.0
        self.power_W = self.base_power_W + self.idle_total

    def reset(self, soc: float = 1.0):
        self.soc = float(soc)
        self.power_W = self.base_power_W + self.idle_total
        return self.soc

    def step(self, data, dt: float):
        """현재 액추에이터 부하로 전력을 계산하고 SoC를 갱신한다."""
        mech = np.abs(data.actuator_force * data.actuator_velocity)  # 기계적 출력 [W]
        self.power_W = float(np.sum(mech / self.eff) + self.idle_total + self.base_power_W)
        self.soc = max(0.0, self.soc - self.power_W * dt / self.capacity_J)
        return self.power_W, self.soc
