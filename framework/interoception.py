"""
내수용감각(interoception) — 고통의 입력이 되는 내부상태들.

v1 구현: 에너지(SoC/전력) + 충격/손상(ImpactState).
전력은 로봇 robot_config의 실측 부품 모델로 계산한다:
  P_elec = Σ(|τ_i·ω_i| / η_i) + Σ(서보 idle) + 컴퓨터/센서 기저 전력
추후 확장: 독성, 피로.
"""
import math

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

    def step(self, data, dt: float, consumption_gain: float = 1.0):
        """현재 액추에이터 부하로 전력을 계산하고 SoC를 갱신한다.
        consumption_gain: 호르몬(아드레날린/코르티솔)에 의한 소모 배율."""
        mech = np.abs(data.actuator_force * data.actuator_velocity)  # 기계적 출력 [W]
        self.power_W = float(np.sum(mech / self.eff) + self.idle_total + self.base_power_W) * consumption_gain
        self.soc = max(0.0, self.soc - self.power_W * dt / self.capacity_J)
        return self.power_W, self.soc


class ImpactState:
    """충격/손상 감각.

    - 충격(impact): IMU 가속도의 물리 서브스텝별 피크가 임계값을 넘은 초과분 (정규화 0~).
      정상 보행의 착지(~15-35 m/s^2)는 임계값(기본 50) 아래 → 아프지 않다.
      추락/충돌의 스파이크만 "강하고 급격한 충격"으로 인식된다.
    - 손상(damage 0~1): 큰 충격이 누적되고 천천히 회복되는 상태값 (타박상 모사).
      저SoC처럼 상태 고통(전위 차분)의 입력이 된다.
    """

    def __init__(self, model, accel_sensor: str = "imu_acc",
                 threshold: float = 50.0, ref: float = 150.0,
                 damage_gain: float = 0.25, recovery_halflife_s: float = 30.0):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, accel_sensor)
        self._adr = model.sensor_adr[sid]
        self.threshold = threshold
        self.ref = ref              # 이 크기의 충격이면 impact_norm ≈ 1
        self.damage_gain = damage_gain
        self.recovery_halflife_s = recovery_halflife_s
        self._peak = 0.0
        self.impact_norm = 0.0
        self.damage = 0.0

    def reset(self):
        self._peak = 0.0
        self.impact_norm = 0.0
        self.damage = 0.0

    def substep_sample(self, data):
        """물리 서브스텝마다 호출 — 짧은 스파이크를 놓치지 않도록 피크를 추적."""
        acc = float(np.linalg.norm(data.sensordata[self._adr:self._adr + 3]))
        if acc > self._peak:
            self._peak = acc

    def step(self, dt: float):
        """env 스텝마다 호출: 이번 스텝의 충격 크기를 확정하고 손상을 갱신."""
        excess = max(0.0, self._peak - self.threshold)
        self.impact_norm = excess / (self.ref - self.threshold)
        self._peak = 0.0
        # 손상: 충격 초과분만큼 누적, 반감기 기반 회복
        self.damage = min(1.0, self.damage + self.damage_gain * self.impact_norm)
        decay = math.exp(-math.log(2.0) * dt / self.recovery_halflife_s)
        self.damage *= decay
        return self.impact_norm, self.damage
