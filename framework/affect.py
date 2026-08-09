"""
정서(affect) v0 — 에너지 고통과 그 보상화.

DESIGN.md 규칙:
- 저에너지 고통(상태 고통): SoC가 setpoint 아래로 내려간 편차의 제곱.
  보상화는 반드시 전위 차분 pain(t-1) - pain(t) 으로 (자해-해소 루프 방지).
- 소비 고통(흐름 고통): 전력[W]에 비례. 흐름은 전위로 만들면 정상상태 기울기가
  0이 되어 절약을 학습할 수 없으므로 스텝당 직접 비용으로 부과한다.
추후 확장: 충격 고통, 독성 고통, 호르몬에 의한 가중치 변조(hormones.py).
"""


class EnergyAffect:
    def __init__(self, power_ref_W: float = 30.0, w_power: float = 0.3,
                 w_soc: float = 2.0, soc_setpoint: float = 0.3):
        self.power_ref_W = power_ref_W    # "보통 걷기" 수준 전력 (정규화 기준)
        self.w_power = w_power            # 소비 고통 가중치
        self.w_soc = w_soc                # 저에너지 고통 가중치
        self.soc_setpoint = soc_setpoint  # 이 아래로 내려가면 아프기 시작
        self._prev_state_pain = 0.0

    def state_pain(self, soc: float) -> float:
        d = max(0.0, self.soc_setpoint - soc) / self.soc_setpoint
        return self.w_soc * d * d

    def consumption_pain(self, power_W: float) -> float:
        return self.w_power * power_W / self.power_ref_W

    def reset(self, soc: float):
        self._prev_state_pain = self.state_pain(soc)

    def reward_terms(self, soc: float, power_W: float):
        """(보상 기여, 현재 총 고통) 반환.
        보상 기여 = -소비고통(직접 비용) + (이전 상태고통 - 현재 상태고통)(전위 차분)"""
        sp = self.state_pain(soc)
        cp = self.consumption_pain(power_W)
        contrib = -cp + (self._prev_state_pain - sp)
        self._prev_state_pain = sp
        return contrib, sp + cp


class ImpactAffect:
    """충격/손상 고통 (에너지와 같은 이원 구조):
    - 충격 고통(흐름): 임계 초과 충격 크기에 비례, 스텝당 직접 비용.
      호르몬 배율(acute_pain_gain: 아드레날린 진통/코르티솔 민감화)이 곱해진다.
    - 손상 고통(상태): damage 제곱 — 전위 차분으로 보상화 (자해 루프 방지)."""

    def __init__(self, w_impact: float = 3.0, w_damage: float = 1.5):
        self.w_impact = w_impact
        self.w_damage = w_damage
        self._prev_damage_pain = 0.0

    def damage_pain(self, damage: float) -> float:
        return self.w_damage * damage * damage

    def reset(self, damage: float = 0.0):
        self._prev_damage_pain = self.damage_pain(damage)

    def reward_terms(self, impact_norm: float, damage: float, acute_gain: float = 1.0):
        ip = self.w_impact * min(2.0, impact_norm) * acute_gain
        dp = self.damage_pain(damage) * acute_gain
        contrib = -ip + (self._prev_damage_pain - dp)
        self._prev_damage_pain = dp
        return contrib, ip + dp
