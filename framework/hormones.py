"""
호르몬 층 v0 — 아드레날린 + 코르티솔 (DESIGN.md 호르몬 표의 첫 2종).

같은 자극을 "다르게 느끼고 다르게 반응하게" 만드는 조절자.
세 경로로 작동한다: ① 관측에 포함(정책이 조건부 행동) ② 고통 가중치 변조 ③ 물리 변조(토크).

- 아드레날린 A: 급성 충격에 즉시 분비, 수 초 반감기.
  효과: 토크 한계 상승(+30%·A), 에너지 소모 증가(+50%·A), 일시 진통(급성 고통 -40%·A).
- 코르티솔 C: 고통의 장기 누적으로 서서히 상승, 수 분 반감기.
  효과: 고통 민감화(+50%·C), 만성 에너지 효율 저하(+20%·C 소모).

개체 기질(성격): profile 파라미터로 분비량/반감기를 개체마다 다르게 줄 수 있다
(겁 많은 개체 = cortisol_gain 높음 등). 학습 시 랜덤화하면 성격 분화 실험 가능.
"""
import math


DEFAULT_PROFILE = dict(
    adrenaline_gain=0.8,          # 충격 1.0(=ref급)당 분비량
    adrenaline_halflife_s=3.0,
    cortisol_gain=0.004,          # 스텝당 고통 1.0이 지속될 때의 상승률
    cortisol_halflife_s=90.0,
)


class Hormones:
    def __init__(self, profile: dict = None):
        self.p = dict(DEFAULT_PROFILE)
        if profile:
            self.p.update(profile)
        self.adrenaline = 0.0  # [0,1]
        self.cortisol = 0.0    # [0,1]

    def reset(self):
        self.adrenaline = 0.0
        self.cortisol = 0.0

    def step(self, dt: float, impact_norm: float, pain: float):
        # 분비
        self.adrenaline = min(1.0, self.adrenaline + self.p["adrenaline_gain"] * impact_norm)
        self.cortisol = min(1.0, self.cortisol + self.p["cortisol_gain"] * pain * dt / 0.01)
        # 감쇠 (반감기)
        self.adrenaline *= math.exp(-math.log(2.0) * dt / self.p["adrenaline_halflife_s"])
        self.cortisol *= math.exp(-math.log(2.0) * dt / self.p["cortisol_halflife_s"])
        return self.adrenaline, self.cortisol

    # ---- 변조 인터페이스 ----
    def torque_gain(self) -> float:
        """아드레날린의 순간 출력 부스트 (actuator forcerange에 곱함)."""
        return 1.0 + 0.3 * self.adrenaline

    def energy_gain(self) -> float:
        """에너지 소모 배율: 아드레날린 급증 + 코르티솔 만성 비용."""
        return 1.0 + 0.5 * self.adrenaline + 0.2 * self.cortisol

    def acute_pain_gain(self) -> float:
        """급성(충격/손상) 고통 배율: 아드레날린 진통 - 코르티솔 민감화."""
        return max(0.2, 1.0 - 0.4 * self.adrenaline + 0.5 * self.cortisol)

    def obs(self):
        return [self.adrenaline, self.cortisol]
