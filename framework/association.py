"""
연합령(association cortex) v0 — 욕구 평가와 프로그램 선택.

지각(percept)에서 욕구 신호(고통/호르몬/에너지)를 평가해 어떤 행동 프로그램을
돌릴지 결정한다. 규칙 기반 v0 (DESIGN.md: 매니저는 규칙 기반 → RL/LLM 숙고로 발전).

v0 규칙 (우선순위 순):
1. 놀람 반사: 아드레날린 급등(>0.5) → freeze (가라앉을 때까지, 히스테리시스 0.15)
2. 에너지 보존: SoC 낮음(<0.27) → rest (충전 수단이 생기기 전까지는 가만히 아끼는 게 최선)
3. 기본: patrol (순찰 — 서기/걷기 반복)

LLM 숙고 훅: 새 목표가 주어지거나 프로그램이 반복 실패하면 이 층에서
프로그램 재작성을 요청하게 된다 (v0에서는 고정 프로그램 라이브러리 사용).
"""
from framework.program import Program, Sequencer, Step


def build_program_library(available_skills):
    lib = {}
    has_walk = "walk" in available_skills

    patrol_steps = [Step("stand", timeout_s=2.0, min_s=2.0)]
    if "pose:stretch" in available_skills:
        # L0 포즈(무학습 반사)를 프로그램에 섞는 예: 순찰 사이 기지개
        patrol_steps.append(Step("pose:stretch", timeout_s=1.5, min_s=1.5))
        patrol_steps.append(Step("stand", timeout_s=1.0, min_s=1.0))
    if has_walk:
        patrol_steps.append(Step("walk", timeout_s=5.0, min_s=5.0))
    lib["patrol"] = Program("patrol", patrol_steps, loop=True)

    # 놀람-정지: 아드레날린이 가라앉을 때까지 얼어붙기 (최소 1초)
    lib["startle_freeze"] = Program(
        "startle_freeze",
        [Step("stand", until=lambda p: p.get("adrenaline", 0.0) < 0.15, min_s=1.0)],
        loop=False,
    )

    # 휴식: 에너지 아끼기 (냄새/충전소가 없는 월드의 최선)
    lib["rest"] = Program("rest", [Step("stand", timeout_s=10.0)], loop=True)

    # 충전소 찾아가기: 냄새 좌우차로 방향을 잡고 → 걸어가고 → 패드 위에서 충전.
    # 방향 규칙은 프로그램 작성자가 명시한 것 (의도적 배치 — 무작위 혼합 아님).
    has_turns = "turn_left" in available_skills and "turn_right" in available_skills
    steps = []
    if has_turns:
        steps.append(Step(
            skill=lambda p: "turn_left" if p.get("scent_dg", 0.0) > 0 else "turn_right",
            until=lambda p: abs(p.get("scent_dg", 0.0)) < 0.02 * max(p.get("scent", 1e-6), 1e-6) * 20
                            or p.get("scent", 0.0) > 0.5,
            timeout_s=4.0, min_s=0.3))
    if has_walk:
        # 패드 첫 접촉이 아니라 중심부(냄새 ≥0.9 ≈ 중심 16cm 이내)까지 걸어 들어간다
        # — 가장자리에 멈추면 서 있는 동안 이탈해 충전이 끊긴다 (실측 교훈)
        steps.append(Step("walk",
                          until=lambda p: p.get("scent", 0.0) > 0.9,
                          timeout_s=15.0))
    steps.append(Step("stand", until=lambda p: p.get("soc", 0.0) > 0.85, timeout_s=40.0))
    lib["seek_charger"] = Program("seek_charger", steps, loop=True)
    return lib


class AssociationCortex:
    def __init__(self, available_skills):
        self.library = build_program_library(available_skills)
        self.seq = Sequencer()
        self.active_name = None
        self.override = None   # 조작패널 등 외부에서 프로그램 강제 (None=자동/욕구 선택)

    def _wanted(self, percept):
        if self.override is not None:
            return self.override
        return self._wanted_auto(percept)

    def _wanted_auto(self, percept):
        adren = percept.get("adrenaline", 0.0)
        soc = percept.get("soc", 1.0)
        if self.active_name == "startle_freeze":
            if adren > 0.15:          # 히스테리시스: 진정될 때까지 유지
                return "startle_freeze"
        elif adren > 0.5:
            return "startle_freeze"
        # 저에너지: 냄새(충전소 단서)가 있으면 찾아가고, 없으면 아껴 쓰기.
        # seek 진입 0.45 (넘어져 깨어나도 미완의 충전 의도가 이어지도록 — 실측 교훈),
        # 일단 시작하면 충분히 충전(85%)될 때까지 유지 (중도 포기 방지)
        low = soc < 0.45 or (self.active_name == "rest" and soc < 0.5) \
              or (self.active_name == "seek_charger" and soc < 0.85)
        if low:
            if percept.get("scent", 0.0) > 0.01 and "seek_charger" in self.library \
                    and len(self.library["seek_charger"].steps) > 1:
                return "seek_charger"
            return "rest"
        return "patrol"

    def select(self, percept):
        """매 스텝 호출: 필요 시 프로그램 전환 후 현재 스킬 반환."""
        wanted = self._wanted(percept)
        if wanted != self.active_name:
            self.seq.set_program(self.library[wanted], percept["t"])
            self.active_name = wanted
        skill = self.seq.current_skill(percept)
        if skill is None:            # 프로그램 종료(비루프) → 재평가 위해 초기화
            self.active_name = None
            return self.select(percept)
        return skill
