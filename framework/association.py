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
    if has_walk:
        patrol_steps.append(Step("walk", timeout_s=5.0, min_s=5.0))
    lib["patrol"] = Program("patrol", patrol_steps, loop=True)

    # 놀람-정지: 아드레날린이 가라앉을 때까지 얼어붙기 (최소 1초)
    lib["startle_freeze"] = Program(
        "startle_freeze",
        [Step("stand", until=lambda p: p.get("adrenaline", 0.0) < 0.15, min_s=1.0)],
        loop=False,
    )

    # 휴식: 에너지 아끼기 (월드에 충전소가 생기면 '충전소 찾기' 프로그램으로 대체 예정)
    lib["rest"] = Program("rest", [Step("stand", timeout_s=10.0)], loop=True)
    return lib


class AssociationCortex:
    def __init__(self, available_skills):
        self.library = build_program_library(available_skills)
        self.seq = Sequencer()
        self.active_name = None

    def _wanted(self, percept):
        adren = percept.get("adrenaline", 0.0)
        soc = percept.get("soc", 1.0)
        if self.active_name == "startle_freeze":
            if adren > 0.15:          # 히스테리시스: 진정될 때까지 유지
                return "startle_freeze"
        elif adren > 0.5:
            return "startle_freeze"
        if soc < 0.27 or (self.active_name == "rest" and soc < 0.30):
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
