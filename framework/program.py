"""
운동피질(motor cortex) v0 — 행동 프로그램 시퀀서.

복합 운동 = 스킬들의 "의도적인" 배치 (DESIGN.md: 무작위 혼합 금지).
프로그램 = 스텝 목록. 각 스텝: 어떤 스킬을, 어떤 조건까지, 최소/최대 얼마 동안.
until 조건은 지각(percept) 술어 — 나중에 LLM이 JSON DSL로 작성할 수 있도록
호출 가능 객체로 통일해 둔다.
"""
from dataclasses import dataclass, field
from typing import Callable, List, Optional


@dataclass
class Step:
    skill: str
    until: Optional[Callable] = None   # until(percept) -> bool. None이면 timeout만
    timeout_s: float = 1e9
    min_s: float = 0.0


@dataclass
class Program:
    name: str
    steps: List[Step]
    loop: bool = False


class Sequencer:
    """현재 프로그램을 실행: 스텝 전이 판단 후 지금 실행할 스킬을 반환."""

    def __init__(self):
        self.program: Optional[Program] = None
        self.idx = 0
        self._t_enter = 0.0

    def set_program(self, program: Program, t: float):
        self.program = program
        self.idx = 0
        self._t_enter = t

    def current_skill(self, percept) -> Optional[str]:
        if self.program is None or self.idx >= len(self.program.steps):
            return None
        t = percept["t"]
        step = self.program.steps[self.idx]
        elapsed = t - self._t_enter
        done = False
        if elapsed >= step.min_s:
            if step.until is not None and step.until(percept):
                done = True
            elif elapsed >= step.timeout_s:
                done = True
        if done:
            self.idx += 1
            self._t_enter = t
            if self.idx >= len(self.program.steps):
                if self.program.loop:
                    self.idx = 0
                else:
                    return None
            step = self.program.steps[self.idx]
        return step.skill

    @property
    def step_label(self):
        if self.program is None:
            return "-"
        if self.idx >= len(self.program.steps):
            return f"{self.program.name}:done"
        return f"{self.program.name}[{self.idx}]:{self.program.steps[self.idx].skill}"
