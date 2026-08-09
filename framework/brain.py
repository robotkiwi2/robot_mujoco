"""
두뇌 통합(brain) v0 — 감각피질 → 연합령 → 운동피질(프로그램) → 소뇌 → 근육.

호르몬/내수용감각은 env 내부에서 물리·고통을 변조하고(interoception/hormones/affect),
이 파일의 루프에서는 행동 선택(놀람/휴식 인터럽트)까지 변조한다.
"""
from framework.association import AssociationCortex
from framework.cerebellum import Cerebellum
from framework.sensory import SensoryCortex


class Brain:
    def __init__(self, env, env_factory):
        self.env = env
        self.sensory = SensoryCortex(env)
        self.cerebellum = Cerebellum(env_factory)
        self.association = AssociationCortex(self.cerebellum.available())
        self.current_skill = None

    def step(self, obs, last_reward=0.0):
        percept = self.sensory.perceive(obs)
        skill = self.association.select(percept)
        if self.current_skill is not None:
            self.cerebellum.record_reward(self.current_skill, last_reward)
        self.current_skill = skill
        action = self.cerebellum.act(skill, obs)
        return action, {
            "program": self.association.active_name,
            "step": self.association.seq.step_label,
            "skill": skill,
            "percept": percept,
        }
