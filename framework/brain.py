"""
두뇌 통합(brain) v0 — 감각피질 → 연합령 → 운동피질(프로그램) → 소뇌 → 근육.

호르몬/내수용감각은 env 내부에서 물리·고통을 변조하고(interoception/hormones/affect),
이 파일의 루프에서는 행동 선택(놀람/휴식 인터럽트)까지 변조한다.
"""
import numpy as np

from framework.association import AssociationCortex
from framework.cerebellum import Cerebellum
from framework.poses import POSES, apply_pose, is_pose, pose_name
from framework.sensory import SensoryCortex
from framework.skill_registry import SKILLS as REGISTRY


class Brain:
    def __init__(self, env, env_factory):
        self.env = env
        self.sensory = SensoryCortex(env)
        self.cerebellum = Cerebellum(env_factory)
        # 레퍼토리 = 학습된 스킬(소뇌) + L0 포즈(반사층, 무학습)
        available = self.cerebellum.available() + [f"pose:{n}" for n in POSES]
        self.association = AssociationCortex(available)
        self.current_skill = None
        self.manual_skill = None   # 조작패널의 스킬 직접 실행 (None=프로그램이 결정)

    def reset(self):
        """에피소드 리셋(기절/넘어짐) 시 호출 필수 — 시뮬레이션 시간이 0으로 돌아가므로
        시퀀서 시계도 함께 리셋해야 한다 (안 하면 t_enter가 미래가 되어 스텝에 갇힘)."""
        self.association.active_name = None
        self.current_skill = None
        key = self.env.model.key_ctrl[0]
        self.env.home_ctrl[:] = key
        self.env.leg_home = key[self.env.leg_act_ids].copy()

    def _sync_env_mode(self, skill, obs):
        """스킬이 요구하는 env 모드로 전환 (예: toe_curl=12차원·눕기 <-> 이동=16차원·서기).
        모드가 바뀌면 env를 리셋하되 내부상태(SoC/손상/호르몬)는 이어붙인다 — 몸의 자세가
        바뀌는 것이지 정신을 잃는 게 아니므로. (LESSONS #12의 brain 경로 버전)"""
        if is_pose(skill):
            required = "walk" if self.env.mode == "toe_curl" else self.env.mode
            kw = {}
        else:
            kw = REGISTRY[skill]["env_kwargs"]
            required = kw["mode"]
        if required == self.env.mode:
            return obs
        env = self.env
        soc, dmg = env.energy_state.soc, env.impact_state.damage
        adren, cort = env.hormones.adrenaline, env.hormones.cortisol
        env.mode = required
        env.target_speed = kw.get("target_speed", env.target_speed)
        env.target_yaw_rate = kw.get("target_yaw_rate", env.target_yaw_rate)
        env.toe_curl_freq = kw.get("toe_curl_freq", env.toe_curl_freq)
        obs, _ = env.reset()
        env.energy_state.reset(soc)
        env.energy_affect.reset(soc)
        env.impact_state.damage = dmg
        env.hormones.adrenaline, env.hormones.cortisol = adren, cort
        self.association.active_name = None if self.manual_skill is None else "manual"
        return obs

    def step(self, obs, last_reward=0.0):
        percept = self.sensory.perceive(obs)
        if self.manual_skill is not None:
            skill = self.manual_skill
            self.association.active_name = "manual"
        else:
            skill = self.association.select(percept)
        obs = self._sync_env_mode(skill, obs)
        if self.current_skill is not None and not is_pose(self.current_skill):
            self.cerebellum.record_reward(self.current_skill, last_reward)
        self.current_skill = skill

        if is_pose(skill):
            # L0 반사: env.step은 다리 목표를 leg_home + action*scale로 계산하므로
            # 포즈의 다리각을 leg_home에, 허리/목은 home_ctrl에 주입하고 action=0을 준다.
            p = POSES[pose_name(skill)]
            self.env.leg_home = np.clip(np.asarray(p["legs"]), self.env.leg_lo, self.env.leg_hi)
            self.env.home_ctrl[0:3] = p["spine"]
            self.env.home_ctrl[3:5] = p["neck"]
            action = np.zeros(self.env.action_space.shape[0], dtype=np.float32)
        else:
            # 학습 스킬로 복귀 시 home 원복 (포즈가 바꿔둔 기준 자세 복구)
            key = self.env.model.key_ctrl[0]
            self.env.home_ctrl[:] = key
            self.env.leg_home = key[self.env.leg_act_ids].copy()
            action = self.cerebellum.act(skill, obs)

        return action, {
            "program": self.association.active_name,
            "step": self.association.seq.step_label,
            "skill": skill,
            "percept": percept,
        }
