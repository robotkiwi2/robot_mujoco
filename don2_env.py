"""
don2 보행 스킬 학습용 Gymnasium 환경 (framework 이식 전 레거시 스타일).

DESIGN.md 원칙:
- 요소행동(걷기)은 욕구와 무관하게 인위적 보상으로 독립 학습한다.
- 에너지는 고통이다(energy=True): 전력 소비(흐름)는 스텝당 직접 비용,
  저에너지(SoC<setpoint, 상태)는 전위 차분으로 보상에 반영. 둘 다 관측에 포함(내수용감각).
- 관측: 전체 센서 149 + 몸통높이 1 (+ energy 시 [SoC, 전력, 고통] 3) = 150 또는 153
- 행동: 다리 서보 16개의 목표각 오프셋 [-1,1] (home 기준, 위치서보 추종)
"""
import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from framework.affect import EnergyAffect, ImpactAffect
from framework.fields import ScentField, ScentSource
from framework.hormones import Hormones
from framework.interoception import EnergyState, ImpactState
from robots.don2 import robot_config as rc

ROBOT_XML = "robots/don2/don2.xml"  # 프로젝트 루트 기준 상대경로 (한글경로 이슈 회피)

LEGS = ["FL", "FR", "RL", "RR"]
LEG_ACT_NAMES = [f"{leg}_{part}_act" for leg in LEGS for part in ["abd", "hip", "knee", "ankle"]]
TOE_ACT_NAMES = [f"{leg}_toe_{t}_act" for leg in LEGS for t in ["f1", "f2", "b"]]
TOE_JOINT_NAMES = [f"{leg}_toe_{t}_j" for leg in LEGS for t in ["f1", "f2", "b"]]
# home 기준 행동 오프셋 스케일 [rad] (abd, hip, knee, ankle)
ACTION_SCALE = np.tile(np.array([0.3, 0.6, 0.6, 0.5]), 4)
LOCOMOTION_MODES = ("stand", "step_in_place", "sprint", "walk", "turn_left", "turn_right")
# 180도 뒤집힌(등을 바닥에 댄) 자세의 쿼터니언 (x축 기준 roll 180도: w,x,y,z)
SUPINE_QUAT = np.array([0.0, 1.0, 0.0, 0.0])


CHARGER_POS = (2.5, 0.0)
CHARGER_RADIUS = 0.35       # 이 반경 안에 있으면 충전
CHARGER_RATE = 0.05         # SoC/초 (55.5Wh 기준 약 10kW급 급속충전 — 데모 스케일)


def load_model_with_floor(world: str = "flat"):
    """world: "flat"(바닥만) | "nursery"(바닥 + 충전소 패드).
    임시 구현 — worlds/ 레이어의 compose(MjSpec)로 이관 예정."""
    with open(ROBOT_XML, encoding="utf-8") as f:
        xml = f.read()
    extra = ('<geom name="floor" type="plane" size="0 0 0.05" friction="1.0 0.005 0.0001"/>'
             '<light pos="0 0 2" dir="0 0 -1" directional="true"/>')
    if world == "nursery":
        # 충전소: 초록 발광 패드 (감각 시그니처 규약 — 빛 + 냄새 채널0)
        extra += (f'<geom name="charger_pad" type="cylinder" size="{CHARGER_RADIUS} 0.006" '
                  f'pos="{CHARGER_POS[0]} {CHARGER_POS[1]} 0.006" rgba="0.15 0.95 0.35 0.85" '
                  f'contype="0" conaffinity="0"/>'
                  f'<light name="charger_glow" pos="{CHARGER_POS[0]} {CHARGER_POS[1]} 0.6" '
                  f'dir="0 0 -1" diffuse="0.15 0.9 0.3"/>')
    return mujoco.MjModel.from_xml_string(xml.replace("<worldbody>", "<worldbody>" + extra, 1))


class Don2Env(gym.Env):
    metadata = {"render_modes": [], "render_fps": 100}

    def __init__(self, frame_skip: int = 5, max_episode_steps: int = 500,
                 mode: str = "sprint", target_speed: float = 0.35,
                 target_yaw_rate: float = 0.6, toe_curl_freq: float = 0.4, energy: bool = True,
                 world: str = "flat"):
        """mode: "stand"(기립 정지 균형 — 발달 계보의 뿌리 스킬) /
                 "sprint"(속도 최대화) / "walk"(target_speed 추종) /
                 "turn_left"/"turn_right"(target_yaw_rate 추종, gyro z+ = 좌회전, 물리 검증됨.
                 turn_right는 target_yaw_rate에 음수를 넣는다) /
                 "toe_curl"(등을 대고 누운 채 발가락 12개를 주기적으로 오므렸다 폈다;
                 다리/허리/목은 home 고정, 행동공간이 12차원으로 달라짐).
        energy: 에너지 내수용감각+고통 활성화 (관측 +3, 보상에 에너지 항 추가).
                False는 에너지 도입 전 체크포인트(don2__flat__forward 초기) 재생용."""
        super().__init__()
        assert mode in LOCOMOTION_MODES + ("toe_curl",)
        self.mode = mode
        self.target_speed = target_speed
        self.target_yaw_rate = target_yaw_rate
        self.toe_curl_freq = toe_curl_freq
        self.energy = energy
        self.world = world
        self.model = load_model_with_floor(world)
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps

        self.front_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_front")
        self.leg_act_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in LEG_ACT_NAMES]
        )
        self.home_ctrl = self.model.key_ctrl[0].copy()          # 허리/목 0, 다리 home, 발가락 0
        self.leg_home = self.home_ctrl[self.leg_act_ids].copy()
        self.leg_lo = self.model.actuator_ctrlrange[self.leg_act_ids, 0]
        self.leg_hi = self.model.actuator_ctrlrange[self.leg_act_ids, 1]
        gyro_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
        self._gyro_adr = self.model.sensor_adr[gyro_id]
        # 발별 접촉(발바닥+발가락3) 센서 주소 — step_in_place의 대각 교대 보상용
        self._foot_touch_adr = {}
        for leg in LEGS:
            ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, n)
                   for n in (f"{leg}_sole_t", f"{leg}_toe_f1_t", f"{leg}_toe_f2_t", f"{leg}_toe_b_t")]
            self._foot_touch_adr[leg] = [self.model.sensor_adr[i] for i in ids]

        self.toe_act_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in TOE_ACT_NAMES]
        )
        self.toe_hi = self.model.actuator_ctrlrange[self.toe_act_ids, 1]  # lo는 전부 0(당김 전용)
        toe_joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in TOE_JOINT_NAMES]
        self.toe_qpos_adr = self.model.jnt_qposadr[toe_joint_ids]
        self.toe_range_hi = self.model.jnt_range[toe_joint_ids, 1]

        if self.energy:
            self.energy_state = EnergyState(
                self.model, rc.ACTUATOR_MODEL, rc.ACTUATOR_SPECS,
                battery_capacity_Wh=rc.BATTERY["capacity_Wh"],
                base_power_W=rc.COMPUTE_POWER_W + rc.SENSOR_POWER_W,
            )
            self.energy_affect = EnergyAffect()
            # 충격/손상 + 호르몬 (아드레날린/코르티솔) — DESIGN.md 운동제어/정서 층
            self.impact_state = ImpactState(self.model)
            self.impact_affect = ImpactAffect()
            self.hormones = Hormones()
            self._leg_forcerange_base = self.model.actuator_forcerange[self.leg_act_ids].copy()

        # 월드 필드/오브젝트 — 지각(percept) 전용. 스킬 정책 관측에는 넣지 않는다
        # (스킬은 냄새를 몰라도 되고, 방향 판단은 프로그램/연합령의 몫 → 재학습 불필요).
        self.scent_field = None
        self.scent_nose = np.zeros((2, ScentField.N_CHANNELS))  # [왼쪽, 오른쪽]
        self.on_charger = False
        self._nostril_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, n)
            for n in ("nostril_l", "nostril_r")
        ]
        if world == "nursery":
            self.scent_field = ScentField(
                [ScentSource(CHARGER_POS, strength=1.0, decay_length=1.5, channel=0)])

        # 관측: 센서 + 높이 + (에너지3 + 충격/손상2 + 호르몬2 = 7)
        n_obs = self.model.nsensordata + 1 + (7 if self.energy else 0)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(n_obs,), dtype=np.float32)
        n_act = len(TOE_ACT_NAMES) if mode == "toe_curl" else len(LEG_ACT_NAMES)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n_act,), dtype=np.float32)

        self._steps = 0
        self._prev_x = 0.0
        self._prev_action = np.zeros(n_act, dtype=np.float32)

    def _get_obs(self):
        base = [self.data.sensordata, [self.data.qpos[2]]]
        if self.energy:
            es, af = self.energy_state, self.energy_affect
            ims, imf, hor = self.impact_state, self.impact_affect, self.hormones
            pain = (af.state_pain(es.soc) + af.consumption_pain(es.power_W)
                    + imf.damage_pain(ims.damage) * hor.acute_pain_gain())
            base.append([es.soc, es.power_W / af.power_ref_W, pain,
                         ims.impact_norm, ims.damage,
                         hor.adrenaline, hor.cortisol])
        return np.concatenate(base).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.data.ctrl[:] = self.home_ctrl

        if self.mode == "toe_curl":
            # 등을 바닥에 대고 눕는 자세: freejoint 쿼터니언을 180도(roll) 뒤집는다.
            # 다리/허리/목은 home 위치 서보로 고정 유지되므로 별도 포즈 튜닝 불필요(자세 유지는 위치제어가 담당).
            self.data.qpos[3:7] = SUPINE_QUAT
            self.data.qpos[2] = 0.08
        else:
            # 다리 관절 초기각에 소량 노이즈 (freejoint 7 + 허리3 + 목2 이후가 다리)
            self.data.qpos[12:] += self.np_random.uniform(-0.04, 0.04, size=self.model.nq - 12)

        mujoco.mj_forward(self.model, self.data)
        self._steps = 0
        self._prev_x = float(self.data.qpos[0])
        # self.action_space는 생성 시점의 mode로 고정되어 이후 mode를 동적으로 바꿔도
        # 갱신되지 않으므로(예: 인터랙티브 뷰어의 스킬 전환), 여기서 현재 mode 기준으로 직접 계산한다.
        n_act = len(TOE_ACT_NAMES) if self.mode == "toe_curl" else len(LEG_ACT_NAMES)
        self._prev_action = np.zeros(n_act, dtype=np.float32)
        if self.energy:
            # 초기 SoC 랜덤화: 저에너지 상태의 고통도 경험하도록 (domain randomization)
            soc0 = float(self.np_random.uniform(0.25, 1.0))
            self.energy_state.reset(soc0)
            self.energy_affect.reset(soc0)
            self.impact_state.reset()
            self.impact_affect.reset()
            self.hormones.reset()
            self.model.actuator_forcerange[self.leg_act_ids] = self._leg_forcerange_base
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        self.data.ctrl[:] = self.home_ctrl
        if self.mode == "toe_curl":
            # [-1,1] -> [0, toe_hi] 장력(N). 텐던은 당김 전용이라 0 미만은 없음.
            self.data.ctrl[self.toe_act_ids] = np.clip((action + 1.0) * 0.5 * self.toe_hi, 0.0, self.toe_hi)
        else:
            targets = np.clip(self.leg_home + action * ACTION_SCALE, self.leg_lo, self.leg_hi)
            self.data.ctrl[self.leg_act_ids] = targets

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            if self.energy:
                self.impact_state.substep_sample(self.data)  # 짧은 충격 스파이크 피크 추적

        x = float(self.data.qpos[0])
        dt = self.model.opt.timestep * self.frame_skip
        forward_vel = (x - self._prev_x) / dt
        self._prev_x = x

        upright = float(self.data.xmat[self.front_id].reshape(3, 3)[2, 2])
        z = float(self.data.qpos[2])
        tilt_penalty = 0.5 * max(0.0, 0.7 - upright)

        if self.mode == "step_in_place":
            # L1 자세 프리미티브: 제자리에서 대각 발쌍(FL+RR ↔ FR+RL)을 교대로 들기.
            # 걷기의 구성요소(리듬+균형)를 이동 없이 학습 → walk의 워름스타트 부모 후보.
            contact = {leg: sum(float(self.data.sensordata[a]) for a in adrs) > 0.5
                       for leg, adrs in self._foot_touch_adr.items()}
            diag_a = contact["FL"] and contact["RR"] and not contact["FR"] and not contact["RL"]
            diag_b = contact["FR"] and contact["RL"] and not contact["FL"] and not contact["RR"]
            gait = 1.0 if (diag_a or diag_b) else 0.0
            drift = float(np.linalg.norm(self.data.qpos[0:2]))
            ctrl_cost = 0.02 * float(np.sum(np.square(action)))
            jerk_cost = 0.02 * float(np.sum(np.square(action - self._prev_action)))
            reward = (1.5 * gait + 0.4 * (1.0 - np.tanh(2.0 * drift))
                      - ctrl_cost - jerk_cost - tilt_penalty + 0.4)
        elif self.mode == "stand":  # 제자리 기립: 이동/회전/동작을 모두 억제 (계보의 뿌리)
            gyro = self.data.sensordata[self._gyro_adr:self._gyro_adr + 3]
            still = 1.0 - np.tanh(4.0 * abs(forward_vel) + 1.5 * float(np.linalg.norm(gyro)))
            ctrl_cost = 0.03 * float(np.sum(np.square(action)))
            jerk_cost = 0.03 * float(np.sum(np.square(action - self._prev_action)))
            reward = 1.2 * still - ctrl_cost - jerk_cost - tilt_penalty + 0.5
        elif self.mode == "sprint":
            ctrl_cost = 0.01 * float(np.sum(np.square(action)))
            reward = 2.0 * forward_vel - ctrl_cost - tilt_penalty + 0.5
        elif self.mode == "walk":  # 목표 속도 추종 + 동작 크기·급격함 페널티 강화(저속·저에너지 보행)
            speed_error = abs(forward_vel - self.target_speed)
            ctrl_cost = 0.03 * float(np.sum(np.square(action)))
            jerk_cost = 0.02 * float(np.sum(np.square(action - self._prev_action)))
            reward = 1.5 * (1.0 - np.tanh(2.0 * speed_error)) - ctrl_cost - jerk_cost - tilt_penalty + 0.5
        elif self.mode in ("turn_left", "turn_right"):  # 목표 회전율 추종(gyro z, 우회전은 음수 목표)
            yaw_rate = float(self.data.sensordata[self._gyro_adr + 2])
            yaw_error = abs(yaw_rate - self.target_yaw_rate)
            ctrl_cost = 0.02 * float(np.sum(np.square(action)))
            jerk_cost = 0.02 * float(np.sum(np.square(action - self._prev_action)))
            reward = (1.5 * (1.0 - np.tanh(2.0 * yaw_error)) + 0.3 * max(0.0, forward_vel)
                      - ctrl_cost - jerk_cost - tilt_penalty + 0.5)
        else:  # toe_curl: 등을 대고 누운 채 발가락 12개를 사인파 위상으로 오므렸다 폈다
            onback = -upright  # 서 있을 땐 upright≈+1, 등을 대고 누우면 upright≈-1 → onback≈+1
            onback_penalty = 0.5 * max(0.0, 0.7 - onback)
            phase = 2.0 * np.pi * self.toe_curl_freq * self.data.time
            target = 0.5 * (1.0 + np.sin(phase)) * self.toe_range_hi
            toe_angle = self.data.qpos[self.toe_qpos_adr]
            curl_error = float(np.mean(np.abs(toe_angle - target)))
            ctrl_cost = 0.02 * float(np.sum(np.square(action)))
            jerk_cost = 0.02 * float(np.sum(np.square(action - self._prev_action)))
            reward = 1.5 * (1.0 - np.tanh(3.0 * curl_error)) - ctrl_cost - jerk_cost - onback_penalty + 0.5
        self._prev_action = action.copy()

        info = {"forward_vel": forward_vel, "upright": upright, "z": z,
                "yaw_rate": float(self.data.sensordata[self._gyro_adr + 2])}
        depleted = False
        if self.energy:
            # 1) 충격 확정 → 2) 호르몬 갱신(분비/감쇠) → 3) 호르몬 배율로 고통/에너지 계산
            impact_norm, damage = self.impact_state.step(dt)
            prelim_pain = self.impact_affect.w_impact * min(2.0, impact_norm)
            self.hormones.step(dt, impact_norm, prelim_pain)

            power_W, soc = self.energy_state.step(self.data, dt,
                                                  consumption_gain=self.hormones.energy_gain())

            # 충전소/냄새 (nursery 월드): 코 샘플링 + 패드 위 충전.
            # 충전으로 soc가 오르면 저SoC 고통의 전위 차분이 양수 = "고통의 해소가 곧 행복" (DESIGN)
            if self.scent_field is not None:
                for i, sid in enumerate(self._nostril_ids):
                    self.scent_nose[i] = self.scent_field.sample(self.data.site_xpos[sid])
                dist = float(np.linalg.norm(self.data.qpos[0:2] - np.array(CHARGER_POS)))
                self.on_charger = dist < CHARGER_RADIUS
                if self.on_charger:
                    self.energy_state.soc = min(1.0, self.energy_state.soc + CHARGER_RATE * dt)
                    soc = self.energy_state.soc

            energy_reward, e_pain = self.energy_affect.reward_terms(soc, power_W)
            impact_reward, i_pain = self.impact_affect.reward_terms(
                impact_norm, damage, acute_gain=self.hormones.acute_pain_gain())
            reward += energy_reward + impact_reward

            # 아드레날린 토크 부스트: 다리 서보 forcerange를 순간적으로 확대 (물리 변조)
            self.model.actuator_forcerange[self.leg_act_ids] = \
                self._leg_forcerange_base * self.hormones.torque_gain()

            depleted = soc <= 0.0  # 완전 방전 = "기절"
            info.update({"power_W": power_W, "soc": soc, "pain": e_pain + i_pain,
                         "impact": impact_norm, "damage": damage,
                         "adrenaline": self.hormones.adrenaline,
                         "cortisol": self.hormones.cortisol,
                         "on_charger": self.on_charger})

        if self.mode == "toe_curl":
            fell = onback < 0.3  # 등이 바닥에서 크게 벗어남(옆으로 굴러감) = 실패
        else:
            fell = (z < 0.14) or (upright < 0.4)
        if fell or depleted:
            reward -= 5.0
        self._steps += 1

        obs = self._get_obs()
        return obs, reward, bool(fell or depleted), self._steps >= self.max_episode_steps, info
