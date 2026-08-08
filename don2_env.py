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

from framework.affect import EnergyAffect
from framework.interoception import EnergyState
from robots.don2 import robot_config as rc

ROBOT_XML = "robots/don2/don2.xml"  # 프로젝트 루트 기준 상대경로 (한글경로 이슈 회피)

LEGS = ["FL", "FR", "RL", "RR"]
LEG_ACT_NAMES = [f"{leg}_{part}_act" for leg in LEGS for part in ["abd", "hip", "knee", "ankle"]]
# home 기준 행동 오프셋 스케일 [rad] (abd, hip, knee, ankle)
ACTION_SCALE = np.tile(np.array([0.3, 0.6, 0.6, 0.5]), 4)


def load_model_with_floor():
    with open(ROBOT_XML, encoding="utf-8") as f:
        xml = f.read()
    floor = ('<geom name="floor" type="plane" size="0 0 0.05" friction="1.0 0.005 0.0001"/>'
             '<light pos="0 0 2" dir="0 0 -1" directional="true"/>')
    return mujoco.MjModel.from_xml_string(xml.replace("<worldbody>", "<worldbody>" + floor, 1))


class Don2Env(gym.Env):
    metadata = {"render_modes": [], "render_fps": 100}

    def __init__(self, frame_skip: int = 5, max_episode_steps: int = 500,
                 mode: str = "sprint", target_speed: float = 0.35, energy: bool = True):
        """mode: "sprint"(속도 최대화) 또는 "walk"(target_speed 추종).
        energy: 에너지 내수용감각+고통 활성화 (관측 +3, 보상에 에너지 항 추가).
                False는 에너지 도입 전 체크포인트(don2__flat__forward 초기) 재생용."""
        super().__init__()
        assert mode in ("sprint", "walk")
        self.mode = mode
        self.target_speed = target_speed
        self.energy = energy
        self.model = load_model_with_floor()
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

        if self.energy:
            self.energy_state = EnergyState(
                self.model, rc.ACTUATOR_MODEL, rc.ACTUATOR_SPECS,
                battery_capacity_Wh=rc.BATTERY["capacity_Wh"],
                base_power_W=rc.COMPUTE_POWER_W + rc.SENSOR_POWER_W,
            )
            self.energy_affect = EnergyAffect()

        n_obs = self.model.nsensordata + 1 + (3 if self.energy else 0)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(n_obs,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(len(LEG_ACT_NAMES),), dtype=np.float32)

        self._steps = 0
        self._prev_x = 0.0
        self._prev_action = np.zeros(len(LEG_ACT_NAMES), dtype=np.float32)

    def _get_obs(self):
        base = [self.data.sensordata, [self.data.qpos[2]]]
        if self.energy:
            es, af = self.energy_state, self.energy_affect
            pain = af.state_pain(es.soc) + af.consumption_pain(es.power_W)
            base.append([es.soc, es.power_W / af.power_ref_W, pain])
        return np.concatenate(base).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        # 다리 관절 초기각에 소량 노이즈 (freejoint 7 + 허리3 + 목2 이후가 다리)
        self.data.qpos[12:] += self.np_random.uniform(-0.04, 0.04, size=self.model.nq - 12)
        self.data.ctrl[:] = self.home_ctrl
        mujoco.mj_forward(self.model, self.data)
        self._steps = 0
        self._prev_x = float(self.data.qpos[0])
        self._prev_action = np.zeros(len(LEG_ACT_NAMES), dtype=np.float32)
        if self.energy:
            # 초기 SoC 랜덤화: 저에너지 상태의 고통도 경험하도록 (domain randomization)
            soc0 = float(self.np_random.uniform(0.25, 1.0))
            self.energy_state.reset(soc0)
            self.energy_affect.reset(soc0)
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        targets = np.clip(self.leg_home + action * ACTION_SCALE, self.leg_lo, self.leg_hi)
        self.data.ctrl[:] = self.home_ctrl
        self.data.ctrl[self.leg_act_ids] = targets

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        x = float(self.data.qpos[0])
        dt = self.model.opt.timestep * self.frame_skip
        forward_vel = (x - self._prev_x) / dt
        self._prev_x = x

        upright = float(self.data.xmat[self.front_id].reshape(3, 3)[2, 2])
        z = float(self.data.qpos[2])
        tilt_penalty = 0.5 * max(0.0, 0.7 - upright)

        if self.mode == "sprint":
            ctrl_cost = 0.01 * float(np.sum(np.square(action)))
            reward = 2.0 * forward_vel - ctrl_cost - tilt_penalty + 0.5
        else:  # walk: 목표 속도 추종 + 동작 크기·급격함 페널티 강화(저속·저에너지 보행 유도)
            speed_error = abs(forward_vel - self.target_speed)
            ctrl_cost = 0.03 * float(np.sum(np.square(action)))
            jerk_cost = 0.02 * float(np.sum(np.square(action - self._prev_action)))
            reward = 1.5 * (1.0 - np.tanh(2.0 * speed_error)) - ctrl_cost - jerk_cost - tilt_penalty + 0.5
        self._prev_action = action.copy()

        info = {"forward_vel": forward_vel, "upright": upright, "z": z}
        depleted = False
        if self.energy:
            power_W, soc = self.energy_state.step(self.data, dt)
            energy_reward, pain = self.energy_affect.reward_terms(soc, power_W)
            reward += energy_reward
            depleted = soc <= 0.0  # 완전 방전 = "기절"
            info.update({"power_W": power_W, "soc": soc, "pain": pain})

        fell = (z < 0.14) or (upright < 0.4)
        if fell or depleted:
            reward -= 5.0
        self._steps += 1

        obs = self._get_obs()
        return obs, reward, bool(fell or depleted), self._steps >= self.max_episode_steps, info
