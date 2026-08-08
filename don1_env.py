"""
don1(박스 몸체 + 무릎 없는 막대 다리 4개) 로봇을 SB3(PPO 등)로 학습시키기 위한
Gymnasium 환경. 목표는 전진 보행(forward locomotion).

관측(observation) = don1.xml에 정의된 모든 센서값(jointpos 4, jointvel 4,
imu_orient 4, imu_gyro 3, imu_acc 3, touch 6 = 24) + 몸통 높이(1) = 25차원.

행동(action) = 4개 힙 모터에 대한 [-1, 1] 정규화 값 (ctrl_range로 스케일링).

보상(reward) = 전진 속도 - 관절 사용량 페널티 - 기울어짐 페널티 + 생존 보너스,
쓰러지면(높이가 너무 낮아지거나 몸이 심하게 기울면) 에피소드 종료.
"""
import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

# 절대경로에 한글 폴더명("앱 개발")이 섞이면 MuJoCo의 XML 로더가 열지 못하는 Windows 이슈가
# 있어, 프로젝트 루트를 작업 디렉터리로 실행한다는 전제로 상대경로를 사용한다.
MODEL_PATH = "robots/don1/scene.xml"

LEGS = ["FL", "FR", "RL", "RR"]


class Don1Env(gym.Env):
    metadata = {"render_modes": [], "render_fps": 100}

    def __init__(self, frame_skip: int = 5, max_episode_steps: int = 500):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps

        self.torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        self.actuator_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{leg}_hip_motor") for leg in LEGS]
        )
        self.ctrl_scale = self.model.actuator_ctrlrange[self.actuator_ids, 1].copy()  # 각 액추에이터 최대 토크

        n_obs = self.model.nsensordata + 1  # 센서 전부 + 몸통 높이
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        self._elapsed_steps = 0
        self._prev_x = 0.0

    def _get_obs(self):
        torso_z = np.array([self.data.qpos[2]], dtype=np.float32)
        return np.concatenate([self.data.sensordata, torso_z]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # 관절 초기 각도에 약간의 랜덤 노이즈를 줘서 매 에피소드가 조금씩 다르게 시작하게 함
        # (특정 초기자세에만 과적합되지 않도록 하는 일반적인 domain randomization).
        n_leg_joints = self.model.nq - 7  # freejoint(7) 이후가 다리 관절
        self.data.qpos[7:] += self.np_random.uniform(-0.05, 0.05, size=n_leg_joints)

        mujoco.mj_forward(self.model, self.data)

        self._elapsed_steps = 0
        self._prev_x = float(self.data.qpos[0])

        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        self.data.ctrl[self.actuator_ids] = action * self.ctrl_scale

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()

        x = float(self.data.qpos[0])
        dt = self.model.opt.timestep * self.frame_skip
        forward_vel = (x - self._prev_x) / dt
        self._prev_x = x

        # 몸통 z축이 세계 z축과 얼마나 정렬돼 있는지 (1=완전 직립, 0=옆으로 누움, -1=뒤집힘)
        upright = float(self.data.xmat[self.torso_id].reshape(3, 3)[2, 2])
        torso_z = float(self.data.qpos[2])

        ctrl_cost = 0.01 * float(np.sum(np.square(action)))
        tilt_penalty = 0.5 * max(0.0, 0.7 - upright)
        alive_bonus = 0.5

        reward = 2.0 * forward_vel - ctrl_cost - tilt_penalty + alive_bonus

        fell = (torso_z < 0.12) or (upright < 0.3)
        terminated = bool(fell)
        if terminated:
            reward -= 5.0

        self._elapsed_steps += 1
        truncated = self._elapsed_steps >= self.max_episode_steps

        info = {"forward_vel": forward_vel, "upright": upright, "torso_z": torso_z}
        return obs, reward, terminated, truncated, info

    def close(self):
        pass
