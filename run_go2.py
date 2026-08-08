"""
Unitree Go2 (go2_description) 사족보행 로봇을 MuJoCo Menagerie에서 불러와,
사인파(Sine Wave)로 다리 관절을 움직이는 시뮬레이션.
mujoco.viewer.launch_passive로 3D 뷰어 창을 띄운다.
"""
import time

import mujoco
import mujoco.viewer
import os

import numpy as np
from robot_descriptions import go2_mj_description

# Go2 액추에이터는 전부 순수 토크(motor) 제어이므로(PD 내장 X),
# 목표 각도를 따라가도록 자체 PD 제어를 적용한 뒤 사인파로 목표 각도를 흔든다.
KP = 60.0   # 위치 게인
KD = 3.0    # 속도(댐핑) 게인

FREQ_HZ = 0.8            # 다리 움직임 주파수
THIGH_AMPLITUDE = 0.35   # thigh(허벅지) 관절 진폭 [rad]
CALF_AMPLITUDE = 0.35    # calf(무릎) 관절 진폭 [rad]

# 대각선 다리(FL-RR, FR-RL)가 반대 위상으로 움직이는 트롯(trot) 형태의 걸음걸이.
LEG_PHASE = {
    "FL": 0.0,
    "RR": 0.0,
    "FR": np.pi,
    "RL": np.pi,
}

LEGS = ["FL", "FR", "RL", "RR"]


def main():
    # go2.xml 자체에는 바닥(floor)이 없다 — 같은 폴더의 scene.xml이 바닥/조명을 포함해
    # go2.xml을 include 하므로, 로봇이 허공에서 끝없이 낙하하지 않도록 scene.xml을 불러온다.
    scene_path = os.path.join(
        os.path.dirname(go2_mj_description.MJCF_PATH), "scene.xml"
    )
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    # "home" 키프레임(서 있는 기본 자세)으로 초기화.
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home_id)
    else:
        mujoco.mj_resetData(model, data)

    # thigh / calf 관절의 기본(정지) 각도를 현재 자세에서 읽어 목표 각도의 기준으로 사용한다.
    base_angle = {}
    actuator_id = {}
    for leg in LEGS:
        for part in ("thigh", "calf"):
            joint_name = f"{leg}_{part}_joint"
            actuator_name = f"{leg}_{part}"
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            qpos_adr = model.jnt_qposadr[joint_id]
            base_angle[(leg, part)] = data.qpos[qpos_adr]
            actuator_id[(leg, part)] = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
            )

    dof_adr = {}
    qpos_adr = {}
    for leg in LEGS:
        for part in ("thigh", "calf"):
            joint_name = f"{leg}_{part}_joint"
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            dof_adr[(leg, part)] = model.jnt_dofadr[joint_id]
            qpos_adr[(leg, part)] = model.jnt_qposadr[joint_id]

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            t = data.time
            phase_base = 2.0 * np.pi * FREQ_HZ * t

            for leg in LEGS:
                phase = phase_base + LEG_PHASE[leg]

                # thigh와 calf를 서로 반대 방향으로 굽혔다 펴서 다리를 접었다 펴는 동작을 만든다.
                thigh_target = base_angle[(leg, "thigh")] + THIGH_AMPLITUDE * np.sin(phase)
                calf_target = base_angle[(leg, "calf")] - CALF_AMPLITUDE * np.sin(phase)

                for part, target in (("thigh", thigh_target), ("calf", calf_target)):
                    q = data.qpos[qpos_adr[(leg, part)]]
                    qd = data.qvel[dof_adr[(leg, part)]]
                    torque = KP * (target - q) - KD * qd
                    act_id = actuator_id[(leg, part)]
                    ctrl_range = model.actuator_ctrlrange[act_id]
                    data.ctrl[act_id] = np.clip(torque, ctrl_range[0], ctrl_range[1])

            mujoco.mj_step(model, data)
            viewer.sync()

            # 물리 시뮬레이션의 timestep과 실제 시간(wall-clock)을 동기화한다.
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        print("스크립트 종료됨", flush=True)
