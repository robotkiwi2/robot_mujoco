"""
직접 설계한 4족 로봇 'don1'(박스 몸체 + 무릎 없는 막대 다리 4개) 시뮬레이션.
힙 관절 4개를 사인파로 움직이고, 부착된 센서(관절각/IMU/터치)를 주기적으로 출력한다.
"""
import time

import mujoco
import mujoco.viewer
import numpy as np

MODEL_PATH = "robots/don1/scene.xml"

# don1의 액추에이터도 순수 토크(motor) 제어라서 목표 각도를 PD로 추종시킨다.
KP = 8.0
KD = 0.4

FREQ_HZ = 0.6
AMPLITUDE = 0.5  # [rad]

# 대각선 다리(FL-RR, FR-RL)가 반대 위상으로 움직이는 트롯 걸음걸이.
LEG_PHASE = {
    "FL": 0.0,
    "RR": 0.0,
    "FR": np.pi,
    "RL": np.pi,
}
LEGS = ["FL", "FR", "RL", "RR"]


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    joint_id = {leg: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_hip") for leg in LEGS}
    qpos_adr = {leg: model.jnt_qposadr[joint_id[leg]] for leg in LEGS}
    dof_adr = {leg: model.jnt_dofadr[joint_id[leg]] for leg in LEGS}
    actuator_id = {leg: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{leg}_hip_motor") for leg in LEGS}

    sensor_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i) for i in range(model.nsensor)]

    last_print = 0.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            phase_base = 2.0 * np.pi * FREQ_HZ * data.time
            for leg in LEGS:
                target = AMPLITUDE * np.sin(phase_base + LEG_PHASE[leg])
                q = data.qpos[qpos_adr[leg]]
                qd = data.qvel[dof_adr[leg]]
                torque = KP * (target - q) - KD * qd
                act_id = actuator_id[leg]
                ctrl_range = model.actuator_ctrlrange[act_id]
                data.ctrl[act_id] = np.clip(torque, ctrl_range[0], ctrl_range[1])

            mujoco.mj_step(model, data)
            viewer.sync()

            # 1초에 한 번 센서 값을 터미널에 출력.
            if data.time - last_print >= 1.0:
                last_print = data.time
                readings = ", ".join(
                    f"{name}={val:.2f}" if np.isscalar(val) else f"{name}={np.round(val, 2)}"
                    for name, val in zip(sensor_names, [
                        data.sensor(name).data if data.sensor(name).data.size > 1 else data.sensor(name).data[0]
                        for name in sensor_names
                    ])
                )
                print(f"[t={data.time:5.1f}s] {readings}", flush=True)

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
