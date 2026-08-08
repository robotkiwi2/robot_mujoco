"""
don2 육안 검증용 뷰어 (framework 구현 전 임시 하네스).
home 자세로 서서 목을 좌우로 천천히 돌리고, 주기적으로 발가락을 쥐었다 편다
— 관절/텐던/센서가 눈으로 확인 가능하도록. 학습 없음.
실행: ./.venv/Scripts/python.exe robots/don2/pose_test.py
"""
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate import load_with_floor


def main():
    model = load_with_floor()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    neck_yaw = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "neck_yaw_act")
    neck_pitch = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "neck_pitch_act")
    spine_yaw = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "spine_yaw_act")
    spine_roll = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "spine_roll_act")
    toe_acts = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{leg}_toe_{t}_act")
                for leg in ["FL", "FR", "RL", "RR"] for t in ["f1", "f2", "b"]]

    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_front")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        viewer.opt.sitegroup[4] = True   # 접촉센서/마운트 site 표시
        viewer.opt.sitegroup[5] = True   # 텐던 경유점 표시
        viewer.cam.trackbodyid = torso_id
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.distance = 1.0
        viewer.cam.elevation = -15

        while viewer.is_running():
            step_start = time.time()
            t = data.time

            # 목: 좌우 스캔 + 가벼운 상하
            data.ctrl[neck_yaw] = 0.8 * np.sin(0.5 * 2 * np.pi * 0.2 * t * 2)
            data.ctrl[neck_pitch] = 0.2 * np.sin(2 * np.pi * 0.1 * t)

            # 허리: 천천히 팬(좌우 굽힘) + 미세 롤 — 상하체 관절 동작 확인용
            data.ctrl[spine_yaw] = 0.25 * np.sin(2 * np.pi * 0.08 * t)
            data.ctrl[spine_roll] = 0.1 * np.sin(2 * np.pi * 0.05 * t)

            # 발가락: 3초 주기로 쥐었다(장력 6N) 폈다(0)
            grip = 6.0 if (t % 6.0) < 3.0 else 0.0
            for a in toe_acts:
                data.ctrl[a] = grip

            mujoco.mj_step(model, data)
            viewer.sync()

            dt = model.opt.timestep - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        print("pose_test 종료", flush=True)
