"""
아주 단순한 상자(Box)가 중력에 의해 바닥으로 떨어지는 MuJoCo 시뮬레이션.
mujoco.viewer를 이용해 3D 뷰어 창을 띄워 시각적으로 확인한다.
"""
import time

import mujoco
import mujoco.viewer

# 바닥(plane)과 상자(box) 하나로 구성된 최소한의 MJCF 모델.
# 상자는 바닥에서 1m 위에서 시작해 중력에 의해 떨어진다.
MODEL_XML = """
<mujoco model="box_drop">
  <option gravity="0 0 -9.81" timestep="0.002"/>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.2 0.3 0.4" rgb2="0.3 0.4 0.5"
             width="300" height="300"/>
    <material name="grid_mat" texture="grid" texrepeat="5 5" reflectance="0.2"/>
  </asset>

  <worldbody>
    <light name="top_light" pos="0 0 3" dir="0 0 -1" diffuse="1 1 1"/>
    <geom name="floor" type="plane" size="2 2 0.1" material="grid_mat"/>

    <body name="box" pos="0 0 1">
      <freejoint/>
      <geom name="box_geom" type="box" size="0.1 0.1 0.1" rgba="0.8 0.2 0.2 1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""


def main():
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    data = mujoco.MjData(model)

    # 상자가 착지한 뒤 이 시간(초)만큼 더 지나면 낙하를 처음부터 반복한다.
    settle_hold_seconds = 2.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        reset_at = None
        while viewer.is_running():
            step_start = time.time()

            mujoco.mj_step(model, data)
            viewer.sync()

            # 상자가 거의 멈추면(착지) 잠시 보여준 뒤 낙하를 재시작한다.
            if reset_at is None:
                if data.qpos[2] <= 0.1005 and abs(data.qvel[2]) < 0.01:
                    reset_at = time.time() + settle_hold_seconds
            elif time.time() >= reset_at:
                mujoco.mj_resetData(model, data)
                data.qpos[2] = 1.0
                reset_at = None

            # 실시간 속도에 맞춰 재생되도록 남는 시간만큼 대기한다.
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
