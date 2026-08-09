"""
don2 두뇌 v0 데모 — 욕구/호르몬이 행동을 고르는 통합 루프.

시나리오:
- 기본: patrol 프로그램 (서기 2초 <-> 걷기 5초 반복)
- 강한 충격(낙하 등) → 아드레날린 급등 → 연합령이 startle_freeze로 인터럽트,
  진정되면 patrol 복귀
- SoC < 27% → rest (에너지 보존)

조작: [6] 로봇을 0.6m 들어올려 떨어뜨림(놀람 반사 테스트)  [R] 리셋
실행: python run_don2_brain.py
"""
import time

import mujoco
import mujoco.viewer
import numpy as np

from don2_env import Don2Env
from framework.brain import Brain

drop_requested = False
reset_requested = False


def key_callback(keycode):
    global drop_requested, reset_requested
    ch = chr(keycode) if 0 < keycode < 256 else ""
    if ch == "6":
        drop_requested = True
    elif ch.upper() == "R":
        reset_requested = True


def main():
    global drop_requested, reset_requested

    env = Don2Env(mode="walk", energy=True)   # mode는 물리 매핑용(16관절 다리 제어)
    obs, _ = env.reset(seed=0)
    env.energy_state.reset(0.9)               # 데모: 넉넉한 초기 배터리
    env.energy_affect.reset(0.9)

    brain = Brain(env, Don2Env)
    print("레퍼토리:", brain.cerebellum.available())

    reward = 0.0
    last_print = 0.0
    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=key_callback) as viewer:
        viewer.cam.trackbodyid = env.front_id
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.distance = 1.4
        viewer.cam.elevation = -15
        print("\n조작: [6] 낙하(놀람 테스트)  [R] 리셋\n")

        while viewer.is_running():
            t0 = time.time()

            if reset_requested:
                obs, _ = env.reset()
                env.energy_state.reset(0.9)
                env.energy_affect.reset(0.9)
                reset_requested = False
            if drop_requested:
                env.data.qpos[2] += 0.6       # 들어올려 떨어뜨리기
                mujoco.mj_forward(env.model, env.data)
                drop_requested = False
                print(">> 낙하!")

            action, binfo = brain.step(obs, reward)
            obs, reward, term, trunc, info = env.step(action)
            if term:
                obs, _ = env.reset()
                env.energy_state.reset(0.9)
                env.energy_affect.reset(0.9)
                brain.reset()   # 시퀀서 시계 동기화 (필수 — framework/brain.py 참조)

            viewer.sync()

            if time.time() - last_print >= 1.0:
                last_print = time.time()
                p = binfo["percept"]
                overlay = (f"program: {binfo['step']}\nskill: {binfo['skill']}",
                           f"soc={p['soc']:.2f}  P={p['power_W']:.0f}W\n"
                           f"adrenaline={p['adrenaline']:.2f}  cortisol={p['cortisol']:.3f}\n"
                           f"damage={p['damage']:.2f}  pain={p['energy_pain']+p['damage_pain']:.2f}")
                viewer.set_texts([
                    (mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, "brain", overlay[0]),
                    (mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, "drives", overlay[1]),
                ])
                print(f"[{binfo['step']}] vx={info['forward_vel']:+.2f} soc={p['soc']:.2f} "
                      f"A={p['adrenaline']:.2f} dmg={p['damage']:.2f}", flush=True)

            dt = env.model.opt.timestep * env.frame_skip - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        print("종료", flush=True)
