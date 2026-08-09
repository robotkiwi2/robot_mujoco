"""
don2 두뇌 데모 — MuJoCo 뷰어 + 조작 패널(tkinter) 동시 구동.

- 뷰어: 3D 시뮬레이션 (오버레이: 프로그램/욕구)
- 조작 패널: 상태 실시간 표시 + 자동/수동 전환, 프로그램/스킬/포즈 강제 실행,
  낙하 주입, SoC 슬라이더, 일시정지, 리셋 (framework/panel.py — 재사용 모듈)

실행: ./.venv/Scripts/python.exe run_don2_brain.py [--world nursery]
"""
import queue
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

from don2_env import Don2Env
from framework.brain import Brain
from framework.panel import ControlPanel
from framework.poses import POSES


def main():
    world = "nursery" if "--world" in sys.argv and "nursery" in sys.argv else "flat"
    env = Don2Env(mode="walk", energy=True, world=world)
    obs, _ = env.reset(seed=0)
    env.energy_state.reset(0.9)
    env.energy_affect.reset(0.9)

    brain = Brain(env, Don2Env)
    print("레퍼토리:", brain.cerebellum.available(), "+ 포즈", list(POSES))

    state = {}
    commands = queue.Queue()
    panel = ControlPanel(
        state, commands,
        skills=brain.cerebellum.available(),
        poses=list(POSES),
        programs=list(brain.association.library),
    )
    panel.start()

    reward, paused, last_print = 0.0, False, 0.0

    def do_reset():
        nonlocal obs, reward
        obs, _ = env.reset()
        env.energy_state.reset(0.9)
        env.energy_affect.reset(0.9)
        brain.reset()
        reward = 0.0

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.trackbodyid = env.front_id
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.distance = 1.4
        viewer.cam.elevation = -15

        while viewer.is_running():
            t0 = time.time()

            # ---- 패널 명령 처리 ----
            try:
                while True:
                    cmd = commands.get_nowait()
                    kind = cmd[0]
                    if kind == "auto":
                        brain.manual_skill = None
                        brain.association.override = None
                        print(">> 자동 모드 (욕구가 선택)")
                    elif kind == "program":
                        brain.manual_skill = None
                        brain.association.override = cmd[1]
                        print(f">> 프로그램 강제: {cmd[1]}")
                    elif kind == "skill":
                        brain.manual_skill = cmd[1]
                        print(f">> 스킬 수동 실행: {cmd[1]}")
                    elif kind == "pose":
                        brain.manual_skill = f"pose:{cmd[1]}"
                        print(f">> 포즈: {cmd[1]}")
                    elif kind == "drop":
                        env.data.qpos[2] += 0.6
                        mujoco.mj_forward(env.model, env.data)
                        print(">> 낙하!")
                    elif kind == "soc":
                        env.energy_state.reset(cmd[1])
                        env.energy_affect.reset(cmd[1])
                        print(f">> SoC = {cmd[1]:.2f}")
                    elif kind == "pause":
                        paused = not paused
                        print(">> 일시정지" if paused else ">> 재개")
                    elif kind == "reset":
                        do_reset()
                        print(">> 리셋")
            except queue.Empty:
                pass

            if paused:
                viewer.sync()
                time.sleep(0.05)
                continue

            action, binfo = brain.step(obs, reward)
            obs, reward, term, trunc, info = env.step(action)
            if term:
                do_reset()

            viewer.sync()

            # ---- 패널 상태 갱신 ----
            p = binfo["percept"]
            state.update({
                "mode": "수동" if brain.manual_skill else
                        ("강제:" + brain.association.override if brain.association.override else "자동"),
                "program": binfo["step"],
                "skill": binfo["skill"],
                "soc": p.get("soc", 0), "power_W": p.get("power_W", 0),
                "pain": p.get("energy_pain", 0) + p.get("damage_pain", 0),
                "adrenaline": p.get("adrenaline", 0), "cortisol": p.get("cortisol", 0),
                "damage": p.get("damage", 0), "vx": info["forward_vel"],
            })

            if time.time() - last_print >= 2.0:
                last_print = time.time()
                overlay = (f"program: {binfo['step']}\nskill: {binfo['skill']}",
                           f"soc={p.get('soc',0):.2f}  P={p.get('power_W',0):.0f}W\n"
                           f"A={p.get('adrenaline',0):.2f} C={p.get('cortisol',0):.3f} "
                           f"dmg={p.get('damage',0):.2f}")
                viewer.set_texts([
                    (mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, "brain", overlay[0]),
                    (mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, "drives", overlay[1]),
                ])

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
