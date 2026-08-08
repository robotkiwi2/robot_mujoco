"""
don2 스킬 인터랙티브 전환 뷰어 — 키보드로 학습된 행동을 직접 지시한다.
DESIGN.md의 skill_manager 개념을 사람이 수동으로 대신하는 임시 버전
(나중엔 매니저가 욕구에 따라 자동으로 이 전환을 하게 됨).

조작 (MuJoCo 뷰어 내장 단축키와 충돌하지 않는 키만 사용 — 아래 참고):
  6 : sprint 스킬 (최대속도)
  7 : walk 스킬 (목표속도 0.35m/s + 에너지 절약)
  8 : turn_left 스킬 (좌회전 0.6 rad/s + 전진 유지)
  9 : 세 스킬 전부 최신 체크포인트로 즉시 갱신(핫 리로드, 창 유지)
  R : 자세 리셋(home)

주의: MuJoCo 뷰어는 커스텀 key_callback을 등록해도 내장 단축키가 함께 발동한다.
숫자 0~5는 geom 그룹(mjNGROUP=6) 표시 토글로 예약되어 있어 피했다(6~9는 미사용 확인).
알파벳은 다수가 시각화 플래그에 물려있어(L=Convex Hull 등) 사용하지 않는다.
현재 스킬/조작법은 뷰어 화면에도 오버레이 텍스트로 표시된다.

사용법: python watch_don2_interactive.py
"""
import time

import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from don2_env import Don2Env
from watch_don2 import find_latest

SKILLS = {
    "sprint": dict(dir="models/don2__flat__forward", obs_dim=150,
                   env_kwargs=dict(mode="sprint", energy=False)),
    "walk": dict(dir="models/don2__flat__walk", obs_dim=153,
                 env_kwargs=dict(mode="walk", target_speed=0.35, energy=True)),
    "turn_left": dict(dir="models/don2__flat__turn_left", obs_dim=153,
                       env_kwargs=dict(mode="turn_left", target_yaw_rate=0.6, energy=True)),
}

current_skill = "sprint"   # 시작 스킬
reset_requested = False
reload_requested = False


def key_callback(keycode):
    global current_skill, reset_requested, reload_requested
    ch = chr(keycode) if 0 < keycode < 256 else ""
    if ch == "6":
        current_skill = "sprint"
        print(">> 스킬 전환: sprint")
    elif ch == "7":
        current_skill = "walk"
        print(">> 스킬 전환: walk")
    elif ch == "8":
        current_skill = "turn_left"
        print(">> 스킬 전환: turn_left")
    elif ch == "9":
        reload_requested = True
    elif ch.upper() == "R":
        reset_requested = True
        print(">> 자세 리셋")


def load_skill(name, cfg):
    model_path, vec_path = find_latest(cfg["dir"])
    policy = PPO.load(model_path)
    dummy = DummyVecEnv([lambda kw=cfg["env_kwargs"]: Don2Env(**kw)])
    normalizer = VecNormalize.load(vec_path, dummy)
    normalizer.training = False
    return policy, normalizer, model_path


def main():
    global reset_requested, reload_requested

    policies, normalizers = {}, {}
    for name, cfg in SKILLS.items():
        policies[name], normalizers[name], model_path = load_skill(name, cfg)
        print(f"[{name}] 로드: {model_path}")

    # 물리 시뮬레이션은 walk(에너지 포함, 153차원)를 마스터로 사용 — sprint는 앞 150차원만 사용
    env = Don2Env(mode="walk", energy=True)
    obs, _ = env.reset(seed=0)

    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=key_callback) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        viewer.cam.trackbodyid = env.front_id
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.distance = 1.3
        viewer.cam.elevation = -15

        last_print = 0.0
        print("\n조작: [6] sprint  [7] walk  [8] turn_left  [9] 최신으로 갱신  [R] 리셋\n")

        def refresh_overlay(info):
            marks = {name: (">" if name == current_skill else " ") for name in SKILLS}
            menu = (f"{marks['sprint']}[6] sprint\n"
                    f"{marks['walk']}[7] walk\n"
                    f"{marks['turn_left']}[8] turn_left\n"
                    f" [9] reload latest   [R] reset")
            stats = f"vx={info['forward_vel']:+.2f} m/s  upright={info['upright']:.2f}"
            if "power_W" in info:
                stats += f"\nP={info['power_W']:.0f}W  soc={info['soc']:.2f}  pain={info['pain']:.2f}"
            if current_skill == "turn_left":
                stats += f"\nyaw={info['yaw_rate']:+.2f} rad/s"
            viewer.set_texts([
                (mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, "skills", menu),
                (mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, "status", stats),
            ])

        while viewer.is_running():
            t0 = time.time()

            if reload_requested:
                for name, cfg in SKILLS.items():
                    policies[name], normalizers[name], model_path = load_skill(name, cfg)
                    print(f"[{name}] 갱신: {model_path}")
                reload_requested = False

            if reset_requested:
                obs, _ = env.reset()
                reset_requested = False

            raw_obs = obs[: SKILLS[current_skill]["obs_dim"]]
            norm_obs = normalizers[current_skill].normalize_obs(raw_obs[None, :])
            action, _ = policies[current_skill].predict(norm_obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action[0])

            if term or trunc:
                obs, _ = env.reset()

            viewer.sync()

            if time.time() - last_print >= 1.0:
                last_print = time.time()
                refresh_overlay(info)
                extra = f" P={info['power_W']:.0f}W soc={info['soc']:.2f}" if "power_W" in info else ""
                extra += f" yaw={info['yaw_rate']:+.2f}rad/s" if current_skill == "turn_left" else ""
                print(f"[{current_skill}] vx={info['forward_vel']:+.2f} m/s "
                      f"upright={info['upright']:.2f}{extra}", flush=True)

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
