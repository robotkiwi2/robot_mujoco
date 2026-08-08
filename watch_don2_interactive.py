"""
don2 스킬 인터랙티브 전환 뷰어 — 학습된 행동을 직접 지시한다.
DESIGN.md의 skill_manager 개념을 사람이 수동으로 대신하는 임시 버전
(나중엔 매니저가 욕구에 따라 자동으로 이 전환을 하게 됨).

선택 방식 두 가지 (스킬 개수가 늘어나도 그대로 확장됨):
  1) 뷰어 창에서: 6=위로 이동  7=아래로 이동  (커서가 가리키는 스킬이 즉시 활성화)
                  9=전체 최신 체크포인트로 갱신   R=자세 리셋
     * MuJoCo 뷰어는 커스텀 key_callback을 등록해도 내장 단축키가 함께 발동한다.
       0~5는 geom 그룹(mjNGROUP=6) 토글, 다수 알파벳은 시각화 플래그(L=Convex Hull 등)에
       물려있어 충돌 없는 6/7/9와 R(리셋)만 사용한다.
  2) 터미널에 스킬 이름을 입력 + Enter  (예: walk, turn_left, toe_curl)
     기타 명령: list(목록), reload(전체 최신 갱신), reset(자세 리셋), quit(종료)
현재 스킬/조작법은 뷰어 화면에도 오버레이 텍스트로 표시된다(커서 위치 강조).

사용법: python watch_don2_interactive.py
"""
import threading
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
    "toe_curl": dict(dir="models/don2__flat__toe_curl", obs_dim=153,
                      env_kwargs=dict(mode="toe_curl", toe_curl_freq=0.4, energy=True)),
}
SKILL_NAMES = list(SKILLS)  # 목록 순서 = 커서 이동 순서

current_idx = 0  # 커서 위치 (SKILL_NAMES의 인덱스) — 이게 곧 "현재 스킬"의 유일한 출처
reset_requested = False
reload_requested = False
quit_requested = False


def current_skill_name():
    return SKILL_NAMES[current_idx]


def key_callback(keycode):
    global current_idx, reset_requested, reload_requested
    ch = chr(keycode) if 0 < keycode < 256 else ""
    if ch == "6":  # 위로
        current_idx = (current_idx - 1) % len(SKILL_NAMES)
        print(f">> 스킬 전환: {current_skill_name()}")
    elif ch == "7":  # 아래로
        current_idx = (current_idx + 1) % len(SKILL_NAMES)
        print(f">> 스킬 전환: {current_skill_name()}")
    elif ch == "9":
        reload_requested = True
    elif ch.upper() == "R":
        reset_requested = True
        print(">> 자세 리셋")


def stdin_loop():
    """터미널 입력으로도 스킬 전환 가능 — 몇 개가 되든 이름만 입력하면 된다."""
    global current_idx, reset_requested, reload_requested, quit_requested
    print(f"\n[터미널 입력] 스킬 이름을 입력하세요: {', '.join(SKILL_NAMES)}")
    print("             기타 명령: list / reload / reset / quit")
    print("             (뷰어 창에서는 6=위 7=아래 로 커서 이동)\n")
    while not quit_requested:
        try:
            cmd = input().strip().lower()
        except EOFError:
            break
        if cmd in SKILLS:
            current_idx = SKILL_NAMES.index(cmd)
            print(f">> 스킬 전환: {cmd}")
        elif cmd == "list":
            print("사용 가능한 스킬:", ", ".join(SKILL_NAMES), f"(현재: {current_skill_name()})")
        elif cmd == "reload":
            reload_requested = True
        elif cmd == "reset":
            reset_requested = True
        elif cmd in ("quit", "q", "exit"):
            quit_requested = True
        elif cmd:
            print(f"?? 알 수 없는 명령: '{cmd}' (list로 스킬 목록 확인)")


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

    threading.Thread(target=stdin_loop, daemon=True).start()

    # 물리 시뮬레이션은 에너지 포함(153차원)을 마스터로 사용 — energy=False 스킬은 앞부분만 사용
    env = Don2Env(mode="walk", energy=True)
    obs, _ = env.reset(seed=0)

    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=key_callback) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        viewer.cam.trackbodyid = env.front_id
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.distance = 1.3
        viewer.cam.elevation = -15

        last_print = 0.0

        def refresh_overlay(info):
            lines = []
            for i, name in enumerate(SKILL_NAMES):
                cursor = "> " if i == current_idx else "  "
                lines.append(f"{cursor}{name}")
            menu = "\n".join(lines) + "\n[6]up [7]down [9]reload  (터미널 이름 입력도 가능)"
            skill = current_skill_name()
            stats = f"vx={info['forward_vel']:+.2f} m/s  upright={info['upright']:.2f}"
            if "power_W" in info:
                stats += f"\nP={info['power_W']:.0f}W  soc={info['soc']:.2f}  pain={info['pain']:.2f}"
            if skill == "turn_left":
                stats += f"\nyaw={info['yaw_rate']:+.2f} rad/s"
            viewer.set_texts([
                (mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, "skills", menu),
                (mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, "status", stats),
            ])

        while viewer.is_running() and not quit_requested:
            t0 = time.time()

            if reload_requested:
                for name, cfg in SKILLS.items():
                    policies[name], normalizers[name], model_path = load_skill(name, cfg)
                    print(f"[{name}] 갱신: {model_path}")
                reload_requested = False

            if reset_requested:
                obs, _ = env.reset()
                reset_requested = False

            skill = current_skill_name()
            raw_obs = obs[: SKILLS[skill]["obs_dim"]]
            norm_obs = normalizers[skill].normalize_obs(raw_obs[None, :])
            action, _ = policies[skill].predict(norm_obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action[0])

            if term or trunc:
                obs, _ = env.reset()

            viewer.sync()

            if time.time() - last_print >= 1.0:
                last_print = time.time()
                refresh_overlay(info)
                extra = f" P={info['power_W']:.0f}W soc={info['soc']:.2f}" if "power_W" in info else ""
                extra += f" yaw={info['yaw_rate']:+.2f}rad/s" if skill == "turn_left" else ""
                print(f"[{skill}] vx={info['forward_vel']:+.2f} m/s "
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
