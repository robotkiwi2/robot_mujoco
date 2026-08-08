"""
don2 정책 재생 뷰어.
사용법: python watch_don2.py [combo] [체크포인트.zip]
  combo: forward(스프린트, 에너지 도입 전 관측 150) | walk(에너지 고통 포함 관측 153). 기본 forward.
"""
import glob
import os
import re
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from don2_env import Don2Env

# combo별 저장 폴더와 환경 설정 (관측 차원이 달라 반드시 일치해야 함)
COMBOS = {
    "forward": dict(dir="don2__flat__forward", env_kwargs=dict(mode="sprint", energy=False)),
    "walk": dict(dir="don2__flat__walk", env_kwargs=dict(mode="walk", target_speed=0.35, energy=True)),
}


def find_latest(model_dir):
    final = os.path.join(model_dir, "ppo_final.zip")
    if os.path.exists(final):
        return final, os.path.join(model_dir, "vecnormalize_final.pkl")
    cands = glob.glob(os.path.join(model_dir, "ppo_*_steps.zip"))
    if not cands:
        raise FileNotFoundError(f"{model_dir}에 체크포인트가 없습니다.")
    steps = lambda p: int(re.search(r"ppo_(\d+)_steps", p).group(1))
    best = max(cands, key=steps)
    return best, os.path.join(model_dir, f"ppo_vecnormalize_{steps(best)}_steps.pkl")


def main():
    combo = sys.argv[1] if len(sys.argv) > 1 else "forward"
    cfg = COMBOS[combo]
    model_dir = os.path.join("models", cfg["dir"])
    if len(sys.argv) > 2:
        model_path = sys.argv[2]
        vec_path = model_path.replace("ppo_", "ppo_vecnormalize_").replace(".zip", ".pkl")
    else:
        model_path, vec_path = find_latest(model_dir)
    print("combo:", combo, "| 로드:", model_path)

    env = Don2Env(**cfg["env_kwargs"])
    vec = DummyVecEnv([lambda: env])
    vec = VecNormalize.load(vec_path, vec)
    vec.training = False
    vec.norm_reward = False
    policy = PPO.load(model_path)

    obs = vec.reset()
    ep_r = 0.0
    last = 0.0
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = True
        viewer.cam.trackbodyid = env.front_id
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.distance = 1.2
        viewer.cam.elevation = -15

        while viewer.is_running():
            t0 = time.time()
            action, _ = policy.predict(obs, deterministic=True)
            obs, r, done, info = vec.step(action)
            ep_r += r[0]
            if done[0]:
                print(f"에피소드 종료. 누적보상 {ep_r:.1f}")
                ep_r = 0.0
                obs = vec.reset()
            viewer.sync()
            if time.time() - last >= 1.0:
                last = time.time()
                i = info[0]
                extra = ""
                if "power_W" in i:
                    extra = f" | P={i['power_W']:.0f}W soc={i['soc']:.3f} pain={i['pain']:.2f}"
                print(f"[t={env.data.time:5.1f}] vx={i['forward_vel']:+.2f} m/s "
                      f"upright={i['upright']:.2f} z={i['z']:.2f} x={env.data.qpos[0]:+.2f}{extra}", flush=True)
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
