"""
train_don1.py로 학습한 PPO 정책(체크포인트 포함)을 불러와 mujoco.viewer로
don1의 실제 보행을 확인한다. 시각화 가능한 요소를 최대한 켜서 보여준다:
  - 접촉점(contact point) / 접촉힘(contact force) 화살표
  - 관절 축, 액추에이터 방향 표시
  - 터치 센서 site 마커 (group 4)
  - 로봇을 자동으로 따라가는 트래킹 카메라
  - 1초마다 보상/전진속도/기울어짐/센서값을 터미널에 출력
사용법: python watch_don1.py [체크포인트.zip 경로]  (생략 시 models/ 안의 최신 체크포인트 자동 선택)
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
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from don1_env import Don1Env

MODELS_DIR = "models"


def find_latest_checkpoint():
    """models/ 안에서 가장 스텝 수가 큰 체크포인트(zip)와 그에 대응하는 vecnormalize(pkl)를 찾는다."""
    final = os.path.join(MODELS_DIR, "don1_ppo_final.zip")
    final_vecnorm = os.path.join(MODELS_DIR, "don1_vecnormalize_final.pkl")
    if os.path.exists(final) and os.path.exists(final_vecnorm):
        return final, final_vecnorm

    candidates = glob.glob(os.path.join(MODELS_DIR, "don1_ppo_*_steps.zip"))
    if not candidates:
        raise FileNotFoundError("models/ 폴더에 don1 체크포인트가 없습니다. 먼저 train_don1.py를 실행하세요.")

    def step_count(path):
        m = re.search(r"don1_ppo_(\d+)_steps\.zip", path)
        return int(m.group(1)) if m else -1

    best = max(candidates, key=step_count)
    steps = step_count(best)
    vecnorm = os.path.join(MODELS_DIR, f"don1_ppo_vecnormalize_{steps}_steps.pkl")
    return best, vecnorm


def main():
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        vecnorm_path = model_path.replace("don1_ppo_", "don1_ppo_vecnormalize_").replace(".zip", ".pkl")
    else:
        model_path, vecnorm_path = find_latest_checkpoint()

    print(f"체크포인트 로드: {model_path}")
    print(f"VecNormalize 로드: {vecnorm_path}")

    env = Don1Env()
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize.load(vecnorm_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    model = PPO.load(model_path)
    sensor_names = [mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_SENSOR, i) for i in range(env.model.nsensor)]

    obs = vec_env.reset()
    ep_reward = 0.0
    last_print = time.time()

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        # --- 시각화 옵션을 최대한 켠다 ---
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False

        # 터치 센서 site(group=4)를 눈에 보이게 켠다. site 그룹은 기본적으로 꺼져 있을 수 있다.
        viewer.opt.sitegroup[4] = True

        # 로봇 몸통을 자동으로 따라가는 트래킹 카메라로 전환.
        viewer.cam.trackbodyid = env.torso_id
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.distance = 1.2
        viewer.cam.azimuth = 120
        viewer.cam.elevation = -20

        while viewer.is_running():
            step_start = time.time()

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            ep_reward += reward[0]

            if done[0]:
                print(f"에피소드 종료. 누적 보상={ep_reward:.1f}")
                ep_reward = 0.0
                obs = vec_env.reset()

            viewer.sync()

            if time.time() - last_print >= 1.0:
                last_print = time.time()
                d = env.data
                upright = float(d.xmat[env.torso_id].reshape(3, 3)[2, 2])
                readings = ", ".join(f"{n}={v:.2f}" for n, v in zip(
                    sensor_names,
                    [d.sensor(n).data[0] if d.sensor(n).data.size == 1 else round(float(np.linalg.norm(d.sensor(n).data)), 2)
                     for n in sensor_names],
                ))
                print(f"[t={d.time:5.1f}s] torso_z={d.qpos[2]:.2f} upright={upright:.2f} "
                      f"reward={reward[0]:.2f} | {readings}", flush=True)

            time_until_next = env.model.opt.timestep * env.frame_skip - (time.time() - step_start)
            if time_until_next > 0:
                time.sleep(time_until_next)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        print("스크립트 종료됨", flush=True)
