"""
don2 '좌회전 보행(turn_left)' 스킬 PPO 학습 — 에너지 고통 활성화.
목표 회전율(gyro z) 0.6 rad/s 추종 + 전진 유지 보너스 + 전력/저에너지 고통.
"""
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from don2_env import Don2Env

COMBO = "don2__flat__turn_left"
N_ENVS = 8
TOTAL_TIMESTEPS = 3_000_000

MODEL_DIR = os.path.join("models", COMBO)
LOG_DIR = os.path.join("runs", COMBO)


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    vec_env = make_vec_env(Don2Env, n_envs=N_ENVS,
                           env_kwargs=dict(mode="turn_left", target_yaw_rate=0.6, energy=True))
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        "MlpPolicy", vec_env, verbose=1,
        n_steps=512, batch_size=1024, n_epochs=10,
        learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, ent_coef=0.0,
        tensorboard_log=LOG_DIR,
    )

    cb = CheckpointCallback(save_freq=max(100_000 // N_ENVS, 1), save_path=MODEL_DIR,
                            name_prefix="ppo", save_vecnormalize=True)

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=cb, progress_bar=False)
    model.save(os.path.join(MODEL_DIR, "ppo_final"))
    vec_env.save(os.path.join(MODEL_DIR, "vecnormalize_final.pkl"))
    print("학습 완료:", MODEL_DIR)


if __name__ == "__main__":
    main()
