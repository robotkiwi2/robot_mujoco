"""
don1 로봇에게 전진 보행을 학습시키는 SB3 PPO 학습 스크립트.
여러 환경을 병렬로 돌려(SubprocVecEnv) 학습 속도를 높이고,
관측값은 VecNormalize로 정규화한다(센서마다 단위/스케일이 크게 달라서 필수적).
"""
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from don1_env import Don1Env

N_ENVS = 8
TOTAL_TIMESTEPS = 2_000_000

LOG_DIR = "runs/don1_ppo"
MODEL_DIR = "models"


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    vec_env = make_vec_env(Don1Env, n_envs=N_ENVS)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        n_steps=512,
        batch_size=512,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.0,
        tensorboard_log=LOG_DIR,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // N_ENVS, 1),
        save_path=MODEL_DIR,
        name_prefix="don1_ppo",
        save_vecnormalize=True,
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback, progress_bar=False)

    model.save(os.path.join(MODEL_DIR, "don1_ppo_final"))
    vec_env.save(os.path.join(MODEL_DIR, "don1_vecnormalize_final.pkl"))
    print("학습 완료. 모델 저장 위치:", MODEL_DIR)


if __name__ == "__main__":
    main()
