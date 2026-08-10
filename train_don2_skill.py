"""
don2 범용 스킬 학습기 — 발달 계보(워름스타트) 지원.

사용:
  python train_don2_skill.py --skill stand --steps 1000000
  python train_don2_skill.py --skill walk --steps 3000000 --init-from stand

--init-from이 주어지면 부모 스킬의 PPO 가중치 + VecNormalize 통계를 함께 승계한다
(통계를 빼먹으면 관측 스케일이 어긋나 워름스타트 효과가 무너짐 — DESIGN.md 규칙).
"""
import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from don2_env import Don2Env
from framework.skill_registry import SKILLS, log_dir, model_dir

N_ENVS = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skill", required=True, choices=list(SKILLS))
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--init-from", default=None, choices=list(SKILLS),
                    help="워름스타트 부모 스킬 (가중치+정규화 통계 승계)")
    ap.add_argument("--ent", type=float, default=None,
                    help="엔트로피 계수 재정의 (탐험 강화 이어학습용, 예: 0.01)")
    ap.add_argument("--std-boost", type=float, default=0.0,
                    help="정책 log_std 가산 (예: 0.5 = 탐험 노이즈 확대) — --init-from과 함께")
    args = ap.parse_args()

    cfg = SKILLS[args.skill]
    mdir, ldir = model_dir(args.skill), log_dir(args.skill)
    os.makedirs(mdir, exist_ok=True)
    os.makedirs(ldir, exist_ok=True)

    vec = make_vec_env(Don2Env, n_envs=N_ENVS, env_kwargs=cfg["env_kwargs"])

    if args.init_from:
        parent_dir = model_dir(args.init_from)
        vec = VecNormalize.load(os.path.join(parent_dir, "vecnormalize_final.pkl"), vec)
        vec.training = True
        vec.norm_reward = True
        load_kw = dict(env=vec, tensorboard_log=ldir)
        if args.ent is not None:
            load_kw["ent_coef"] = args.ent
        model = PPO.load(os.path.join(parent_dir, "ppo_final.zip"), **load_kw)
        if args.std_boost:
            import torch
            with torch.no_grad():
                model.policy.log_std += args.std_boost
            print(f"[std-boost] log_std += {args.std_boost}", flush=True)
        print(f"[warm-start] {args.skill} <- {args.init_from}"
              + (f" (ent={args.ent})" if args.ent is not None else ""), flush=True)
    else:
        vec = VecNormalize(vec, norm_obs=True, norm_reward=True, clip_obs=10.0)
        model = PPO("MlpPolicy", vec, verbose=1,
                    n_steps=512, batch_size=1024, n_epochs=10,
                    learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, ent_coef=0.0,
                    tensorboard_log=ldir)

    cb = CheckpointCallback(save_freq=max(100_000 // N_ENVS, 1), save_path=mdir,
                            name_prefix="ppo", save_vecnormalize=True)
    model.learn(total_timesteps=args.steps, callback=cb, progress_bar=False,
                reset_num_timesteps=True)

    model.save(os.path.join(mdir, "ppo_final"))
    vec.save(os.path.join(mdir, "vecnormalize_final.pkl"))
    print(f"=== SKILL {args.skill} DONE === -> {mdir}", flush=True)


if __name__ == "__main__":
    main()
