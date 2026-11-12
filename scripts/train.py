"""Script principal de entrenamiento: PPO, Rainbow DQN y baseline aleatorio.

Los tres métodos registran en TensorBoard los mismos tags (`combat/*`, `rollout/*`)
bajo un mismo directorio de logs (PPO_1, Rainbow_1, Random_1), de modo que sus curvas
se superponen en un único `tensorboard --logdir logs/tensorboard`.

Ejemplos:
    python scripts/train.py --algo ppo --timesteps 100000
    python scripts/train.py --algo rainbow --timesteps 100000
    python scripts/train.py --algo all --timesteps 100000   # mismo presupuesto para los 3
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import argparse

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3.common.vec_env import VecVideoRecorder

from hadoken_rl.agents.ppo_agent import PPOConfig, build_ppo
from hadoken_rl.agents.rainbow_dqn import RainbowDQN
from hadoken_rl.agents.random_agent import RandomAgent
from hadoken_rl.envs.factory import add_env_args, env_config_from_args, make_env, make_vec_env
from hadoken_rl.evaluation.callbacks import StreetFighterMetricsCallback


def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento para Street Fighter II")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "rainbow", "random", "all"])
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--log-dir", type=str, default="logs/tensorboard")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--video-dir", type=str, default="videos")
    parser.add_argument("--seed", type=int, default=0)
    # PPO
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--ppo-batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    # Rainbow DQN
    parser.add_argument("--rainbow-lr", type=float, default=1e-4)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-starts", type=int, default=2_000)
    parser.add_argument("--target-update", type=int, default=2_000)
    add_env_args(parser)
    return parser.parse_args()


def make_episode_trigger(total_timesteps, frames_per_ep=3000):
    # Para entornos 1-D (RecordVideo cuenta por episodios)
    last_ep = max(1, total_timesteps // frames_per_ep)
    return lambda ep: ep == 0 or ep >= last_ep

def make_step_trigger(total_timesteps, frames_per_ep=3000):
    # Para entornos vectorizados (VecVideoRecorder cuenta por steps globales)
    last_step = max(0, total_timesteps - frames_per_ep)
    return lambda step: step == 0 or step >= last_step


def train_ppo(args, log_path: Path, save_path: Path) -> Path:
    vec_env = make_vec_env(env_config_from_args(args), args.n_envs)
    
    video_path = str(Path(args.video_dir) / "ppo")
    vec_env = VecVideoRecorder(
        vec_env, 
        video_folder=video_path,
        record_video_trigger=make_step_trigger(args.timesteps),
        video_length=3000
    )

    metrics_callback = StreetFighterMetricsCallback(window_size=50, log_freq=1)
    config = PPOConfig(
        n_envs=args.n_envs, n_steps=args.n_steps, batch_size=args.ppo_batch_size,
        learning_rate=args.learning_rate, gamma=args.gamma, clip_range=args.clip_range,
        ent_coef=args.ent_coef,
    )
    model = build_ppo(vec_env, config, tensorboard_log=str(log_path), seed=args.seed)
    model.learn(total_timesteps=args.timesteps, callback=metrics_callback)

    final_model_file = save_path / "ppo_street_fighter_final.zip"
    model.save(str(final_model_file))
    vec_env.close()
    return final_model_file


def train_rainbow(args, log_path: Path, save_path: Path) -> Path:
    env = make_env(env_config_from_args(args))
    
    video_path = str(Path(args.video_dir) / "rainbow")
    env = RecordVideo(env, video_folder=video_path, episode_trigger=make_episode_trigger(args.timesteps))

    model = RainbowDQN(
        env, learning_rate=args.rainbow_lr, gamma=args.gamma, n_step=args.n_step,
        buffer_size=args.buffer_size, batch_size=args.batch_size, learning_starts=args.learning_starts,
        target_update_interval=args.target_update, tensorboard_log=str(log_path), seed=args.seed, verbose=1,
    )
    model.learn(total_timesteps=args.timesteps)

    final_model_file = save_path / "rainbow_street_fighter_final.pt"
    model.save(final_model_file)
    env.close()
    return final_model_file


def train_random(args, log_path: Path, save_path: Path) -> Path:
    env = make_env(env_config_from_args(args))
    
    video_path = str(Path(args.video_dir) / "random")
    env = RecordVideo(env, video_folder=video_path, episode_trigger=make_episode_trigger(args.timesteps))

    agent = RandomAgent(env.action_space, env.observation_space, seed=args.seed)
    agent.run(env, total_timesteps=args.timesteps, tensorboard_log=str(log_path), seed=args.seed)

    final_model_file = save_path / "random_agent.txt"
    agent.save(final_model_file)
    env.close()
    return final_model_file


TRAINERS = {"ppo": train_ppo, "rainbow": train_rainbow, "random": train_random}


def main():
    args = parse_args()
    log_path = Path(args.log_dir)
    save_path = Path(args.save_dir)
    video_path = Path(args.video_dir)
    
    log_path.mkdir(parents=True, exist_ok=True)
    save_path.mkdir(parents=True, exist_ok=True)
    video_path.mkdir(parents=True, exist_ok=True)

    algos = list(TRAINERS) if args.algo == "all" else [args.algo]

    print(f"\n[+] Iniciando pipeline de entrenamiento:")
    print(f"    - Métodos: {', '.join(algos)}")
    print(f"    - Timesteps: {args.timesteps}")
    print(f"    - Entorno: {args.env_kind} (dificultad {args.difficulty})")
    print(f"    - Allow Unset RAM: {args.allow_unset_ram}")
    
    for algo in algos:
        print(f"\n--- Entrenando: {algo.upper()} ---")
        out_file = TRAINERS[algo](args, log_path, save_path)
        print(f"[✓] {algo.upper()} completado. Modelo guardado en: {out_file.resolve()}\n")


if __name__ == "__main__":
    main()