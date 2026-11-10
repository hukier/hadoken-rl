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

from hadoken_rl.agents.ppo_agent import PPOConfig, build_ppo
from hadoken_rl.agents.rainbow_dqn import RainbowDQN
from hadoken_rl.agents.random_agent import RandomAgent
from hadoken_rl.envs.factory import add_env_args, env_config_from_args, make_env, make_vec_env
from hadoken_rl.evaluation.callbacks import StreetFighterMetricsCallback


def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento para Street Fighter II (PPO / Rainbow DQN / Aleatorio)")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "rainbow", "random", "all"],
                        help="Método a entrenar; 'all' ejecuta los tres con el mismo presupuesto de timesteps")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Presupuesto total de timesteps (igual para cada método)")
    parser.add_argument("--log-dir", type=str, default="logs/tensorboard", help="Directorio de TensorBoard")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directorio para guardar modelos")
    parser.add_argument("--seed", type=int, default=0, help="Semilla (misma para todos los métodos)")
    # PPO
    parser.add_argument("--n-envs", type=int, default=8, help="Entornos paralelos de PPO (SubprocVecEnv)")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate de PPO")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Coeficiente de entropía (PPO)")
    parser.add_argument("--clip-range", type=float, default=0.2, help="Clip range (PPO)")
    parser.add_argument("--n-steps", type=int, default=256, help="Pasos por entorno entre actualizaciones (PPO)")
    parser.add_argument("--ppo-batch-size", type=int, default=256, help="Minibatch de PPO (debe dividir n-steps * n-envs)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Factor de descuento (PPO y Rainbow)")
    # Rainbow DQN
    parser.add_argument("--rainbow-lr", type=float, default=1e-4, help="Learning rate de Rainbow DQN")
    parser.add_argument("--n-step", type=int, default=3, help="Pasos del retorno n-step (Rainbow)")
    parser.add_argument("--buffer-size", type=int, default=100_000, help="Tamaño del replay buffer (Rainbow)")
    parser.add_argument("--batch-size", type=int, default=64, help="Tamaño de lote (Rainbow)")
    parser.add_argument("--learning-starts", type=int, default=2_000, help="Pasos aleatorios antes de aprender (Rainbow)")
    parser.add_argument("--target-update", type=int, default=2_000, help="Intervalo de actualización de la red target (Rainbow)")
    add_env_args(parser)
    return parser.parse_args()


def train_ppo(args, log_path: Path, save_path: Path) -> Path:
    """PPO con `--n-envs` entornos paralelos. Los timesteps se cuentan agregados entre entornos."""
    vec_env = make_vec_env(env_config_from_args(args), args.n_envs)
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
    model = RainbowDQN(
        env,
        learning_rate=args.rainbow_lr,
        gamma=args.gamma,
        n_step=args.n_step,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        learning_starts=args.learning_starts,
        target_update_interval=args.target_update,
        tensorboard_log=str(log_path),
        seed=args.seed,
        verbose=1,
    )
    model.learn(total_timesteps=args.timesteps)

    final_model_file = save_path / "rainbow_street_fighter_final.pt"
    model.save(final_model_file)
    env.close()
    return final_model_file


def train_random(args, log_path: Path, save_path: Path) -> Path:
    """El agente aleatorio no aprende: solo juega el mismo presupuesto para dibujar su curva base."""
    env = make_env(env_config_from_args(args))
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
    log_path.mkdir(parents=True, exist_ok=True)
    save_path.mkdir(parents=True, exist_ok=True)

    algos = list(TRAINERS) if args.algo == "all" else [args.algo]

    print(f"\n[+] Iniciando pipeline de entrenamiento:")
    print(f"    - Métodos: {', '.join(algos)}")
    print(f"    - Timesteps (por método): {args.timesteps}")
    print(f"    - Entorno: {args.env_kind} (dificultad {args.difficulty})")
    print(f"    - Semilla: {args.seed}")
    print(f"    - TensorBoard: {log_path.resolve()}")
    print(f"    - Checkpoints: {save_path.resolve()}")

    for algo in algos:
        print(f"\n--- Entrenando: {algo.upper()} ---")
        out_file = TRAINERS[algo](args, log_path, save_path)
        print(f"[✓] {algo.upper()} completado. Modelo guardado en: {out_file.resolve()}\n")


if __name__ == "__main__":
    main()
