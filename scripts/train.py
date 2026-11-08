"""Script principal para entrenar el agente con PPO y registrar métricas en TensorBoard."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import argparse
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from hadoken_rl.envs.reward_wrapper import StreetFighterRewardWrapper
from hadoken_rl.evaluation.callbacks import StreetFighterMetricsCallback


def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento PPO para Street Fighter II")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Presupuesto total de timesteps")
    parser.add_argument("--log-dir", type=str, default="logs/tensorboard", help="Directorio de TensorBoard")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directorio para guardar modelos")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate de PPO")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Coeficiente de entropía")
    return parser.parse_args()


def make_env():
    """
    Construye el pipeline de wrappers.
    Cuando Martín y Alan suban sus módulos, se reemplaza la base por:
        from hadoken_rl.envs.base_env import make_base_sf_env
        from hadoken_rl.envs.action_wrapper import StreetFighterActionWrapper
        from hadoken_rl.envs.observation_wrapper import StreetFighterRAMWrapper

        env = make_base_sf_env(...)
        env = StreetFighterActionWrapper(env)
        env = StreetFighterRewardWrapper(env)
        env = StreetFighterRAMWrapper(env)
    """
    # Placeholder temporal para pruebas:
    from tests.test_callbacks import DummyFightEnv
    env = DummyFightEnv()
    env = StreetFighterRewardWrapper(env)
    return env


def main():
    args = parse_args()
    log_path = Path(args.log_dir)
    save_path = Path(args.save_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"\n[+] Iniciando pipeline de entrenamiento:")
    print(f"    - Timesteps: {args.timesteps}")
    print(f"    - TensorBoard: {log_path.resolve()}")
    print(f"    - Checkpoints: {save_path.resolve()}")

    vec_env = DummyVecEnv([make_env])
    metrics_callback = StreetFighterMetricsCallback(window_size=50, log_freq=1)

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=args.learning_rate,
        ent_coef=args.ent_coef,
        tensorboard_log=str(log_path),
        verbose=1,
    )

    model.learn(total_timesteps=args.timesteps, callback=metrics_callback)

    final_model_file = save_path / "ppo_street_fighter_final.zip"
    model.save(str(final_model_file))
    print(f"\n[✓] Entrenamiento completado. Modelo guardado en: {final_model_file.resolve()}\n")


if __name__ == "__main__":
    main()