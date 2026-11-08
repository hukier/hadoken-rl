"""Script para ejecutar benchmark cuantitativo sobre modelos guardados."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import argparse
from pathlib import Path
from stable_baselines3 import PPO

from hadoken_rl.envs.reward_wrapper import StreetFighterRewardWrapper
from hadoken_rl.evaluation.benchmark import StreetFighterBenchmark


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluación de modelos en SF2")
    parser.add_argument("--model-path", type=str, required=True, help="Ruta al archivo .zip del modelo")
    parser.add_argument("--episodes", type=int, default=100, help="Combates por dificultad")
    parser.add_argument("--output", type=str, default="reports/benchmark_results.json", help="Ruta del JSON final")
    return parser.parse_args()


def env_factory(difficulty: int):
    # Conecta con el entorno base configurado por dificultad
    from tests.test_benchmark import MockCombatEnv
    base_env = MockCombatEnv(difficulty=difficulty)
    return StreetFighterRewardWrapper(base_env)


def main():
    args = parse_args()
    model_path = Path(args.model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo en: {model_path.resolve()}")

    print(f"[+] Cargando política desde: {model_path.resolve()}")
    model = PPO.load(str(model_path))

    benchmark = StreetFighterBenchmark(
        env_fn=env_factory,
        num_episodes=args.episodes,
        difficulties=[1, 4, 7],
    )

    benchmark.run_full_benchmark(model, output_path=args.output)


if __name__ == "__main__":
    main()