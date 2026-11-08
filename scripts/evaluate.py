"""Benchmark comparativo: PPO vs Rainbow DQN vs agente aleatorio.

Evalúa cada agente con el mismo protocolo (mismas dificultades y número de combates)
y genera en `--output-dir` la tabla comparativa (JSON, CSV, Markdown) y un gráfico.

Ejemplos:
    python scripts/evaluate.py --ppo-path checkpoints/ppo_street_fighter_final.zip \\
                               --rainbow-path checkpoints/rainbow_street_fighter_final.pt
    python scripts/evaluate.py --rainbow-path checkpoints/rainbow_street_fighter_final.pt --no-random
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import argparse
from stable_baselines3 import PPO

from hadoken_rl.agents.rainbow_dqn import RainbowDQN
from hadoken_rl.agents.random_agent import RandomAgent
from hadoken_rl.envs.reward_wrapper import StreetFighterRewardWrapper
from hadoken_rl.evaluation.benchmark import StreetFighterBenchmark


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluación comparativa de agentes en SF2")
    parser.add_argument("--ppo-path", type=str, default=None, help="Ruta al .zip del modelo PPO")
    parser.add_argument("--model-path", type=str, default=None, help="(Obsoleto) alias de --ppo-path")
    parser.add_argument("--rainbow-path", type=str, default=None, help="Ruta al .pt del modelo Rainbow DQN")
    parser.add_argument("--no-random", action="store_true", help="Excluir el agente aleatorio de la comparación")
    parser.add_argument("--episodes", type=int, default=100, help="Combates por dificultad")
    parser.add_argument("--seed", type=int, default=0, help="Semilla del agente aleatorio")
    parser.add_argument("--output-dir", type=str, default="reports", help="Directorio de salida del reporte comparativo")
    args = parser.parse_args()
    args.ppo_path = args.ppo_path or args.model_path
    return args


def env_factory(difficulty: int):
    # Conecta con el entorno base configurado por dificultad
    from tests.test_benchmark import MockCombatEnv
    base_env = MockCombatEnv(difficulty=difficulty)
    return StreetFighterRewardWrapper(base_env)


def load_checkpoint(path_str: str, label: str) -> Path:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el modelo {label} en: {path.resolve()}")
    print(f"[+] Cargando {label} desde: {path.resolve()}")
    return path


def main():
    args = parse_args()

    models = {}
    if args.ppo_path:
        models["PPO"] = PPO.load(str(load_checkpoint(args.ppo_path, "PPO")))
    if args.rainbow_path:
        models["Rainbow DQN"] = RainbowDQN.load(load_checkpoint(args.rainbow_path, "Rainbow DQN"))
    if not args.no_random:
        probe_env = env_factory(1)
        models["Aleatorio"] = RandomAgent(probe_env.action_space, probe_env.observation_space, seed=args.seed)
        probe_env.close()

    if len(models) < 2:
        raise SystemExit(
            "Se necesitan al menos 2 agentes para comparar. Indica --ppo-path y/o --rainbow-path "
            "(el agente aleatorio se incluye por defecto)."
        )

    benchmark = StreetFighterBenchmark(
        env_fn=env_factory,
        num_episodes=args.episodes,
        difficulties=[1, 4, 7],
    )
    benchmark.compare_models(models, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
