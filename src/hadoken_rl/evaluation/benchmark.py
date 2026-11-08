"""Módulo de benchmarking y evaluación estadística de políticas en Street Fighter II."""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import gymnasium as gym
import numpy as np


class StreetFighterBenchmark:
    """
    Ejecutor de benchmarks para evaluar agentes entrenados o baselines.
    Mide Win Rate, Diferencial de HP neto y estadísticas de daño sobre N combates.
    """

    def __init__(
        self,
        env_fn: Callable[[int], gym.Env],
        num_episodes: int = 100,
        difficulties: Optional[List[int]] = None,
        deterministic: bool = True,
    ):
        """
        Args:
            env_fn: Función constructora que recibe la dificultad (int) y retorna el entorno envuelto.
            num_episodes: Número de combates independientes por dificultad (default: 100).
            difficulties: Lista de dificultades a evaluar (default: [1, 4, 7]).
            deterministic: Usar acciones deterministas en políticas probabilísticas (default: True).
        """
        self.env_fn = env_fn
        self.num_episodes = num_episodes
        self.difficulties = difficulties if difficulties is not None else [1, 4, 7]
        self.deterministic = deterministic

    def evaluate_difficulty(self, model: Any, difficulty: int) -> Dict[str, Any]:
        """Ejecuta num_episodes combates en una dificultad específica."""
        env = self.env_fn(difficulty)

        wins = 0
        net_hps = []
        damage_dealt_list = []
        damage_taken_list = []
        episode_lengths = []

        for _ in range(self.num_episodes):
            obs, info = env.reset()
            terminated = False
            truncated = False
            ep_len = 0

            last_info = info

            while not (terminated or truncated):
                # Soporte para modelos de Stable-Baselines3 o agentes aleatorios/heurísticos
                if hasattr(model, "predict"):
                    action, _ = model.predict(obs, deterministic=self.deterministic)
                elif callable(model):
                    action = model(obs)
                else:
                    action = env.action_space.sample()

                obs, _, terminated, truncated, last_info = env.step(action)
                ep_len += 1

            # Extraer métricas terminales inyectadas por StreetFighterRewardWrapper
            if last_info.get("is_win", False):
                wins += 1

            net_hps.append(float(last_info.get("net_hp_differential", 0.0)))
            damage_dealt_list.append(float(last_info.get("damage_dealt", 0.0)))
            damage_taken_list.append(float(last_info.get("damage_taken", 0.0)))
            episode_lengths.append(ep_len)

        env.close()

        win_rate = (wins / self.num_episodes) * 100.0

        return {
            "difficulty": difficulty,
            "episodes": self.num_episodes,
            "wins": wins,
            "win_rate_pct": float(np.round(win_rate, 2)),
            "net_hp_mean": float(np.round(np.mean(net_hps), 2)),
            "net_hp_std": float(np.round(np.std(net_hps), 2)),
            "damage_dealt_mean": float(np.round(np.mean(damage_dealt_list), 2)),
            "damage_taken_mean": float(np.round(np.mean(damage_taken_list), 2)),
            "episode_len_mean": float(np.round(np.mean(episode_lengths), 2)),
        }

    def run_full_benchmark(
        self, model: Any, output_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """Ejecuta el protocolo sobre todas las dificultades configuradas y genera el reporte."""
        results = {}

        print(f"\n{'='*60}")
        print(f"Iniciando Benchmark: {self.num_episodes} combates por dificultad")
        print(f"{'='*60}")

        for diff in self.difficulties:
            print(f"Evaluando Dificultad {diff}...")
            res = self.evaluate_difficulty(model, diff)
            results[f"difficulty_{diff}"] = res

            print(
                f"  -> Win Rate: {res['win_rate_pct']}% | "
                f"HP Neto: {res['net_hp_mean']} ± {res['net_hp_std']} | "
                f"Daño: {res['damage_dealt_mean']} infligido / {res['damage_taken_mean']} recibido"
            )

        print(f"{'='*60}\n")

        if output_path is not None:
            out_file = Path(output_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4)
            print(f"Resultados guardados en: {out_file.resolve()}")

        return results