"""Prueba unitaria para el módulo de benchmarking."""

import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import numpy as np

from hadoken_rl.envs.reward_wrapper import StreetFighterRewardWrapper
from hadoken_rl.evaluation.benchmark import StreetFighterBenchmark


class MockCombatEnv(gym.Env):
    """Entorno simulado que varía su dificultad."""

    def __init__(self, difficulty: int):
        super().__init__()
        self.difficulty = difficulty
        self.observation_space = Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        self.action_space = Discrete(12)
        self.step_count = 0

    def reset(self, *, seed=None, options=None):
        self.step_count = 0
        info = {"health": 176, "enemy_health": 176, "x": 40, "enemy_x": 120}
        return np.zeros(12, dtype=np.float32), info

    def step(self, action):
        self.step_count += 1

        # En dificultad 1 el rival pierde rápido (gana el agente)
        # En dificultad 7 el agente pierde rápido
        if self.difficulty == 1:
            enemy_hp = max(0, 176 - (self.step_count * 90))
            agent_hp = 160
        else:
            enemy_hp = 150
            agent_hp = max(0, 176 - (self.step_count * 90))

        terminated = (enemy_hp <= 0) or (agent_hp <= 0)

        info = {
            "health": agent_hp,
            "enemy_health": enemy_hp,
            "x": 60,
            "enemy_x": 90,
        }
        return np.zeros(12, dtype=np.float32), 0.0, terminated, False, info


def test_benchmark_execution(tmp_path):
    def env_factory(diff: int):
        base = MockCombatEnv(difficulty=diff)
        return StreetFighterRewardWrapper(base)

    # Correr un benchmark de prueba con 5 episodios en dificultades [1, 7]
    benchmark = StreetFighterBenchmark(
        env_fn=env_factory,
        num_episodes=5,
        difficulties=[1, 7],
    )

    class DummyPolicy:
        def predict(self, obs, deterministic=True):
            return 0, None

    output_json = tmp_path / "benchmark_results.json"
    results = benchmark.run_full_benchmark(DummyPolicy(), output_path=output_json)

    # Validaciones estadísticas
    assert "difficulty_1" in results
    assert "difficulty_7" in results
    assert results["difficulty_1"]["win_rate_pct"] == 100.0
    assert results["difficulty_7"]["win_rate_pct"] == 0.0
    assert output_json.exists(), "El archivo de resultados JSON debe haberse generado"


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_benchmark_execution(Path(tmp_dir))
        print("✅ Test de Benchmark superado exitosamente.")