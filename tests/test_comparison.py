"""Pruebas del reporte comparativo entre PPO, Rainbow DQN y agente aleatorio."""

from hadoken_rl.agents.random_agent import RandomAgent
from hadoken_rl.envs.reward_wrapper import StreetFighterRewardWrapper
from hadoken_rl.evaluation.benchmark import StreetFighterBenchmark
from hadoken_rl.evaluation.comparison import build_rows, format_markdown_table
from tests.test_benchmark import MockCombatEnv


class AlwaysZero:
    def predict(self, obs, deterministic=True):
        return 0, None


def _env_factory(diff):
    return StreetFighterRewardWrapper(MockCombatEnv(difficulty=diff))


def test_compare_three_agents_generates_report(tmp_path):
    probe = _env_factory(1)
    models = {
        "PPO": AlwaysZero(),
        "Rainbow DQN": AlwaysZero(),
        "Aleatorio": RandomAgent(probe.action_space, probe.observation_space, seed=0),
    }
    benchmark = StreetFighterBenchmark(_env_factory, num_episodes=3, difficulties=[1, 7])
    results = benchmark.compare_models(models, output_dir=tmp_path)

    assert set(results) == {"PPO", "Rainbow DQN", "Aleatorio"}
    assert results["PPO"]["difficulty_1"]["win_rate_pct"] == 100.0
    assert results["Rainbow DQN"]["difficulty_7"]["win_rate_pct"] == 0.0

    for name in ("comparison_results.json", "comparison_results.csv", "comparison_table.md"):
        assert (tmp_path / name).exists(), f"Falta {name}"

    rows = build_rows(results)
    assert len(rows) == 6  # 3 agentes x 2 dificultades
    table = format_markdown_table(results)
    assert "Rainbow DQN" in table and "Aleatorio" in table and "PPO" in table
