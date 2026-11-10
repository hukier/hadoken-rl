"""Pruebas de StreetFighterRAMWrapper y de las utilidades de ram_tools."""

import numpy as np
import pytest

from hadoken_rl.envs.observation_wrapper import (
    RAMFeature, StreetFighterRAMWrapper, default_features, load_layout, save_layout,
)
from hadoken_rl.envs.ram_tools import match_rate, observed_stats, search_candidates
from tests.fakes import ADDR, FakeRetroEnv


def _features():
    feats = default_features()
    mapping = {"agent_x": "agent_x", "agent_y": "agent_y", "enemy_x": "enemy_x", "enemy_y": "enemy_y",
               "timer": "timer", "agent_animation": "agent_anim", "enemy_animation": "enemy_anim"}
    for f in feats:
        if f.name in mapping:
            f.address = ADDR[mapping[f.name]]
    return feats


def test_vector_shape_range_and_values():
    w = StreetFighterRAMWrapper(FakeRetroEnv(agent_x=100, enemy_x=300), features=_features())
    obs, _ = w.reset()
    assert obs.shape == (12,) and obs.dtype == np.float32
    assert w.observation_space.shape == (12,)
    assert obs.min() >= 0.0 and obs.max() <= 1.0
    named = dict(zip(w.feature_names, obs))
    assert np.isclose(named["agent_x"], 100 / 1024)
    assert np.isclose(named["enemy_x"], 300 / 1024)
    assert np.isclose(named["agent_health"], 1.0)           # desde info
    assert np.isclose(named["timer"], 99 / 99)
    assert np.isclose(named["dx"], (200 + 1024) / 2048)     # diferencia normalizada


def test_step_updates_from_info_and_ram():
    w = StreetFighterRAMWrapper(FakeRetroEnv(), features=_features())
    w.reset()
    obs, *_ = w.step(np.zeros(12, dtype=np.int8))
    named = dict(zip(w.feature_names, obs))
    assert np.isclose(named["agent_health"], 175 / 176)
    assert np.isclose(named["enemy_health"], 174 / 176)
    assert np.isclose(named["timer"], 98 / 99)


def test_refuses_unset_addresses_unless_allowed():
    with pytest.raises(ValueError, match="Faltan direcciones RAM"):
        StreetFighterRAMWrapper(FakeRetroEnv())  # default_features sin direcciones
    w = StreetFighterRAMWrapper(FakeRetroEnv(), allow_unset=True)
    obs, _ = w.reset()
    assert obs.shape == (12,)


def test_out_of_range_address_and_non_ram_env():
    feats = [RAMFeature("bad", address=0xFF0000 + 70000, size=1)] + [RAMFeature(f"f{i}", source="info", info_key="health", max_value=176) for i in range(11)]
    with pytest.raises(ValueError, match="fuera de la RAM"):
        StreetFighterRAMWrapper(FakeRetroEnv(), features=feats)

    import gymnasium as gym
    class ImageEnv(gym.Env):
        observation_space = gym.spaces.Box(0, 255, (84, 84, 3), np.uint8)
        action_space = gym.spaces.Discrete(2)
    with pytest.raises(ValueError, match="vector 1-D"):
        StreetFighterRAMWrapper(ImageEnv(), allow_unset=True)


def test_missing_info_key_uses_default_with_warning():
    feats = _features()
    feats[4].info_key = "no_existe"
    w = StreetFighterRAMWrapper(FakeRetroEnv(), features=feats)
    with pytest.warns(UserWarning, match="no_existe"):
        obs, _ = w.reset()
    assert np.isclose(dict(zip(w.feature_names, obs))["agent_health"], 1.0)  # default=176 -> 1.0


def test_layout_roundtrip(tmp_path):
    path = tmp_path / "layout.json"
    save_layout(_features(), path)
    layout = load_layout(path)
    assert [f.name for f in layout["features"]] == [f.name for f in _features()]
    w = StreetFighterRAMWrapper(FakeRetroEnv(), layout_path=path)
    obs, _ = w.reset()
    assert obs.shape == (12,)


def test_repo_layout_file_is_loadable():
    from hadoken_rl.envs.factory import DEFAULT_LAYOUT_PATH
    layout = load_layout(DEFAULT_LAYOUT_PATH)
    assert 12 <= len(layout["features"]) <= 20


def test_ram_tools_find_and_validate_health_address():
    rng = np.random.default_rng(0)
    health = np.clip(176 - np.cumsum(rng.integers(0, 6, 200)), 0, 176)
    ram = rng.integers(0, 255, (200, 64), dtype=np.uint8)
    ram[:, 17] = health
    cands = search_candidates(ram, health)
    assert 17 in cands
    assert match_rate(ram, 17, health) == 1.0
    stats = observed_stats(ram, 17)
    assert stats["max"] == health.max() and stats["min"] == health.min()
