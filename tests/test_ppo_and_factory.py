"""Pruebas de PPOConfig/build_ppo y de la fábrica de entornos."""

import numpy as np
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv

import hadoken_rl.envs.factory as factory
from hadoken_rl.agents.ppo_agent import PPOConfig, build_ppo
from hadoken_rl.envs.factory import EnvConfig, make_env, make_vec_env
from hadoken_rl.envs.observation_wrapper import default_features, save_layout
from tests.fakes import ADDR, FakeRetroEnv


def test_ppo_config_defaults_fit_8_envs():
    cfg = PPOConfig()
    cfg.validate()
    assert cfg.n_envs == 8 and cfg.rollout_size == 2048 and cfg.rollout_size % cfg.batch_size == 0


def test_ppo_config_rejects_bad_batch():
    with pytest.raises(ValueError):
        PPOConfig(batch_size=300).validate()


def test_build_ppo_on_8_mock_envs_and_learn():
    vec_env = make_vec_env(EnvConfig(kind="mock"), 8)
    assert vec_env.num_envs == 8
    model = build_ppo(vec_env, PPOConfig(n_steps=32, batch_size=64), verbose=0, seed=0)
    assert [m.out_features for m in model.policy.mlp_extractor.policy_net if hasattr(m, "out_features")] == [256, 256]
    model.learn(total_timesteps=512)
    vec_env.close()


@pytest.fixture
def real_stack(monkeypatch, tmp_path):
    """Pila real de wrappers sobre un base env falso (sin ROM)."""
    feats = default_features()
    names = {"agent_x": "agent_x", "agent_y": "agent_y", "enemy_x": "enemy_x", "enemy_y": "enemy_y",
             "timer": "timer", "agent_animation": "agent_anim", "enemy_animation": "enemy_anim"}
    for f in feats:
        if f.name in names:
            f.address = ADDR[names[f.name]]
    layout = tmp_path / "layout.json"
    save_layout(feats, layout)
    monkeypatch.setattr(factory, "_build_base_env", lambda cfg: FakeRetroEnv(episode_len=15))
    return EnvConfig(kind="real", layout_path=layout)


def test_full_wrapper_stack_contract(real_stack):
    env = make_env(real_stack)
    assert env.action_space.n == 14
    assert env.observation_space.shape == (12,)
    obs, info = env.reset()
    done, steps = False, 0
    while not done:
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        done = terminated or truncated
        steps += 1
        assert obs.shape == (12,) and np.all((obs >= 0) & (obs <= 1))
    # El reward wrapper publica las métricas que consumen el callback y el benchmark
    assert {"is_win", "net_hp_differential", "damage_dealt", "damage_taken"} <= set(info)
    assert info["is_win"] is True


def test_ppo_trains_on_full_stack(real_stack):
    vec_env = make_vec_env(real_stack, 1)
    assert isinstance(vec_env, DummyVecEnv)
    model = build_ppo(vec_env, PPOConfig(n_envs=1, n_steps=64, batch_size=32), verbose=0, seed=0)
    model.learn(total_timesteps=128)


def test_invalid_kind_raises():
    with pytest.raises(ValueError):
        make_env(EnvConfig(kind="otro"))


def test_real_vec_env_fails_fast_with_clear_message(monkeypatch):
    def boom(cfg):
        raise ImportError("No se encontró hadoken_rl.envs.base_env ni stable-retro")
    monkeypatch.setattr(factory, "_build_base_env", boom)
    with pytest.raises(ImportError, match="stable-retro"):
        make_vec_env(EnvConfig(kind="real"), 8)
