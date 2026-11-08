"""Pruebas unitarias para RainbowDQN, el replay priorizado y el agente aleatorio."""

import numpy as np
import torch

from hadoken_rl.agents.rainbow_dqn import PrioritizedReplayBuffer, RainbowDQN, NoisyLinear
from hadoken_rl.agents.random_agent import RandomAgent
from hadoken_rl.envs.reward_wrapper import StreetFighterRewardWrapper
from tests.test_callbacks import DummyFightEnv


def _make_env():
    return StreetFighterRewardWrapper(DummyFightEnv())


def test_noisy_linear_eval_is_deterministic():
    layer = NoisyLinear(4, 3)
    x = torch.randn(2, 4)
    layer.eval()
    assert torch.allclose(layer(x), layer(x))
    layer.train()
    out1 = layer(x)
    layer.reset_noise()
    out2 = layer(x)
    assert not torch.allclose(out1, out2), "El ruido debe cambiar la salida en modo train"


def test_replay_buffer_prioritized_sampling():
    buf = PrioritizedReplayBuffer(capacity=8, obs_dim=2, alpha=1.0)
    for i in range(8):
        buf.add(np.full(2, i), i % 3, float(i), np.full(2, i + 1), False, 1)
    assert len(buf) == 8

    # Subir mucho la prioridad de un solo elemento: debe dominar el muestreo.
    buf.update_priorities(np.arange(8), np.array([0.0] * 7 + [100.0]))
    rng = np.random.default_rng(0)
    batch = buf.sample(64, beta=0.4, rng=rng)
    assert (batch["idx"] == 7).mean() > 0.8
    assert batch["weights"].max() <= 1.0 + 1e-6

    # Capacidad circular
    buf.add(np.zeros(2), 0, 0.0, np.zeros(2), True, 1)
    assert len(buf) == 8


def test_nstep_return_and_truncation_bootstrap():
    agent = RainbowDQN(_make_env(), n_step=3, gamma=0.5, buffer_size=32, verbose=0)
    from collections import deque
    buf = deque([
        (np.zeros(12), 1, 1.0, np.ones(12), False),
        (np.ones(12), 2, 2.0, np.ones(12) * 2, False),
        (np.ones(12) * 2, 3, 4.0, np.ones(12) * 3, True),
    ])
    obs0, act0, ret, next_obs, term, k = agent._make_nstep(buf)
    assert act0 == 1 and k == 3 and term is True
    assert np.isclose(ret, 1.0 + 0.5 * 2.0 + 0.25 * 4.0)
    assert np.allclose(next_obs, 3.0)


def test_predict_interface_single_and_batch():
    agent = RainbowDQN(_make_env(), verbose=0)
    action, state = agent.predict(np.zeros(12, dtype=np.float32))
    assert state is None and 0 <= action < 12

    actions, _ = agent.predict(np.zeros((5, 12), dtype=np.float32))
    assert actions.shape == (5,)

    # Determinista: misma acción repetida
    a1, _ = agent.predict(np.zeros(12, dtype=np.float32), deterministic=True)
    a2, _ = agent.predict(np.zeros(12, dtype=np.float32), deterministic=True)
    assert a1 == a2


def test_learn_logs_metrics_and_save_load(tmp_path):
    env = _make_env()
    agent = RainbowDQN(
        env, learning_starts=50, batch_size=16, buffer_size=500, train_freq=2,
        target_update_interval=50, n_step=3, tensorboard_log=str(tmp_path / "tb"),
        seed=0, verbose=0,
    )
    agent.learn(total_timesteps=300)
    assert agent.num_timesteps == 300
    assert len(agent.replay) > 0

    # Debe haber creado una corrida Rainbow_1 con eventos de TensorBoard
    run_dir = tmp_path / "tb" / "Rainbow_1"
    assert run_dir.exists() and any(run_dir.iterdir())

    ckpt = tmp_path / "rainbow.pt"
    agent.save(ckpt)
    loaded = RainbowDQN.load(ckpt)
    obs = np.random.default_rng(1).random(12).astype(np.float32)
    assert agent.predict(obs)[0] == loaded.predict(obs)[0]


def test_random_agent_interface_and_run(tmp_path):
    env = _make_env()
    agent = RandomAgent(env.action_space, env.observation_space, seed=0)
    action, state = agent.predict(np.zeros(12))
    assert state is None and 0 <= action < 12
    batch_actions, _ = agent.predict(np.zeros((4, 12)))
    assert batch_actions.shape == (4,)

    agent.run(env, total_timesteps=60, tensorboard_log=str(tmp_path / "tb"), verbose=0)
    assert (tmp_path / "tb" / "Random_1").exists()
