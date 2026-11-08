"""Prueba unitaria para StreetFighterRewardWrapper usando un entorno simulado."""

import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import numpy as np

from hadoken_rl.envs.reward_wrapper import StreetFighterRewardWrapper


class MockSFEnv(gym.Env):
    """Entorno falso que simula variaciones de vida y distancia en el dict info."""

    def __init__(self):
        super().__init__()
        self.observation_space = Box(low=0, high=255, shape=(12,), dtype=np.float32)
        self.action_space = Discrete(12)
        self.agent_hp = 176
        self.enemy_hp = 176
        self.step_count = 0

    def reset(self, *, seed=None, options=None):
        self.agent_hp = 176
        self.enemy_hp = 176
        self.step_count = 0
        info = {"health": self.agent_hp, "enemy_health": self.enemy_hp, "x": 50, "enemy_x": 150}
        return np.zeros(12, dtype=np.float32), info

    def step(self, action):
        self.step_count += 1
        # Simula que en el paso 1 el agente golpea al rival (el rival pierde 20 HP)
        if self.step_count == 1:
            self.enemy_hp -= 20
        # Simula que en el paso 2 ambos caen, pero el rival llega a 0 (K.O.)
        elif self.step_count == 2:
            self.enemy_hp = 0

        terminated = self.enemy_hp <= 0
        info = {"health": self.agent_hp, "enemy_health": self.enemy_hp, "x": 100, "enemy_x": 110}
        return np.zeros(12, dtype=np.float32), 0.0, terminated, False, info


def test_reward_shaping():
    env = MockSFEnv()
    wrapped_env = StreetFighterRewardWrapper(env, alpha=1.0, beta=1.0, win_reward=50.0)

    _, info = wrapped_env.reset()
    assert info["health"] == 176

    # Paso 1: Rival pierde 20 HP -> Recompensa debe ser positiva (+20 aprox)
    _, reward, term, _, info = wrapped_env.step(0)
    assert reward > 15.0, f"Recompensa esperada ~20, obtenida: {reward}"
    assert not term

    # Paso 2: Rival llega a 0 HP -> K.O. y Victoria (+50 de bono)
    _, reward, term, _, info = wrapped_env.step(0)
    assert term
    assert info["is_win"] is True
    assert reward >= 50.0
    print("✅ Test de RewardWrapper superado con éxito.")


if __name__ == "__main__":
    test_reward_shaping()