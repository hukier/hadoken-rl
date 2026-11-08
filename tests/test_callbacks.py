"""Prueba unitaria para StreetFighterMetricsCallback."""

import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from hadoken_rl.envs.reward_wrapper import StreetFighterRewardWrapper
from hadoken_rl.evaluation.callbacks import StreetFighterMetricsCallback


class DummyFightEnv(gym.Env):
    """Entorno simulado que dura 5 pasos y luego termina en victoria."""

    def __init__(self):
        super().__init__()
        self.observation_space = Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        self.action_space = Discrete(12)
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        self.steps = 0
        info = {"health": 176, "enemy_health": 176, "x": 50, "enemy_x": 100}
        return np.zeros(12, dtype=np.float32), info

    def step(self, action):
        self.steps += 1
        # Simular que el rival pierde vida hasta K.O. en el paso 5
        enemy_hp = max(0, 176 - (self.steps * 40))
        agent_hp = 160
        terminated = enemy_hp <= 0

        info = {
            "health": agent_hp,
            "enemy_health": enemy_hp,
            "x": 60,
            "enemy_x": 80,
        }
        return np.zeros(12, dtype=np.float32), 0.0, terminated, False, info


def test_callback_logging():
    def make_env():
        base = DummyFightEnv()
        return StreetFighterRewardWrapper(base)

    vec_env = DummyVecEnv([make_env])
    callback = StreetFighterMetricsCallback(window_size=10, log_freq=1)

    # Entrenar un modelo PPO durante 30 pasos para forzar varios finales de episodio
    model = PPO("MlpPolicy", vec_env, n_steps=16, batch_size=16, verbose=0)
    model.learn(total_timesteps=32, callback=callback)

    assert callback.episode_count >= 1, "Debe haberse completado al menos un combate"
    assert len(callback.win_history) >= 1
    assert callback.win_history[-1] == 1.0  # El entorno simulado siempre gana
    print(f"✅ Callback test superado: {callback.episode_count} episodios procesados.")


if __name__ == "__main__":
    test_callback_logging()