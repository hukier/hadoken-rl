"""Configuración y entrenamiento de agentes RL: PPO (SB3), Rainbow DQN y baseline aleatorio."""

from hadoken_rl.agents.rainbow_dqn import RainbowDQN
from hadoken_rl.agents.random_agent import RandomAgent

__all__ = ["RainbowDQN", "RandomAgent"]
