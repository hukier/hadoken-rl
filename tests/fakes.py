"""Entorno falso con la misma interfaz que el base env de Stable-Retro (sin ROM)."""

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, MultiBinary

from hadoken_rl.envs.action_wrapper import GENESIS_BUTTONS

RAM_BASE = 0xFF0000
# Direcciones sintéticas para las pruebas (NO son las de la ROM real).
ADDR = {"agent_x": 0xFF1000, "agent_y": 0xFF1002, "enemy_x": 0xFF1004, "enemy_y": 0xFF1006,
        "timer": 0xFF1008, "agent_anim": 0xFF1009, "enemy_anim": 0xFF100A}


class FakeRetroEnv(gym.Env):
    """RAM de 65.536 bytes, 12 botones y `info` con salud y coordenadas. Registra las acciones recibidas."""

    buttons = list(GENESIS_BUTTONS)

    def __init__(self, episode_len=20, agent_x=100, enemy_x=300):
        super().__init__()
        self.observation_space = Box(0, 255, shape=(65536,), dtype=np.uint8)
        self.action_space = MultiBinary(12)
        self.episode_len = episode_len
        self.agent_x, self.enemy_x = agent_x, enemy_x
        self.t = 0
        self.received = []

    def _ram(self):
        ram = np.zeros(65536, dtype=np.uint8)
        def put(addr, value, size):
            ram[addr - RAM_BASE: addr - RAM_BASE + size] = np.frombuffer(int(value).to_bytes(size, "big"), dtype=np.uint8)
        put(ADDR["agent_x"], self.agent_x, 2); put(ADDR["agent_y"], 0, 2)
        put(ADDR["enemy_x"], self.enemy_x, 2); put(ADDR["enemy_y"], 0, 2)
        put(ADDR["timer"], 99 - self.t, 1); put(ADDR["agent_anim"], 7, 1); put(ADDR["enemy_anim"], 3, 1)
        return ram

    def _info(self):
        return {"health": 176 - self.t, "enemy_health": 176 - 2 * self.t, "x": self.agent_x, "enemy_x": self.enemy_x}

    def reset(self, *, seed=None, options=None):
        self.t = 0
        return self._ram(), self._info()

    def step(self, action):
        self.received.append(np.array(action))
        self.t += 1
        done = self.t >= self.episode_len
        return self._ram(), 0.0, done, False, self._info()
