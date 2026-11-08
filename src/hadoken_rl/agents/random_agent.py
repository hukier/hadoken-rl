"""Agente aleatorio: piso de referencia para comparar PPO y Rainbow DQN."""

from collections import deque
from pathlib import Path
from typing import Any, Deque, Optional, Tuple, Union

import gymnasium as gym
import numpy as np

from hadoken_rl.agents._logging import next_run_dir
from hadoken_rl.evaluation.callbacks import CombatMetricsTracker


class RandomAgent:
    """
    Política que elige acciones uniformemente al azar del espacio de acciones.

    Expone la misma interfaz mínima que los modelos de Stable-Baselines3
    (`predict`, `save`, `load`), por lo que `StreetFighterBenchmark` lo evalúa
    sin ningún caso especial.
    """

    def __init__(
        self,
        action_space: gym.spaces.Space,
        observation_space: Optional[gym.spaces.Space] = None,
        seed: Optional[int] = None,
    ):
        self.action_space = action_space
        self.observation_space = observation_space
        if seed is not None:
            self.action_space.seed(seed)

    def predict(self, obs: Any, deterministic: bool = True) -> Tuple[Any, None]:
        """Devuelve una acción aleatoria (`deterministic` se ignora por diseño)."""
        if self.observation_space is not None and self.observation_space.shape:
            obs_arr = np.asarray(obs)
            if obs_arr.ndim > len(self.observation_space.shape):
                n = obs_arr.shape[0]
                return np.array([self.action_space.sample() for _ in range(n)]), None
        return self.action_space.sample(), None

    def run(
        self,
        env: gym.Env,
        total_timesteps: int,
        tensorboard_log: Optional[Union[str, Path]] = None,
        metrics_window: int = 50,
        seed: Optional[int] = None,
        verbose: int = 1,
    ) -> "RandomAgent":
        """
        Juega `total_timesteps` pasos con acciones aleatorias registrando las mismas
        métricas que PPO y Rainbow (`combat/*`, `rollout/*`). No aprende nada: sirve
        para dibujar la línea base en TensorBoard con el mismo eje de pasos.
        """
        from torch.utils.tensorboard import SummaryWriter

        writer = None
        if tensorboard_log is not None:
            run_dir = next_run_dir(tensorboard_log, "Random")
            writer = SummaryWriter(log_dir=str(run_dir))
            if verbose:
                print(f"[Random] Logs de TensorBoard en: {run_dir}")

        tracker = CombatMetricsTracker(window_size=metrics_window)
        ep_returns: Deque[float] = deque(maxlen=100)
        ep_lengths: Deque[int] = deque(maxlen=100)

        env.reset(seed=seed)
        ep_return, ep_len = 0.0, 0
        for t in range(1, total_timesteps + 1):
            action, _ = self.predict(None)
            _, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)
            ep_len += 1

            if terminated or truncated:
                tracker.update(info)
                ep_returns.append(ep_return)
                ep_lengths.append(ep_len)
                if writer is not None:
                    for tag, value in tracker.summary().items():
                        writer.add_scalar(tag, value, t)
                    writer.add_scalar("rollout/ep_rew_mean", float(np.mean(ep_returns)), t)
                    writer.add_scalar("rollout/ep_len_mean", float(np.mean(ep_lengths)), t)
                env.reset()
                ep_return, ep_len = 0.0, 0

            if verbose and t % 10_000 == 0:
                summary = tracker.summary()
                if "combat/win_rate_window" in summary:
                    print(f"[Random] {t}/{total_timesteps} pasos | win rate (ventana): "
                          f"{summary['combat/win_rate_window']:.1f}%")

        if writer is not None:
            writer.flush()
            writer.close()
        return self

    def save(self, path: Union[str, Path]) -> None:
        """No hay parámetros que guardar; se deja un marcador para uniformar el flujo."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("random_agent\n", encoding="utf-8")

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        action_space: gym.spaces.Space,
        observation_space: Optional[gym.spaces.Space] = None,
    ) -> "RandomAgent":
        return cls(action_space, observation_space)
