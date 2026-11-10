"""Vectorización paralela de combates en Street Fighter II mediante SubprocVecEnv.

Responsabilidades de este módulo (Infraestructura y cómputo — Martín González):
  - Instanciar múltiples procesos independientes del emulador con ``SubprocVecEnv``
    (por defecto ``n_envs=8``) o ``DummyVecEnv`` para depuración de 1 proceso.
  - Desfasar el estado inicial por worker (``start_state_index=rank``) y fijar semillas
    deterministas por subproceso (``seed + rank``) para recolectar experiencia diversa
    contra los distintos arquetipos de rivales en paralelo.
  - Ensamblar la cadena modular del equipo:
    ``BaseEnv -> ActionWrapper (opcional) -> RewardWrapper -> ObservationWrapper (opcional) -> VecMonitor``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecMonitor

from hadoken_rl.envs.base_env import make_env_thunk
from hadoken_rl.envs.reward_wrapper import StreetFighterRewardWrapper


def build_env_pipeline(
    env: gym.Env,
    action_wrapper_fn: Optional[Callable[[gym.Env], gym.Env]] = None,
    use_reward_wrapper: bool = True,
    reward_kwargs: Optional[Dict[str, Any]] = None,
    observation_wrapper_fn: Optional[Callable[[gym.Env], gym.Env]] = None,
) -> gym.Env:
    """Aplica el orden de composición de wrappers acordado por el equipo (Sección 8.1)."""
    if action_wrapper_fn is not None:
        env = action_wrapper_fn(env)
    if use_reward_wrapper:
        env = StreetFighterRewardWrapper(env, **(reward_kwargs or {}))
    if observation_wrapper_fn is not None:
        env = observation_wrapper_fn(env)
    return env


def make_vec_sf_env(
    n_envs: int = 8,
    difficulty: int = 4,
    seed: int = 42,
    use_subproc: bool = True,
    start_method: Optional[str] = "forkserver",
    use_monitor: bool = True,
    action_wrapper_fn: Optional[Callable[[gym.Env], gym.Env]] = None,
    use_reward_wrapper: bool = True,
    reward_kwargs: Optional[Dict[str, Any]] = None,
    observation_wrapper_fn: Optional[Callable[[gym.Env], gym.Env]] = None,
    **base_env_kwargs: Any,
) -> VecEnv:
    """Construye el entorno vectorizado paralelo listo para PPO / evaluación masiva.

    Args:
        n_envs: Cantidad de entornos paralelos (default confirmado: 8).
        difficulty: Dificultad de la CPU en estrellas (default: 4).
        seed: Semilla base; cada worker recibe ``seed + rank``.
        use_subproc: Si es ``True`` y ``n_envs > 1``, usa ``SubprocVecEnv``; si no, ``DummyVecEnv``.
        start_method: Método de creación de procesos en ``SubprocVecEnv`` (default: ``"forkserver"``).
        use_monitor: Envolver con ``VecMonitor`` para reportar ``ep_rew_mean`` y ``ep_len_mean`` en TensorBoard.
        action_wrapper_fn: Hook opcional para ``StreetFighterActionWrapper`` (Alan).
        use_reward_wrapper: Si es ``True``, aplica ``StreetFighterRewardWrapper`` (Javier).
        reward_kwargs: Argumentos opcionales para ``StreetFighterRewardWrapper``.
        observation_wrapper_fn: Hook opcional para ``StreetFighterRAMWrapper`` (Alan).
        **base_env_kwargs: Parámetros adicionales hacia ``make_base_sf_env`` (``frame_skip``, ``agent``, etc.).
    """
    if n_envs < 1:
        raise ValueError(f"n_envs debe ser >= 1 (recibido: {n_envs}).")

    def _pipeline(base_env: gym.Env) -> gym.Env:
        return build_env_pipeline(
            base_env,
            action_wrapper_fn=action_wrapper_fn,
            use_reward_wrapper=use_reward_wrapper,
            reward_kwargs=reward_kwargs,
            observation_wrapper_fn=observation_wrapper_fn,
        )

    env_fns = [
        make_env_thunk(
            rank=rank,
            seed=seed,
            difficulty=difficulty,
            wrapper_fn=_pipeline,
            **base_env_kwargs,
        )
        for rank in range(n_envs)
    ]

    if use_subproc and n_envs > 1:
        vec_env: VecEnv = SubprocVecEnv(env_fns, start_method=start_method)
    else:
        vec_env = DummyVecEnv(env_fns)

    vec_env.seed(seed)

    if use_monitor:
        vec_env = VecMonitor(vec_env)

    return vec_env
