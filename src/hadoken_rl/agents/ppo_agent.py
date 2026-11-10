"""Construcción del agente PPO (Stable-Baselines3) optimizado para entornos paralelos.

Los valores por defecto son una PROPUESTA razonable para 8 entornos y un vector de
observación de 12-20 dimensiones; los hiperparámetros finales siguen sujetos a tuning
(en particular `ent_coef` y `clip_range`, asignados a Alan en el reparto de roles).
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv

_ACTIVATIONS = {"tanh": nn.Tanh, "relu": nn.ReLU}


@dataclass
class PPOConfig:
    n_envs: int = 8                         # entornos paralelos (SubprocVecEnv)
    n_steps: int = 256                      # pasos por entorno antes de cada actualización
    batch_size: int = 256                   # minibatch; debe dividir n_steps * n_envs
    n_epochs: int = 10
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    net_arch_pi: Tuple[int, ...] = (256, 256)   # MLP de la política (actor)
    net_arch_vf: Tuple[int, ...] = (256, 256)   # MLP del crítico
    activation: str = "tanh"

    @property
    def rollout_size(self) -> int:
        return self.n_steps * self.n_envs

    def validate(self) -> None:
        if self.activation not in _ACTIVATIONS:
            raise ValueError(f"activation debe ser una de {list(_ACTIVATIONS)}")
        if self.rollout_size % self.batch_size != 0:
            raise ValueError(
                f"batch_size={self.batch_size} no divide n_steps*n_envs={self.rollout_size}; "
                "SB3 descartaría un minibatch incompleto."
            )


def build_ppo(
    vec_env: VecEnv,
    config: Optional[PPOConfig] = None,
    tensorboard_log: Optional[str] = None,
    seed: Optional[int] = None,
    verbose: int = 1,
    device: str = "auto",
) -> PPO:
    """Crea un PPO con política MLP (actor y crítico separados) sobre `vec_env`."""
    config = config or PPOConfig(n_envs=vec_env.num_envs)
    # El rollout real lo fija el VecEnv recibido, no el valor declarado en la config.
    config.n_envs = vec_env.num_envs
    config.validate()

    return PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        policy_kwargs=dict(
            net_arch=dict(pi=list(config.net_arch_pi), vf=list(config.net_arch_vf)),
            activation_fn=_ACTIVATIONS[config.activation],
        ),
        tensorboard_log=tensorboard_log,
        seed=seed,
        verbose=verbose,
        device=device,
    )
