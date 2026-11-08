"""Utilidades de logging compartidas por los agentes que no usan el logger de SB3."""

from pathlib import Path
from typing import Union


def next_run_dir(base_dir: Union[str, Path], prefix: str) -> Path:
    """
    Devuelve `<base_dir>/<prefix>_<n>` con el primer `n` libre, igual que hace
    Stable-Baselines3 con `PPO_1`, `PPO_2`... Así TensorBoard muestra
    PPO_1, Rainbow_1 y Random_1 como corridas comparables en un mismo logdir.
    """
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    n = 1
    while (base / f"{prefix}_{n}").exists():
        n += 1
    return base / f"{prefix}_{n}"
