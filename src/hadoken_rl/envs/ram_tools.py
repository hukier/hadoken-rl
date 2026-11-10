"""Herramientas para descubrir y validar direcciones RAM contra variables conocidas de `info`.

Flujo: se juega con acciones aleatorias, se guardan (RAM, info) por frame y se buscan los
bytes de la RAM que coinciden con una variable ya expuesta (p. ej. `health`). Las funciones
son puras (reciben matrices) para poder probarlas sin la ROM.
"""

from typing import Dict, List, Sequence

import numpy as np


def search_candidates(ram_frames: np.ndarray, values: Sequence[float], threshold: float = 0.95,
                      min_unique: int = 2) -> np.ndarray:
    """
    Offsets cuyo byte coincide con `values` en >= `threshold` de los frames.

    Args:
        ram_frames: (T, R) uint8, una fila por frame.
        values: (T,) valor de referencia por frame.
        min_unique: descarta offsets constantes (coinciden por casualidad si la variable no cambia).
    """
    ram_frames = np.asarray(ram_frames, dtype=np.uint8)
    values = np.asarray(values, dtype=np.float64)
    match_rate = (ram_frames == values[:, None]).mean(axis=0)
    varying = np.array([len(np.unique(ram_frames[:, i])) >= min_unique for i in range(ram_frames.shape[1])])
    return np.where((match_rate >= threshold) & varying)[0]


def decode_series(ram_frames: np.ndarray, offset: int, size: int = 1, byteorder: str = "big",
                  signed: bool = False) -> np.ndarray:
    """Decodifica `size` bytes en `offset` para todos los frames."""
    ram_frames = np.asarray(ram_frames, dtype=np.uint8)
    return np.array([
        int.from_bytes(row[offset:offset + size].tobytes(), byteorder, signed=signed) for row in ram_frames
    ], dtype=np.float64)


def match_rate(ram_frames: np.ndarray, offset: int, reference: Sequence[float], size: int = 1,
               byteorder: str = "big", signed: bool = False) -> float:
    """Fracción de frames en que la RAM decodificada coincide con la referencia de `info`."""
    return float(np.mean(decode_series(ram_frames, offset, size, byteorder, signed) == np.asarray(reference)))


def observed_stats(ram_frames: np.ndarray, offset: int, size: int = 1, byteorder: str = "big",
                   signed: bool = False) -> Dict[str, float]:
    """min / max / nº de valores distintos observados; sirve para calibrar `min_value`/`max_value`."""
    s = decode_series(ram_frames, offset, size, byteorder, signed)
    return {"min": float(s.min()), "max": float(s.max()), "unique": int(len(np.unique(s)))}
