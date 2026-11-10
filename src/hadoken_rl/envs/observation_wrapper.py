"""Wrapper de observaciones: RAM cruda (65.536 bytes) -> vector normalizado de 12-20 dimensiones.

Cada dimensión se describe con un `RAMFeature` y se normaliza a [0, 1]:
  - source="ram":  lee `size` bytes en `address` de la RAM cruda.
  - source="info": toma `info_key` del diccionario `info` (variables ya expuestas por data.json).
  - source="diff": diferencia entre dos features ya definidos (p. ej. distancia enemigo - agente).

IMPORTANTE: las direcciones RAM de `default_features()` están en `None` a propósito. Deben
descubrirse y validarse sobre la ROM real (ver `hadoken_rl.envs.ram_tools` y
`scripts/verify_ram_layout.py`) y escribirse en `configs/ram_layout.json`. El wrapper se
niega a construirse con direcciones sin definir, salvo `allow_unset=True` (solo para humo/tests).
"""

import json
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import gymnasium as gym
import numpy as np

GENESIS_RAM_BASE = 0xFF0000  # las direcciones de data.json de Genesis están en el espacio 68k
GENESIS_RAM_SIZE = 0x10000   # 65.536 bytes


@dataclass
class RAMFeature:
    name: str
    source: str = "ram"                  # "ram" | "info" | "diff"
    address: Optional[int] = None        # dirección absoluta (como en data.json)
    size: int = 1                        # bytes a leer
    signed: bool = False
    bcd: bool = False                    # el valor está en BCD (p. ej. 0x99 -> 99)
    byteorder: str = "big"               # el 68000 de Genesis es big-endian
    info_key: Optional[str] = None       # para source="info" (y referencia de validación en "ram")
    minuend: Optional[str] = None        # para source="diff": nombre del feature a restar de
    subtrahend: Optional[str] = None     # para source="diff": nombre del feature restado
    min_value: float = 0.0               # rango usado para normalizar a [0, 1]
    max_value: float = 1.0
    default: float = 0.0                 # valor si falta la variable en `info`
    note: str = field(default="", repr=False)

    def normalize(self, value: float) -> float:
        span = self.max_value - self.min_value
        return float(np.clip((value - self.min_value) / span, 0.0, 1.0)) if span > 0 else 0.0


def default_features() -> List[RAMFeature]:
    """Vector propuesto de 12 dimensiones. Direcciones y rangos de x/y/timer/animación por calibrar."""
    return [
        RAMFeature("agent_x", address=None, size=2, min_value=0, max_value=1024, note="calibrar rango con verify_ram_layout"),
        RAMFeature("agent_y", address=None, size=2, min_value=0, max_value=256, note="calibrar rango"),
        RAMFeature("enemy_x", address=None, size=2, min_value=0, max_value=1024, note="calibrar rango"),
        RAMFeature("enemy_y", address=None, size=2, min_value=0, max_value=256, note="calibrar rango"),
        RAMFeature("agent_health", source="info", info_key="health", min_value=0, max_value=176, default=176),
        RAMFeature("enemy_health", source="info", info_key="enemy_health", min_value=0, max_value=176, default=176),
        RAMFeature("timer", address=None, size=1, min_value=0, max_value=99),
        RAMFeature("agent_animation", address=None, size=1, min_value=0, max_value=255, note="estado categórico escalado"),
        RAMFeature("enemy_animation", address=None, size=1, min_value=0, max_value=255, note="estado categórico escalado"),
        RAMFeature("dx", source="diff", minuend="enemy_x", subtrahend="agent_x", min_value=-1024, max_value=1024),
        RAMFeature("dy", source="diff", minuend="enemy_y", subtrahend="agent_y", min_value=-256, max_value=256),
        RAMFeature("health_diff", source="diff", minuend="agent_health", subtrahend="enemy_health",
                   min_value=-176, max_value=176),
    ]


# --------------------------------------------------------------------------- #
# Carga / guardado del layout
# --------------------------------------------------------------------------- #
def _to_int(v: Any) -> Optional[int]:
    if v is None or isinstance(v, int):
        return v
    return int(str(v), 0)  # acepta "0xFF8000" o "16744448"


def load_layout(path: Union[str, Path]) -> Dict[str, Any]:
    """Lee un JSON {address_base, ram_size, features:[...]} y devuelve features ya construidos."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    features = []
    for raw in data["features"]:
        raw = dict(raw)
        raw["address"] = _to_int(raw.get("address"))
        features.append(RAMFeature(**{k: v for k, v in raw.items() if not k.startswith("_")}))
    return {
        "address_base": _to_int(data.get("address_base", GENESIS_RAM_BASE)),
        "ram_size": _to_int(data.get("ram_size", GENESIS_RAM_SIZE)),
        "features": features,
        "byte_swapped": bool(data.get("byte_swapped", False)),
    }


def save_layout(features: Sequence[RAMFeature], path: Union[str, Path],
                address_base: int = GENESIS_RAM_BASE, ram_size: int = GENESIS_RAM_SIZE) -> None:
    out = {
        "address_base": hex(address_base),
        "ram_size": ram_size,
        "features": [
            {**asdict(f), "address": None if f.address is None else hex(f.address)} for f in features
        ],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Wrapper
# --------------------------------------------------------------------------- #
class StreetFighterRAMWrapper(gym.ObservationWrapper):
    """Convierte la RAM cruda del entorno base en un vector float32 en [0, 1]."""

    def __init__(
        self,
        env: gym.Env,
        features: Optional[Sequence[RAMFeature]] = None,
        layout_path: Optional[Union[str, Path]] = None,
        address_base: int = GENESIS_RAM_BASE,
        allow_unset: bool = False,
        byte_swapped: bool = False, 
    ):
        super().__init__(env)
        if layout_path is not None:
            layout = load_layout(layout_path)
            features, address_base = layout["features"], layout["address_base"]
        self.features: List[RAMFeature] = list(features) if features is not None else default_features()
        self.address_base = address_base
        self.allow_unset = allow_unset
        self.byte_swapped = byte_swapped 

        if not 12 <= len(self.features) <= 20:
            warnings.warn(f"El vector tiene {len(self.features)} dimensiones (propuesta: 12 a 20).")
        names = [f.name for f in self.features]
        if len(set(names)) != len(names):
            raise ValueError(f"Nombres de feature duplicados: {names}")

        ram_shape = getattr(env.observation_space, "shape", None)
        if not ram_shape or len(ram_shape) != 1:
            raise ValueError(
                "El entorno base debe entregar la RAM como vector 1-D "
                "(crear con obs_type=retro.Observations.RAM)."
            )
        self.ram_size = int(ram_shape[0])

        self._unset = [f.name for f in self.features if f.source == "ram" and f.address is None]
        if self._unset and not allow_unset:
            raise ValueError(
                "Faltan direcciones RAM para: " + ", ".join(self._unset) + ". Descúbrelas con "
                "scripts/verify_ram_layout.py y complétalas en configs/ram_layout.json "
                "(o usa allow_unset=True solo para pruebas de humo)."
            )
        for f in self.features:
            self._validate(f, names)

        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(len(self.features),), dtype=np.float32)
        self._warned: set = set()
        self._last_info: Dict[str, Any] = {}

    @property
    def feature_names(self) -> List[str]:
        return [f.name for f in self.features]

    def _validate(self, f: RAMFeature, names: List[str]) -> None:
        if f.source == "ram" and f.address is not None:
            off = f.address - self.address_base
            if off < 0 or off + f.size > self.ram_size:
                raise ValueError(
                    f"{f.name}: dirección {hex(f.address)} fuera de la RAM "
                    f"({hex(self.address_base)}..{hex(self.address_base + self.ram_size)})"
                )
        elif f.source == "info" and not f.info_key:
            raise ValueError(f"{f.name}: source='info' requiere info_key")
        elif f.source == "diff":
            for ref in (f.minuend, f.subtrahend):
                target = next((g for g in self.features if g.name == ref), None)
                if target is None or target.source == "diff":
                    raise ValueError(f"{f.name}: referencia inválida {ref!r} en source='diff'")
        elif f.source not in ("ram", "info", "diff"):
            raise ValueError(f"{f.name}: source desconocido {f.source!r}")

    # ------------------------------------------------------------------ #
    def _read_raw(self, f: RAMFeature, ram: np.ndarray, info: Dict[str, Any]) -> float:
        if f.source == "ram":
            if f.address is None:  # solo con allow_unset
                return f.default
            off = f.address - self.address_base
            idx = [((off + k) ^ 1) if self.byte_swapped else (off + k) for k in range(f.size)]
            raw_bytes = ram[idx].tobytes()
            if f.bcd:
                return float(sum(((b >> 4) * 10 + (b & 0x0F)) * 100 ** (f.size - 1 - i)
                                 for i, b in enumerate(raw_bytes)))
            return float(int.from_bytes(raw_bytes, f.byteorder, signed=f.signed))
        if f.info_key not in info:
            if f.name not in self._warned:
                warnings.warn(f"'{f.info_key}' no está en info; se usa {f.default} para '{f.name}'.")
                self._warned.add(f.name)
            return f.default
        return float(info[f.info_key])

    def _vectorize(self, ram: Any, info: Dict[str, Any]) -> np.ndarray:
        ram = np.asarray(ram, dtype=np.uint8)
        raw: Dict[str, float] = {}
        for f in self.features:
            if f.source != "diff":
                raw[f.name] = self._read_raw(f, ram, info)
        out = np.zeros(len(self.features), dtype=np.float32)
        for i, f in enumerate(self.features):
            value = raw[f.minuend] - raw[f.subtrahend] if f.source == "diff" else raw[f.name]
            out[i] = f.normalize(value)
        return out

    # ObservationWrapper solo recibe obs; se sobreescriben reset/step para disponer de `info`.
    def observation(self, observation: Any) -> np.ndarray:
        return self._vectorize(observation, self._last_info)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_info = info
        return self._vectorize(obs, info), info

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_info = info
        return self._vectorize(obs, info), reward, terminated, truncated, info
