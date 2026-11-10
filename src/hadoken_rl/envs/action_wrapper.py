"""Wrapper de acciones: 14 macroacciones discretas -> 12 botones binarios del emulador.

El agente decide con `Discrete(14)`; este wrapper traduce cada macroacción a una
secuencia de estados de los 12 botones de Genesis y la ejecuta sobre el entorno base
(`MultiBinary(12)`), repitiendo cada estado durante varios frames.

PROPUESTA PENDIENTE DE APROBACIÓN DEL EQUIPO: el catálogo de macroacciones, los
frames por acción y el mapeo de botones físicos son decisiones abiertas en el
documento de contexto. Todo está concentrado en `DEFAULT_MACROS`, `PHYSICAL_BUTTONS`
y `hold_frames` para poder cambiarlo en un solo lugar.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np

# Orden de botones de Genesis en Stable-Retro. Se usa solo si el entorno base no
# expone `env.unwrapped.buttons` (lo normal es leerlo del propio entorno).
GENESIS_BUTTONS: Tuple[str, ...] = (
    "B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z",
)

# Entradas simbólicas -> botón físico del pad de 6 botones de Genesis.
# FORWARD / BACK se resuelven a RIGHT / LEFT según hacia dónde mire el personaje.
# Convención SF2 Genesis: X/Y/Z = puño débil/medio/fuerte; A/B/C = patada débil/media/fuerte.
PHYSICAL_BUTTONS: Dict[str, str] = {
    "UP": "UP", "DOWN": "DOWN",
    "LP": "X", "MP": "Y", "HP": "Z",
    "LK": "A", "MK": "B", "HK": "C",
}

# Un segmento es (entradas simbólicas, frames). frames=None usa `hold_frames`.
Segment = Tuple[Tuple[str, ...], Optional[int]]

DEFAULT_MACROS: List[Tuple[str, List[Segment]]] = [
    ("NOOP",              [((), None)]),
    ("FORWARD",           [(("FORWARD",), None)]),
    ("BLOCK_STAND",       [(("BACK",), None)]),            # en SF2 bloquear = mantener atrás
    ("CROUCH",            [(("DOWN",), None)]),
    ("BLOCK_CROUCH",      [(("DOWN", "BACK"), None)]),
    ("JUMP",              [(("UP",), None)]),
    ("JUMP_FORWARD",      [(("UP", "FORWARD"), None)]),
    ("LIGHT_PUNCH",       [(("LP",), None)]),
    ("HEAVY_PUNCH",       [(("HP",), None)]),
    ("LIGHT_KICK",        [(("LK",), None)]),
    ("HEAVY_KICK",        [(("HK",), None)]),
    ("CROUCH_HEAVY_KICK", [(("DOWN", "HK"), None)]),
    # Especiales: dependen del personaje (válidos para Ryu/Ken). Secuencias de ~6 frames.
    ("HADOKEN",           [(("DOWN",), 2), (("DOWN", "FORWARD"), 2), (("FORWARD", "MP"), 2)]),
    ("SHORYUKEN",         [(("FORWARD",), 2), (("DOWN",), 2), (("DOWN", "FORWARD", "HP"), 2)]),
]


class StreetFighterActionWrapper(gym.Wrapper):
    """
    Traduce `Discrete(N)` a las entradas binarias del emulador.

    Args:
        env: entorno base con `action_space = MultiBinary(12)`.
        macros: catálogo [(nombre, segmentos)]; por defecto `DEFAULT_MACROS` (N=14).
        buttons: nombres de los botones en el orden del entorno. Por defecto se lee de
            `env.unwrapped.buttons` y, si no existe, se usa `GENESIS_BUTTONS`.
        hold_frames: frames que se mantiene cada segmento sin duración explícita.
        agent_x_key / enemy_x_key: claves de `info` para decidir hacia dónde mira el
            personaje (adelante = hacia el rival). Si faltan, se asume `default_facing`.
        default_facing: +1 mira a la derecha, -1 a la izquierda.
    """

    def __init__(
        self,
        env: gym.Env,
        macros: Optional[Sequence[Tuple[str, List[Segment]]]] = None,
        buttons: Optional[Sequence[str]] = None,
        hold_frames: int = 4,
        agent_x_key: str = "x",
        enemy_x_key: str = "enemy_x",
        default_facing: int = 1,
    ):
        super().__init__(env)
        if hold_frames < 1:
            raise ValueError("hold_frames debe ser >= 1")
        self.hold_frames = hold_frames
        self.agent_x_key = agent_x_key
        self.enemy_x_key = enemy_x_key
        self.default_facing = 1 if default_facing >= 0 else -1
        self.facing = self.default_facing

        self.buttons = tuple(buttons or getattr(env.unwrapped, "buttons", None) or GENESIS_BUTTONS)
        n_buttons = getattr(env.action_space, "n", None)
        if n_buttons is not None and n_buttons != len(self.buttons):
            raise ValueError(
                f"El entorno base espera {n_buttons} botones pero se declararon {len(self.buttons)}: {self.buttons}"
            )

        macros = list(macros if macros is not None else DEFAULT_MACROS)
        if not macros:
            raise ValueError("El catálogo de macroacciones no puede estar vacío")
        self.macro_names: List[str] = [name for name, _ in macros]
        # Para cada macro y orientación (+1/-1): lista de (vector de 12 botones, frames)
        self._compiled: List[Dict[int, List[Tuple[np.ndarray, int]]]] = [
            {facing: self._compile(segments, facing) for facing in (1, -1)} for _, segments in macros
        ]
        self.action_space = gym.spaces.Discrete(len(macros))

    # ------------------------------------------------------------------ #
    def _resolve(self, token: str, facing: int) -> str:
        if token == "FORWARD":
            return "RIGHT" if facing > 0 else "LEFT"
        if token == "BACK":
            return "LEFT" if facing > 0 else "RIGHT"
        if token not in PHYSICAL_BUTTONS:
            raise ValueError(f"Entrada simbólica desconocida: {token!r}")
        return PHYSICAL_BUTTONS[token]

    def _compile(self, segments: List[Segment], facing: int) -> List[Tuple[np.ndarray, int]]:
        compiled = []
        for tokens, frames in segments:
            vec = np.zeros(len(self.buttons), dtype=np.int8)
            for token in tokens:
                name = self._resolve(token, facing)
                if name not in self.buttons:
                    raise ValueError(f"El botón {name!r} no existe en el entorno base: {self.buttons}")
                vec[self.buttons.index(name)] = 1
            compiled.append((vec, int(frames if frames is not None else self.hold_frames)))
        return compiled

    def _update_facing(self, info: Dict[str, Any]) -> None:
        if self.agent_x_key in info and self.enemy_x_key in info:
            self.facing = 1 if float(info[self.agent_x_key]) <= float(info[self.enemy_x_key]) else -1

    # ------------------------------------------------------------------ #
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.facing = self.default_facing
        self._update_facing(info)
        return obs, info

    def step(self, action: Any):
        idx = int(np.asarray(action).item())
        if not 0 <= idx < len(self._compiled):
            raise ValueError(f"Macroacción inválida: {idx} (válidas: 0..{len(self._compiled) - 1})")

        total_reward = 0.0
        obs, terminated, truncated, info = None, False, False, {}
        for vec, frames in self._compiled[idx][self.facing]:
            for _ in range(frames):
                obs, reward, terminated, truncated, info = self.env.step(vec.copy())
                total_reward += float(reward)
                self._update_facing(info)
                if terminated or truncated:
                    break
            if terminated or truncated:
                break

        info = dict(info)
        info["macro_action"] = self.macro_names[idx]
        return obs, total_reward, terminated, truncated, info
