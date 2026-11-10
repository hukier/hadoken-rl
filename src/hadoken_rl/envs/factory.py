"""Fábrica del entorno completo de Street Fighter II con todos los wrappers.

Cadena de ensamblaje (ver sección 8.1 del contexto):
    BaseEnv (Martín) -> ActionWrapper -> RewardWrapper -> RAMWrapper -> (Sub)VecEnv

`kind="mock"` conserva los entornos simulados de `tests/` para desarrollo sin ROM.
"""

import inspect
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Callable, List, Optional

import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from hadoken_rl.envs.action_wrapper import StreetFighterActionWrapper
from hadoken_rl.envs.observation_wrapper import StreetFighterRAMWrapper
from hadoken_rl.envs.reward_wrapper import StreetFighterRewardWrapper

DEFAULT_GAME = "StreetFighterIISpecialChampionEdition-Genesis-v0"
DEFAULT_LAYOUT_PATH = Path(__file__).resolve().parents[3] / "configs" / "ram_layout.json"


@dataclass
class EnvConfig:
    kind: str = "real"                  # "real" (Stable-Retro) | "mock" (sin ROM)
    difficulty: int = 4                 # dificultad principal de entrenamiento (contexto, sec. 1)
    game: str = DEFAULT_GAME
    state: Optional[str] = None         # estado inicial .state; admite "{difficulty}" como placeholder
    hold_frames: int = 4                # frames por macroacción (propuesta)
    layout_path: Path = DEFAULT_LAYOUT_PATH
    allow_unset_ram: bool = False       # solo humo: permite direcciones RAM sin definir
    seed: Optional[int] = None


def _build_base_env(cfg: EnvConfig) -> gym.Env:
    """
    Entorno base con RAM como observación y `MultiBinary(12)` como acción.

    Usa `make_base_sf_env` de `hadoken_rl.envs.base_env` (módulo de Martín) si existe,
    pasándole solo los argumentos que su firma acepte. Si ese módulo aún no está en el
    repositorio, construye el entorno directamente con Stable-Retro.
    """
    state = cfg.state.format(difficulty=cfg.difficulty) if cfg.state else None
    try:
        from hadoken_rl.envs.base_env import make_base_sf_env  # type: ignore
    except ImportError:
        make_base_sf_env = None

    if make_base_sf_env is not None:
        candidates = {"difficulty": cfg.difficulty, "game": cfg.game, "state": state, "seed": cfg.seed}
        accepted = inspect.signature(make_base_sf_env).parameters
        kwargs = {k: v for k, v in candidates.items() if k in accepted and v is not None}
        return make_base_sf_env(**kwargs)

    try:
        import stable_retro as retro  # nombre actual del paquete (stable-retro >= 1.0)
    except ImportError:
        try:
            import retro  # nombre antiguo (deprecado)
        except ImportError as exc:
            raise ImportError(
                "No se encontró hadoken_rl.envs.base_env ni stable-retro. Instala stable-retro "
                "(Linux/WSL2) y la ROM, o ejecuta con --env mock."
            ) from exc
    return retro.make(
        game=cfg.game,
        state=state if state is not None else retro.State.DEFAULT,
        obs_type=retro.Observations.RAM,
        use_restricted_actions=retro.Actions.ALL,
        render_mode=None,
    )


def _build_mock_env(cfg: EnvConfig, for_eval: bool) -> gym.Env:
    if for_eval:
        from tests.test_benchmark import MockCombatEnv
        base = MockCombatEnv(difficulty=cfg.difficulty)
    else:
        from tests.test_callbacks import DummyFightEnv
        base = DummyFightEnv()
    return StreetFighterRewardWrapper(base)


def make_env(cfg: EnvConfig, for_eval: bool = False) -> gym.Env:
    """Construye un entorno individual con la pila completa de wrappers."""
    if cfg.kind == "mock":
        return _build_mock_env(cfg, for_eval)
    if cfg.kind != "real":
        raise ValueError(f"kind debe ser 'real' o 'mock', no {cfg.kind!r}")

    env = _build_base_env(cfg)
    env = StreetFighterActionWrapper(env, hold_frames=cfg.hold_frames)
    env = StreetFighterRewardWrapper(env)
    env = StreetFighterRAMWrapper(env, layout_path=cfg.layout_path, allow_unset=cfg.allow_unset_ram)
    return env


def make_env_fns(cfg: EnvConfig, n_envs: int) -> List[Callable[[], gym.Env]]:
    """Una función constructora por entorno (semilla distinta por proceso)."""
    base_seed = cfg.seed if cfg.seed is not None else 0
    return [partial(make_env, replace(cfg, seed=base_seed + rank)) for rank in range(n_envs)]


def make_vec_env(cfg: EnvConfig, n_envs: int) -> VecEnv:
    """SubprocVecEnv para el entorno real (un emulador por proceso); DummyVecEnv para el mock."""
    fns = make_env_fns(cfg, n_envs)
    if cfg.kind == "real" and n_envs > 1:
        # Verificación previa en el proceso principal: si falta la ROM, stable-retro, direcciones RAM
        # sin definir, etc., el error aparece aquí con su mensaje claro y no enterrado en un subproceso.
        make_env(cfg).close()
        return SubprocVecEnv(fns)
    return DummyVecEnv(fns)


def add_env_args(parser) -> None:
    """Argumentos de línea de comandos compartidos por train.py y evaluate.py."""
    g = parser.add_argument_group("entorno")
    g.add_argument("--env", dest="env_kind", choices=["real", "mock"], default="real",
                   help="'real' usa Stable-Retro + wrappers; 'mock' usa entornos simulados sin ROM")
    g.add_argument("--difficulty", type=int, default=4, help="Dificultad de la CPU para entrenar (default: 4)")
    g.add_argument("--game", type=str, default=DEFAULT_GAME, help="Id del juego en Stable-Retro")
    g.add_argument("--state", type=str, default=None,
                   help="Estado inicial .state; admite el placeholder {difficulty}, p. ej. 'Champion.Level{difficulty}'")
    g.add_argument("--hold-frames", type=int, default=4, help="Frames que dura cada macroacción")
    g.add_argument("--ram-layout", type=str, default=str(DEFAULT_LAYOUT_PATH), help="JSON con el layout del vector RAM")
    g.add_argument("--allow-unset-ram", action="store_true",
                   help="Permite direcciones RAM sin definir (rellena con 0). Solo para pruebas de humo")


def env_config_from_args(args, difficulty: Optional[int] = None) -> EnvConfig:
    return EnvConfig(
        kind=args.env_kind,
        difficulty=args.difficulty if difficulty is None else difficulty,
        game=args.game,
        state=args.state,
        hold_frames=args.hold_frames,
        layout_path=Path(args.ram_layout),
        allow_unset_ram=args.allow_unset_ram,
        seed=getattr(args, "seed", None),
    )
