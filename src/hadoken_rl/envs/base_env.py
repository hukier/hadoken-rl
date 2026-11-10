"""Entorno base de Street Fighter II sobre Stable-Retro y Gymnasium.

Responsabilidades de este módulo (Infraestructura — Martín González):
  - Instanciar ``StreetFighterIISpecialChampionEdition-Genesis-v0`` en modo headless.
  - Cargar y rotar estados iniciales (``.state``) verificados por dificultad (1..8) y rival.
  - Ejecutar repetición de acción configurable (``frame_skip=4`` por defecto) cortando al
    instante si ocurre K.O. o fin de tiempo de la ronda.
  - Garantizar que 1 episodio equivalga exactamente a 1 ronda de combate activo (evitando
    pantallas intermedias o de Continue que alteran la RAM).
  - Poblar ``info`` tanto en ``reset()`` como en ``step()`` incluyendo coordenadas ``(x, y)``
    y ``(enemy_x, enemy_y)`` validadas en RAM para ``StreetFighterRewardWrapper``.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import gymnasium as gym
import stable_retro

GAME_ID = "StreetFighterIISpecialChampionEdition-Genesis-v0"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_JSON = PROJECT_ROOT / "configs" / "sf2_data.json"
DEFAULT_STATES_DIR = PROJECT_ROOT / "configs" / "states"

CHARACTER_NAMES: Dict[int, str] = {
    0: "Ryu",
    1: "Honda",
    2: "Blanka",
    3: "Guile",
    4: "Ken",
    5: "ChunLi",
    6: "Zangief",
    7: "Dhalsim",
    8: "Balrog",
    9: "Vega",
    10: "Sagat",
    11: "Bison",
}


def discover_state_files(
    difficulty: int = 4,
    agent: str = "Ryu",
    opponents: Optional[Sequence[str]] = None,
    states_dir: Union[str, Path] = DEFAULT_STATES_DIR,
    mode: str = "Champion",
) -> List[Path]:
    """Descubre archivos ``.state`` para una dificultad, personaje agente y conjunto de rivales."""
    if not (1 <= difficulty <= 8):
        raise ValueError(f"La dificultad debe estar entre 1 y 8 (recibido: {difficulty}).")

    base_dir = Path(states_dir)
    if not base_dir.exists():
        raise FileNotFoundError(
            f"No existe el directorio de estados: {base_dir}. "
            "Ejecuta 'uv run python scripts/generate_states.py' primero."
        )

    if opponents is not None:
        paths: List[Path] = []
        for opp in opponents:
            candidate = base_dir / f"{mode}.Level{difficulty}.{agent}Vs{opp}.state"
            if not candidate.exists():
                raise FileNotFoundError(
                    f"No se encontró el estado requerido: {candidate}. "
                    "Puedes generarlo con scripts/generate_states.py."
                )
            paths.append(candidate)
        return paths

    pattern = f"{mode}.Level{difficulty}.{agent}Vs*.state"
    matched = sorted(base_dir.glob(pattern))
    if not matched:
        raise FileNotFoundError(
            f"No se encontraron archivos '{pattern}' en {base_dir}. "
            "Ejecuta 'uv run python scripts/generate_states.py' para generarlos."
        )
    return matched


class StreetFighterBaseEnv(gym.Wrapper):
    """Wrapper de infraestructura base para combates de 1 ronda en Stable-Retro."""

    def __init__(
        self,
        env: gym.Env,
        state_files: Sequence[Path],
        frame_skip: int = 4,
        state_selection: Literal["round_robin", "random"] = "round_robin",
        start_state_index: int = 0,
        max_episode_steps: Optional[int] = 2500,
    ):
        if frame_skip < 1:
            raise ValueError(f"frame_skip debe ser >= 1 (recibido: {frame_skip}).")
        if not state_files:
            raise ValueError("Se requiere al menos un archivo .state para inicializar StreetFighterBaseEnv.")
        if state_selection not in ("round_robin", "random"):
            raise ValueError(f"state_selection inválido: {state_selection}. Usa 'round_robin' o 'random'.")

        super().__init__(env)
        self.frame_skip = frame_skip
        self.state_selection = state_selection
        self.max_episode_steps = max_episode_steps

        # Precargar los bytes descomprimidos de cada .state en RAM (~50 KB c/u) para cero I/O en reset()
        self._states: List[Tuple[str, bytes]] = []
        for path in state_files:
            p = Path(path)
            with gzip.open(p, "rb") as fh:
                self._states.append((p.stem, fh.read()))

        self._rr_cursor = start_state_index % len(self._states)
        self._active_state_name: str = self._states[self._rr_cursor][0]
        self._elapsed_steps: int = 0

    @property
    def state_names(self) -> List[str]:
        """Lista de nombres de estados cargados en el pool."""
        return [name for name, _ in self._states]

    def _select_next_state(self, options: Optional[Dict[str, Any]] = None) -> Tuple[str, bytes]:
        if options is not None:
            if "state_index" in options:
                idx = int(options["state_index"]) % len(self._states)
                return self._states[idx]
            if "opponent" in options:
                target_suffix = f"Vs{options['opponent']}"
                for name, data in self._states:
                    if name.endswith(target_suffix):
                        return name, data
                raise ValueError(f"Rival '{options['opponent']}' no está en states cargados: {self.state_names}")

        if self.state_selection == "random":
            idx = int(self.np_random.integers(0, len(self._states)))
            return self._states[idx]

        selected = self._states[self._rr_cursor]
        self._rr_cursor = (self._rr_cursor + 1) % len(self._states)
        return selected

    def _build_clean_info(self, raw_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not raw_info:
            raw_info = dict(self.env.unwrapped.data.lookup_all())
        else:
            raw_info = dict(raw_info)

        # En SF2 Genesis la vida es >i2 y cae a -1 en ciertos K.O.; truncar a >= 0 evita sesgar ΔHP en 1 punto
        raw_info["health"] = max(0, int(raw_info.get("health", 176)))
        raw_info["enemy_health"] = max(0, int(raw_info.get("enemy_health", 176)))

        diff_ram = int(raw_info.get("difficulty", 3))
        agent_id = int(raw_info.get("agent_char_id", 0))
        enemy_id = int(raw_info.get("enemy_char_id", 0))

        raw_info["difficulty_level"] = diff_ram + 1
        raw_info["agent_name"] = CHARACTER_NAMES.get(agent_id, f"Unknown({agent_id})")
        raw_info["enemy_name"] = CHARACTER_NAMES.get(enemy_id, f"Unknown({enemy_id})")
        raw_info["state_name"] = self._active_state_name
        return raw_info

    @staticmethod
    def _is_round_over(info: Dict[str, Any]) -> bool:
        if info["health"] <= 0 or info["enemy_health"] <= 0:
            return True
        if int(info.get("timer", 99)) <= 0:
            return True
        if int(info.get("matches_won", 0)) > 0 or int(info.get("enemy_matches_won", 0)) > 0:
            return True
        return False

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        if seed is not None:
            # Inicializa self.np_random del wrapper de forma determinista
            gym.Wrapper.reset(self, seed=seed)

        state_name, state_bytes = self._select_next_state(options)
        self._active_state_name = state_name

        retro_env = self.env.unwrapped
        retro_env.initial_state = state_bytes
        retro_env.statename = state_name

        obs, _ = self.env.reset(seed=seed, options=options)
        self._elapsed_steps = 0
        info = self._build_clean_info()
        return obs, info

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        obs = None
        total_native_reward = 0.0
        info: Dict[str, Any] = {}
        round_over = False
        frames_executed = 0

        for _ in range(self.frame_skip):
            obs, rew, _, _, raw_info = self.env.step(action)
            total_native_reward += float(rew)
            frames_executed += 1
            info = self._build_clean_info(raw_info)
            if self._is_round_over(info):
                round_over = True
                break

        self._elapsed_steps += 1
        terminated = round_over
        truncated = bool(
            not terminated
            and self.max_episode_steps is not None
            and self._elapsed_steps >= self.max_episode_steps
        )
        info["frames_executed"] = frames_executed
        info["elapsed_steps"] = self._elapsed_steps
        return obs, total_native_reward, terminated, truncated, info


def make_base_sf_env(
    difficulty: int = 4,
    agent: str = "Ryu",
    opponents: Optional[Sequence[str]] = None,
    state_paths: Optional[Sequence[Union[str, Path]]] = None,
    states_dir: Union[str, Path] = DEFAULT_STATES_DIR,
    data_json_path: Union[str, Path] = DEFAULT_DATA_JSON,
    frame_skip: int = 4,
    state_selection: Literal["round_robin", "random"] = "round_robin",
    start_state_index: int = 0,
    max_episode_steps: Optional[int] = 2500,
    obs_type: Literal["ram", "image"] = "ram",
    render_mode: Optional[str] = None,
) -> StreetFighterBaseEnv:
    """Construye una instancia headless y limpia de Street Fighter II lista para wrappers RL.

    Args:
        difficulty: Nivel de dificultad de la CPU en estrellas (1 a 8; default: 4).
        agent: Nombre del personaje del agente P1 (default: ``"Ryu"``).
        opponents: Lista opcional de rivales CPU; si es ``None`` carga todos los disponibles
            para ``difficulty`` y ``agent`` en ``states_dir``.
        state_paths: Rutas explícitas a archivos ``.state`` (sobreescribe ``discover_state_files``).
        states_dir: Carpeta donde residen los archivos ``.state`` versionados.
        data_json_path: Ruta al archivo ``sf2_data.json`` con variables RAM validadas.
        frame_skip: Número de frames del emulador por decisión del agente (default: 4).
        state_selection: ``"round_robin"`` (ciclo determinista) o ``"random"`` (muestreo con semilla).
        start_state_index: Desplazamiento inicial del pool de estados (útil para ``SubprocVecEnv``).
        max_episode_steps: Tope de pasos por ronda como resguardo de truncamiento (default: 2500).
        obs_type: ``"ram"`` (vector de 65536 bytes sin costo de copia de píxeles) o ``"image"``.
        render_mode: ``None`` para ejecución *headless* rápida, ``"rgb_array"`` o ``"human"``.
    """
    resolved_states = (
        [Path(p) for p in state_paths]
        if state_paths is not None
        else discover_state_files(
            difficulty=difficulty,
            agent=agent,
            opponents=opponents,
            states_dir=states_dir,
        )
    )

    data_path = Path(data_json_path)
    if not data_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de variables RAM: {data_path}")

    retro_obs = (
        stable_retro.Observations.RAM
        if obs_type == "ram"
        else stable_retro.Observations.IMAGE
    )

    raw_env = stable_retro.make(
        game=GAME_ID,
        state=stable_retro.State.NONE,
        info=str(data_path),
        use_restricted_actions=stable_retro.Actions.FILTERED,
        obs_type=retro_obs,
        render_mode=render_mode,
    )

    return StreetFighterBaseEnv(
        env=raw_env,
        state_files=resolved_states,
        frame_skip=frame_skip,
        state_selection=state_selection,
        start_state_index=start_state_index,
        max_episode_steps=max_episode_steps,
    )


def make_env_thunk(
    rank: int = 0,
    seed: int = 0,
    difficulty: int = 4,
    wrapper_fn: Optional[Callable[[gym.Env], gym.Env]] = None,
    **base_env_kwargs: Any,
) -> Callable[[], gym.Env]:
    """Crea un constructor serializable por proceso (compatible con ``SubprocVecEnv``)."""

    def _init() -> gym.Env:
        import os

        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")

        env: gym.Env = make_base_sf_env(
            difficulty=difficulty,
            start_state_index=rank,
            **base_env_kwargs,
        )
        if wrapper_fn is not None:
            env = wrapper_fn(env)
        env.action_space.seed(seed + rank)
        return env

    return _init
