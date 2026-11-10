"""Generador reproducible de archivos .state para Street Fighter II': Special Champion Edition.

Navega por los menús reales del juego (Option Mode -> Champion -> Game Start -> Player Select)
para generar estados iniciales limpios al comenzar el Round 1 (justo al habilitarse el control
tras 'FIGHT!') con:
  - Modo Champion (velocidad estándar clásica)
  - Nivel de dificultad exacto de la CPU (1 a 8 estrellas) verificado en RAM
  - Personaje del agente (P1) configurable (por defecto Ryu)
  - Rivales rotativos de la CPU (P2) verificados por ID de personaje en RAM
"""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path
from typing import Dict, List, Tuple

import stable_retro

GAME_ID = "StreetFighterIISpecialChampionEdition-Genesis-v0"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_JSON = PROJECT_ROOT / "configs" / "sf2_data.json"
DEFAULT_STATES_DIR = PROJECT_ROOT / "configs" / "states"

# Grilla de selección de personajes (fila, columna) e ID interno en RAM
CHARACTER_SPECS: Dict[str, Tuple[int, int, int]] = {
    "Ryu": (0, 0, 0),
    "Honda": (0, 1, 1),
    "Blanka": (0, 2, 2),
    "Guile": (0, 3, 3),
    "Ken": (1, 0, 4),
    "ChunLi": (1, 1, 5),
    "Zangief": (1, 2, 6),
    "Dhalsim": (1, 3, 7),
    "Balrog": (0, 4, 8),
    "Vega": (0, 5, 9),
    "Sagat": (1, 4, 10),
    "Bison": (1, 5, 11),
}

ID_TO_NAME: Dict[int, str] = {spec[2]: name for name, spec in CHARACTER_SPECS.items()}


def _make_action(btn_map: Dict[str, int], name: str | None = None) -> List[int]:
    act = [0] * 12
    if name is not None:
        act[btn_map[name]] = 1
    return act


def _press(env: stable_retro.RetroEnv, btn_map: Dict[str, int], name: str, hold: int = 4, wait: int = 20) -> dict:
    act = _make_action(btn_map, name)
    noop = _make_action(btn_map, None)
    info: dict = {}
    for _ in range(hold):
        _, _, _, _, info = env.step(act)
    for _ in range(wait):
        _, _, _, _, info = env.step(noop)
    return info


def _step_noop(env: stable_retro.RetroEnv, btn_map: Dict[str, int], frames: int) -> dict:
    noop = _make_action(btn_map, None)
    info: dict = {}
    for _ in range(frames):
        _, _, _, _, info = env.step(noop)
    return info


def _move_cursor_to_character(env: stable_retro.RetroEnv, btn_map: Dict[str, int], char_name: str) -> None:
    row, col, _ = CHARACTER_SPECS[char_name]
    for _ in range(row):
        _press(env, btn_map, "DOWN", hold=3, wait=10)
    for _ in range(col):
        _press(env, btn_map, "RIGHT", hold=3, wait=10)


def _find_playable_round_start(env: stable_retro.RetroEnv, btn_map: Dict[str, int], max_frames: int = 550) -> Tuple[bytes, dict]:
    """Avanza desde la carga del escenario hasta el primer frame con control activo tras 'FIGHT!'."""
    noop = _make_action(btn_map, None)
    right = _make_action(btn_map, "RIGHT")

    for _ in range(max_frames):
        _, _, _, _, info = env.step(noop)
        if info.get("health") != 176 or info.get("enemy_health") != 176:
            continue
        if info.get("timer", 0) < 98:
            continue

        candidate_state = env.em.get_state()
        x_before = info["x"]
        test_info = info
        for _ in range(2):
            _, _, _, _, test_info = env.step(right)
        env.em.set_state(candidate_state)

        if (
            test_info["x"] != x_before
            and test_info["health"] == 176
            and test_info["enemy_health"] == 176
        ):
            return candidate_state, info

    raise RuntimeError("No se detectó inicio jugable limpio en el límite de frames esperado.")


def _prepare_demo_state_with_difficulty(
    env: stable_retro.RetroEnv,
    btn_map: Dict[str, int],
    title_state: bytes,
    difficulty: int,
) -> bytes:
    """Configura la dificultad (1 a 8) en OPTION MODE y avanza hasta la pelea demo para muestrear rivales."""
    if not (1 <= difficulty <= 8):
        raise ValueError(f"La dificultad debe estar entre 1 y 8 (recibido: {difficulty}).")

    expected_diff_ram = difficulty - 1
    env.em.set_state(title_state)

    # 1. Desde pantalla de título -> abrir menú CHAMPION / HYPER / OPTIONS y entrar a OPTIONS
    _press(env, btn_map, "START", hold=5, wait=90)
    _press(env, btn_map, "DOWN", hold=4, wait=25)
    _press(env, btn_map, "DOWN", hold=4, wait=25)
    _press(env, btn_map, "START", hold=5, wait=120)

    # 2. Llevar DIFFICULTY a 1 estrella (RAM=0) y subir a (difficulty - 1)
    for _ in range(8):
        _press(env, btn_map, "LEFT", hold=3, wait=12)
    for _ in range(expected_diff_ram):
        _press(env, btn_map, "RIGHT", hold=3, wait=12)

    # 3. Salir de OPTION MODE (vuelve a la intro conservando la dificultad) y avanzar hasta el demo
    _press(env, btn_map, "START", hold=5, wait=100)
    info = _step_noop(env, btn_map, 2500)
    if int(info["difficulty"]) != expected_diff_ram:
        raise AssertionError(
            f"No se conservó la dificultad en RAM tras salir de OPTIONS: {info['difficulty']} != {expected_diff_ram}"
        )
    return env.em.get_state()


def generate_states_for_difficulty(
    env: stable_retro.RetroEnv,
    btn_map: Dict[str, int],
    title_state: bytes,
    difficulty: int,
    agent_name: str,
    target_opponents: List[str],
    output_dir: Path,
) -> List[Path]:
    """Genera los archivos .state para una dificultad dada (1 a 8)."""
    diff_demo_state = _prepare_demo_state_with_difficulty(env, btn_map, title_state, difficulty)
    expected_diff_ram = difficulty - 1
    expected_agent_id = CHARACTER_SPECS[agent_name][2]
    remaining = {CHARACTER_SPECS[opp][2]: opp for opp in target_opponents}
    saved_paths: List[Path] = []

    # El demo de atracción avanza el PRNG cada frame, permitiendo obtener distintos rivales en Stage 1
    for demo_offset in range(0, 1800, 13):
        if not remaining:
            break

        env.em.set_state(diff_demo_state)
        _step_noop(env, btn_map, demo_offset)

        # Demo -> Título -> CHAMPION -> GAME START -> PLAYER SELECT
        _press(env, btn_map, "START", hold=5, wait=90)
        _press(env, btn_map, "START", hold=5, wait=90)
        _press(env, btn_map, "START", hold=5, wait=90)
        _press(env, btn_map, "START", hold=5, wait=140)

        # Elegir personaje del agente (P1) con botón X (Light Punch, color clásico)
        _move_cursor_to_character(env, btn_map, agent_name)
        info = _press(env, btn_map, "X", hold=5, wait=580)

        enemy_id = int(info.get("enemy_char_id", -1))
        if enemy_id not in remaining:
            continue

        opp_name = remaining.pop(enemy_id)
        playable_state, start_info = _find_playable_round_start(env, btn_map)

        # Verificaciones estrictas de integridad en RAM
        if int(start_info["difficulty"]) != expected_diff_ram:
            raise AssertionError(
                f"Dificultad RAM inesperada: {start_info['difficulty']} != {expected_diff_ram}"
            )
        if int(start_info["agent_char_id"]) != expected_agent_id:
            raise AssertionError(
                f"Agente RAM inesperado: {start_info['agent_char_id']} != {expected_agent_id}"
            )
        if int(start_info["enemy_char_id"]) != enemy_id:
            raise AssertionError(
                f"Rival RAM inesperado: {start_info['enemy_char_id']} != {enemy_id}"
            )

        out_file = output_dir / f"Champion.Level{difficulty}.{agent_name}Vs{opp_name}.state"
        with gzip.open(out_file, "wb") as fh:
            fh.write(playable_state)

        saved_paths.append(out_file)
        print(
            f"  [OK] {out_file.name} | diff={difficulty} (ram={start_info['difficulty']}) | "
            f"HP={start_info['health']}/{start_info['enemy_health']} | "
            f"pos=({start_info['x']},{start_info['enemy_x']}) | timer={start_info['timer']}"
        )

    if remaining:
        missing = ", ".join(remaining.values())
        raise RuntimeError(f"No se lograron encontrar todos los rivales en Stage 1; faltaron: {missing}")

    return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera archivos .state verificados para SF2 Special Champion Edition"
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="Ryu",
        choices=list(CHARACTER_SPECS.keys()),
        help="Personaje controlado por el agente P1 (default: Ryu)",
    )
    parser.add_argument(
        "--opponents",
        type=str,
        nargs="+",
        default=["Guile", "Ken", "ChunLi", "Zangief"],
        choices=[k for k, v in CHARACTER_SPECS.items() if v[2] < 8],
        help="Lista de rivales CPU en Stage 1 (World Warriors)",
    )
    parser.add_argument(
        "--difficulties",
        type=int,
        nargs="+",
        default=[1, 4, 7],
        help="Dificultades de CPU (1 a 8 estrellas, default: 1 4 7)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_STATES_DIR,
        help="Directorio destino para los archivos .state",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    env = stable_retro.make(
        game=GAME_ID,
        state=stable_retro.State.NONE,
        info=str(DEFAULT_DATA_JSON),
        use_restricted_actions=stable_retro.Actions.ALL,
        render_mode=None,
    )
    try:
        env.reset()
        btn_map = {name: idx for idx, name in enumerate(env.buttons)}
        print("[+] Avanzando arranque inicial hasta la pantalla de título...")
        _step_noop(env, btn_map, 2100)
        title_state = env.em.get_state()

        total_saved: List[Path] = []
        for diff in args.difficulties:
            print(f"\n[+] Generando estados para Dificultad {diff} ({args.agent} vs {args.opponents}):")
            paths = generate_states_for_difficulty(
                env=env,
                btn_map=btn_map,
                title_state=title_state,
                difficulty=diff,
                agent_name=args.agent,
                target_opponents=args.opponents,
                output_dir=args.output_dir,
            )
            total_saved.extend(paths)

        print(f"\n[OK] Se generaron exitosamente {len(total_saved)} archivos .state en {args.output_dir}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
