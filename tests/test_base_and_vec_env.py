"""Pruebas unitarias e integración real para StreetFighterBaseEnv y make_vec_sf_env."""

from __future__ import annotations

import numpy as np
import stable_retro

from hadoken_rl.envs.base_env import GAME_ID, discover_state_files, make_base_sf_env
from hadoken_rl.envs.reward_wrapper import StreetFighterRewardWrapper
from hadoken_rl.envs.vec_env import make_vec_sf_env


def _is_rom_imported() -> bool:
    try:
        stable_retro.data.get_romfile_path(GAME_ID)
        return True
    except FileNotFoundError:
        return False


def test_discover_state_files() -> None:
    for diff in (1, 4, 7):
        files = discover_state_files(difficulty=diff, agent="Ryu")
        assert len(files) == 4, f"Se esperaban 4 estados en dificultad {diff}, obtenidos: {len(files)}"
        stems = [f.stem for f in files]
        for opp in ("ChunLi", "Guile", "Ken", "Zangief"):
            assert f"Champion.Level{diff}.RyuVs{opp}" in stems


def test_base_env_reset_step_and_rotation() -> None:
    if not _is_rom_imported():
        print("⚠️ ROM local no importada; omitiendo test de emulador en vivo.")
        return

    env = make_base_sf_env(difficulty=4, frame_skip=4, state_selection="round_robin")
    try:
        seen_opponents = []
        for _ in range(4):
            obs, info = env.reset()
            assert obs.shape == (65536,)
            assert obs.dtype == np.uint8
            assert info["difficulty_level"] == 4
            assert info["health"] == 176
            assert info["enemy_health"] == 176
            assert info["x"] == 205
            assert info["enemy_x"] == 307
            seen_opponents.append(info["enemy_name"])

        assert set(seen_opponents) == {"ChunLi", "Guile", "Ken", "Zangief"}

        # Verificar movimiento real en el paso 1 (sin frames muertos de intro)
        env.reset(options={"opponent": "Guile"})
        act_right = np.zeros(12, dtype=np.int8)
        act_right[7] = 1  # RIGHT
        _, _, term, trunc, step_info = env.step(act_right)
        assert not term and not trunc
        assert step_info["x"] > 205, f"Se esperaba avance en x desde 205, obtenido: {step_info['x']}"
    finally:
        env.close()


def test_subproc_vec_env_and_reward_wrapper_integration() -> None:
    if not _is_rom_imported():
        return

    vec_env = make_vec_sf_env(n_envs=4, difficulty=4, seed=123, use_subproc=True, frame_skip=4)
    try:
        obs = vec_env.reset()
        assert obs.shape == (4, 65536)

        # Verificar que cada subproceso inicia contra un rival distinto gracias a start_state_index=rank
        actions = np.zeros((4, 12), dtype=np.int8)
        obs, rewards, dones, infos = vec_env.step(actions)
        assert obs.shape == (4, 65536)
        assert rewards.shape == (4,)
        assert dones.shape == (4,)
        enemies = {info["enemy_name"] for info in infos}
        assert enemies == {"ChunLi", "Guile", "Ken", "Zangief"}, f"Rivales en paralelo inesperados: {enemies}"
        for info in infos:
            assert "step_reward" in info
            assert "net_hp_differential" in info
    finally:
        vec_env.close()


def test_single_round_ko_termination_clean_hp() -> None:
    if not _is_rom_imported():
        return

    env = StreetFighterRewardWrapper(make_base_sf_env(difficulty=7, frame_skip=4))
    try:
        env.reset(seed=7)
        noop = np.zeros(12, dtype=np.int8)
        terminated = False
        truncated = False
        info = {}
        while not (terminated or truncated):
            _, _, terminated, truncated, info = env.step(noop)

        assert terminated, "La ronda debía terminar por K.O. o tiempo agotado."
        assert info["health"] >= 0 and info["enemy_health"] >= 0
        assert "is_win" in info
        assert info["is_win"] is False
        assert info["net_hp_differential"] < 0
    finally:
        env.close()


if __name__ == "__main__":
    test_discover_state_files()
    test_base_env_reset_step_and_rotation()
    test_subproc_vec_env_and_reward_wrapper_integration()
    test_single_round_ko_termination_clean_hp()
    print("✅ Todos los tests de BaseEnv y SubprocVecEnv superados con éxito.")
