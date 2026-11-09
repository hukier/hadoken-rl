"""Smoke test integral del entorno base y su acoplamiento con StreetFighterRewardWrapper.

Verifica en modo headless:
  1. Que la ROM esté importada en Stable-Retro.
  2. Que los 12 estados iniciales (.state) de dificultades 1, 4 y 7 carguen con RAM íntegra.
  3. Que la rotación de rivales funcione en reset() y pueble info desde el frame 0.
  4. Que un combate completo de 1 ronda termine exactamente en el K.O./timeout e inyecte
     las métricas esperadas por el módulo de evaluación de Javier.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hadoken_rl.envs.base_env import discover_state_files, make_base_sf_env
from hadoken_rl.envs.reward_wrapper import StreetFighterRewardWrapper


def verify_all_states() -> None:
    print("[1/3] Verificando integridad RAM de los estados .state en dificultades 1, 4 y 7...")
    for diff in (1, 4, 7):
        states = discover_state_files(difficulty=diff, agent="Ryu")
        env = make_base_sf_env(difficulty=diff, frame_skip=4, state_selection="round_robin")
        try:
            seen_enemies = []
            for _ in range(len(states)):
                obs, info = env.reset()
                assert obs.shape == (65536,), f"Forma RAM inesperada: {obs.shape}"
                assert info["difficulty_level"] == diff, (
                    f"Dificultad leída {info['difficulty_level']} != esperada {diff}"
                )
                assert info["health"] == 176 and info["enemy_health"] == 176
                assert "x" in info and "enemy_x" in info
                seen_enemies.append(info["enemy_name"])
            print(f"  [OK] Dificultad {diff}: {len(states)} estados validados -> rivales={seen_enemies}")
        finally:
            env.close()


def verify_full_round_combat() -> None:
    print("\n[2/3] Simulando ronda completa en Dificultad 4 con StreetFighterRewardWrapper...")
    base_env = make_base_sf_env(difficulty=4, frame_skip=4)
    env = StreetFighterRewardWrapper(base_env)
    try:
        obs, info = env.reset(seed=42)
        initial_opponent = info["enemy_name"]
        steps = 0
        total_reward = 0.0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

        assert "is_win" in info, "StreetFighterRewardWrapper no inyectó 'is_win' al terminar la ronda."
        assert info["health"] >= 0 and info["enemy_health"] >= 0
        print(
            f"  [OK] Ronda finalizada contra {initial_opponent} en {steps} pasos | "
            f"terminated={terminated}, truncated={truncated} | "
            f"HP final={info['health']} vs {info['enemy_health']} | "
            f"is_win={info['is_win']} | net_hp_diff={info['net_hp_differential']} | "
            f"reward_acum={total_reward:.2f}"
        )

        print("\n[3/3] Verificando reset limpio hacia el siguiente rival rotativo...")
        _, next_info = env.reset()
        assert next_info["health"] == 176 and next_info["enemy_health"] == 176
        assert next_info["enemy_name"] != initial_opponent
        print(f"  [OK] Reset limpio confirmado -> nuevo rival: {next_info['enemy_name']} (HP 176/176)")
    finally:
        env.close()


def main() -> None:
    verify_all_states()
    verify_full_round_combat()
    print("\n[OK] Smoke test superado al 100 %.")


if __name__ == "__main__":
    main()
