"""Profiling reproducible de rendimiento (SPS / FPS) y escalado hacia 1.000.000 de timesteps.

Compara el throughput de recolección de experiencia entre distintos números de procesos
paralelos (por defecto [1, 2, 4, 8, 12]) usando SubprocVecEnv y exporta métricas reales
a JSON para documentar el escalamiento de infraestructura.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hadoken_rl.envs.vec_env import make_vec_sf_env


def profile_worker_count(
    n_envs: int,
    steps_per_env: int,
    difficulty: int,
    frame_skip: int,
    seed: int = 42,
) -> Dict[str, Any]:
    """Mide SPS (decisiones RL/s) y FPS emulados para una cantidad dada de entornos paralelos."""
    use_subproc = n_envs > 1
    vec_env = make_vec_sf_env(
        n_envs=n_envs,
        difficulty=difficulty,
        seed=seed,
        use_subproc=use_subproc,
        frame_skip=frame_skip,
        obs_type="ram",
    )
    try:
        vec_env.reset()
        # Pre-muestrear acciones para medir el throughput real del entorno vectorizado e IPC
        actions = np.random.randint(0, 2, size=(steps_per_env, n_envs, 12), dtype=np.int8)

        # Calentamiento corto (20 iteraciones)
        for i in range(min(20, steps_per_env)):
            vec_env.step(actions[i])

        episodes_done = 0
        t0 = time.perf_counter()
        for i in range(steps_per_env):
            _, _, dones, _ = vec_env.step(actions[i])
            episodes_done += int(np.sum(dones))
        elapsed = time.perf_counter() - t0
    finally:
        vec_env.close()

    total_timesteps = steps_per_env * n_envs
    sps = total_timesteps / elapsed
    effective_fps = sps * frame_skip
    est_min_500k = (500_000 / sps) / 60.0
    est_min_1m = (1_000_000 / sps) / 60.0

    return {
        "n_envs": n_envs,
        "backend": "SubprocVecEnv" if use_subproc else "DummyVecEnv",
        "steps_per_env": steps_per_env,
        "total_timesteps": total_timesteps,
        "frame_skip": frame_skip,
        "elapsed_sec": round(elapsed, 3),
        "timesteps_per_sec_sps": round(sps, 1),
        "emulated_frames_per_sec_fps": round(effective_fps, 1),
        "episodes_completed": episodes_done,
        "est_env_minutes_for_500k_steps": round(est_min_500k, 2),
        "est_env_minutes_for_1m_steps": round(est_min_1m, 2),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profiling de FPS/SPS con SubprocVecEnv en SF2")
    parser.add_argument(
        "--workers",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 12],
        help="Lista de n_envs a evaluar (default: 1 2 4 8 12)",
    )
    parser.add_argument(
        "--steps-per-env",
        type=int,
        default=500,
        help="Pasos vectorizados por cada entorno durante la medición (default: 500)",
    )
    parser.add_argument("--difficulty", type=int, default=4, help="Dificultad de CPU (default: 4)")
    parser.add_argument("--frame-skip", type=int, default=4, help="Frame skip por paso (default: 4)")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/fps_profile.json"),
        help="Ruta para guardar los resultados medidos en JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 78)
    print(
        f"Profiling de Infraestructura SF2 | difficulty={args.difficulty} | "
        f"frame_skip={args.frame_skip} | steps_per_env={args.steps_per_env}"
    )
    print("=" * 78)

    runs: List[Dict[str, Any]] = []
    baseline_sps: float | None = None

    for n_envs in args.workers:
        res = profile_worker_count(
            n_envs=n_envs,
            steps_per_env=args.steps_per_env,
            difficulty=args.difficulty,
            frame_skip=args.frame_skip,
        )
        if baseline_sps is None:
            baseline_sps = res["timesteps_per_sec_sps"]
        speedup = res["timesteps_per_sec_sps"] / baseline_sps if baseline_sps > 0 else 1.0
        res["speedup_vs_first"] = round(speedup, 2)
        runs.append(res)

        print(
            f"  n_envs={n_envs:2d} ({res['backend']:13s}) -> "
            f"{res['timesteps_per_sec_sps']:7.1f} SPS | "
            f"{res['emulated_frames_per_sec_fps']:8.1f} FPS | "
            f"speedup={speedup:4.2f}x | "
            f"1M steps (solo env) ~ {res['est_env_minutes_for_1m_steps']:.2f} min"
        )

    best = max(runs, key=lambda r: r["timesteps_per_sec_sps"])
    summary = {
        "system": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "processor": platform.processor(),
        },
        "best_n_envs": best["n_envs"],
        "best_sps": best["timesteps_per_sec_sps"],
        "best_emulated_fps": best["emulated_frames_per_sec_fps"],
        "results": runs,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("=" * 78)
    print(
        f"[OK] Mejor configuración medida: n_envs={best['n_envs']} con "
        f"{best['timesteps_per_sec_sps']} SPS ({best['emulated_frames_per_sec_fps']} FPS emulados)."
    )
    print(f"[OK] Reporte guardado en: {args.output.resolve()}")


if __name__ == "__main__":
    main()
