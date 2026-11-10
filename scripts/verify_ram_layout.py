"""Descubre y valida direcciones RAM sobre la ROM real (requiere stable-retro).

1) Descubrir: juega con acciones aleatorias y lista los offsets de RAM cuyo valor coincide
   con una variable ya expuesta en `info` (p. ej. `health`):
       python scripts/verify_ram_layout.py --search health --search enemy_health
2) Validar y calibrar: con las direcciones ya escritas en configs/ram_layout.json, muestra
   para cada feature de RAM su rango observado y, si tiene `info_key`, su tasa de coincidencia:
       python scripts/verify_ram_layout.py --frames 2000
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import argparse

import numpy as np

from hadoken_rl.envs.factory import _build_base_env, add_env_args, env_config_from_args
from hadoken_rl.envs.observation_wrapper import load_layout
from hadoken_rl.envs.ram_tools import match_rate, observed_stats, search_candidates


def parse_args():
    p = argparse.ArgumentParser(description="Descubrimiento y validación de direcciones RAM")
    p.add_argument("--frames", type=int, default=1000, help="Frames a muestrear con acciones aleatorias")
    p.add_argument("--search", action="append", default=[], metavar="INFO_KEY",
                   help="Clave de info a buscar en la RAM (repetible)")
    p.add_argument("--threshold", type=float, default=0.95, help="Tasa mínima de coincidencia para listar un candidato")
    p.add_argument("--seed", type=int, default=0)
    add_env_args(p)
    return p.parse_args()


def collect(env, frames: int):
    ram, infos = [], []
    obs, info = env.reset()
    for _ in range(frames):
        obs, _, terminated, truncated, info = env.step(env.action_space.sample())
        ram.append(np.asarray(obs, dtype=np.uint8))
        infos.append(info)
        if terminated or truncated:
            obs, info = env.reset()
    return np.stack(ram), infos


def main():
    args = parse_args()
    layout = load_layout(args.ram_layout)
    base = layout["address_base"]
    env = _build_base_env(env_config_from_args(args))
    print(f"[+] Muestreando {args.frames} frames...")
    ram, infos = collect(env, args.frames)
    env.close()

    for key in args.search:
        if not all(key in i for i in infos):
            print(f"[!] '{key}' no está en info; claves disponibles: {sorted(infos[0])}")
            continue
        values = [float(i[key]) for i in infos]
        cands = search_candidates(ram, values, args.threshold)
        shown = ", ".join(hex(int(c) + base) for c in cands[:15]) or "ninguno"
        print(f"[{key}] candidatos (1 byte, >= {args.threshold:.0%} de coincidencia): {shown}")

    print("\n[+] Validación del layout actual:")
    for f in layout["features"]:
        if f.source != "ram":
            continue
        if f.address is None:
            print(f"  - {f.name}: SIN DIRECCIÓN")
            continue
        off = f.address - base
        st = observed_stats(ram, off, f.size, f.byteorder, f.signed)
        msg = f"  - {f.name} @ {hex(f.address)}: min={st['min']:.0f} max={st['max']:.0f} distintos={st['unique']}"
        if f.info_key and all(f.info_key in i for i in infos):
            ref = [float(i[f.info_key]) for i in infos]
            msg += f" | coincide con info['{f.info_key}']: {match_rate(ram, off, ref, f.size, f.byteorder, f.signed):.1%}"
        if st["max"] > f.max_value or st["min"] < f.min_value:
            msg += "  [!] fuera de [min_value, max_value]: ajustar rango"
        print(msg)


if __name__ == "__main__":
    main()
