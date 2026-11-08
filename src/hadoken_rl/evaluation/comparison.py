"""Generación de reportes comparativos entre agentes (PPO vs Rainbow DQN vs Aleatorio)."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Estructura de `results`: {nombre_agente: {"difficulty_N": {métricas del benchmark}}}
ComparisonResults = Dict[str, Dict[str, Dict[str, Any]]]

METRIC_COLUMNS = [
    ("win_rate_pct", "Win Rate (%)"),
    ("net_hp_mean", "HP neto (media)"),
    ("net_hp_std", "HP neto (desv.)"),
    ("damage_dealt_mean", "Daño infligido"),
    ("damage_taken_mean", "Daño recibido"),
    ("episode_len_mean", "Duración (pasos)"),
]


def _difficulty_keys(results: ComparisonResults) -> List[str]:
    keys = set()
    for per_diff in results.values():
        keys.update(per_diff.keys())
    return sorted(keys, key=lambda k: int(k.split("_")[-1]))


def build_rows(results: ComparisonResults) -> List[Dict[str, Any]]:
    """Aplana los resultados en filas (dificultad, agente, métricas), ordenadas por dificultad."""
    rows = []
    for diff_key in _difficulty_keys(results):
        for agent, per_diff in results.items():
            if diff_key not in per_diff:
                continue
            res = per_diff[diff_key]
            row = {"difficulty": int(diff_key.split("_")[-1]), "agent": agent}
            for key, _ in METRIC_COLUMNS:
                row[key] = res.get(key)
            row["episodes"] = res.get("episodes")
            rows.append(row)
    return rows


def format_markdown_table(results: ComparisonResults) -> str:
    """Tabla Markdown con una fila por (dificultad, agente)."""
    headers = ["Dificultad", "Agente"] + [label for _, label in METRIC_COLUMNS]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in build_rows(results):
        cells = [str(row["difficulty"]), row["agent"]] + [
            "-" if row[k] is None else f"{row[k]:.2f}" for k, _ in METRIC_COLUMNS
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def format_console_table(results: ComparisonResults) -> str:
    """Tabla de texto plano alineada, para imprimir al terminar el benchmark."""
    headers = ["Dif", "Agente"] + [label for _, label in METRIC_COLUMNS]
    body = [
        [str(r["difficulty"]), r["agent"]]
        + ["-" if r[k] is None else f"{r[k]:.2f}" for k, _ in METRIC_COLUMNS]
        for r in build_rows(results)
    ]
    widths = [max(len(h), *(len(row[i]) for row in body)) if body else len(h) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    out = [fmt.format(*headers), fmt.format(*["-" * w for w in widths])]
    out.extend(fmt.format(*row) for row in body)
    return "\n".join(out)


def save_comparison_csv(results: ComparisonResults, path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["difficulty", "agent", "episodes"] + [k for k, _ in METRIC_COLUMNS]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(build_rows(results))
    return path


def plot_comparison(results: ComparisonResults, path: Union[str, Path]) -> Optional[Path]:
    """
    Gráfico de barras agrupadas (Win Rate y HP neto por dificultad, una barra por agente).
    Devuelve None si matplotlib no está instalado (dependencia opcional).
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    diff_keys = _difficulty_keys(results)
    agents = list(results.keys())
    x = np.arange(len(diff_keys))
    width = 0.8 / max(1, len(agents))

    panels = [
        ("win_rate_pct", "Win Rate (%)", "Win Rate por dificultad"),
        ("net_hp_mean", "HP neto (agente - rival)", "Diferencial de HP neto por dificultad"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4.5))
    for ax, (metric, ylabel, title) in zip(axes, panels):
        for i, agent in enumerate(agents):
            vals = [results[agent].get(k, {}).get(metric, 0.0) or 0.0 for k in diff_keys]
            ax.bar(x + (i - (len(agents) - 1) / 2) * width, vals, width, label=agent)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Dif. {k.split('_')[-1]}" for k in diff_keys])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_comparison_report(
    results: ComparisonResults, output_dir: Union[str, Path]
) -> Dict[str, Optional[Path]]:
    """Guarda JSON, CSV, tabla Markdown y gráfico (si hay matplotlib) en `output_dir`."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / "comparison_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    md_path = out / "comparison_table.md"
    md_path.write_text(format_markdown_table(results) + "\n", encoding="utf-8")

    return {
        "json": json_path,
        "csv": save_comparison_csv(results, out / "comparison_results.csv"),
        "markdown": md_path,
        "plot": plot_comparison(results, out / "comparison_plot.png"),
    }
