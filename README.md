# hadoken-rl

Agente autónomo para **Street Fighter II** entrenado mediante aprendizaje por refuerzo profundo (*Deep Reinforcement Learning*) utilizando **Stable-Retro**, **Gymnasium**, **Stable-Baselines3 (PPO)**, **PyTorch** y **TensorBoard**.

> **Documento rector del proyecto:** Revisa [`contexto_proyecto_street_fighter_rl.md`](./contexto_proyecto_street_fighter_rl.md) para consultar el alcance definido, las decisiones pendientes y la propuesta de distribución técnica de módulos entre los integrantes del equipo (Sección 8.1).

## Estructura del repositorio

```text
hadoken-rl/
├── configs/                  # Configuraciones experimentales reproducibles
├── roms/                     # Carpeta local para ROMs (ignorada en Git)
├── scripts/                  # Importación de ROM, smoke tests headless y profiling FPS
├── src/hadoken_rl/
│   ├── envs/                 # Entorno base Retro/Gymnasium, wrappers RAM, acciones y reward
│   ├── agents/               # Rainbow DQN (PyTorch), agente aleatorio y utilidades de logging
│   └── evaluation/           # Win Rate, ΔHP neto, TensorBoard, benchmarks y reporte comparativo
└── tests/                    # Pruebas de validación de interfaces por separado
```

## Instalación del entorno (Python 3.12)

Puedes configurar el entorno con **`uv`** (recomendado, usa `uv.lock` para reproducibilidad exacta) o con **`venv` + `pip`** tradicional usando `requirements.txt`.

### Opción A: Usando `uv` (recomendado)

```bash
# Instala Python 3.12 (si no está instalado), crea .venv e instala todo el proyecto editable
uv sync

# Ejecutar cualquier script dentro del entorno
uv run python -c "import hadoken_rl, stable_retro, gymnasium, stable_baselines3, torch; print('OK')"
```

### Opción B: Usando `venv` + `pip` (`requirements.txt`)

Asegúrate de usar **Python 3.12**:

```bash
python3.12 -m venv .venv

# En Linux / WSL / macOS:
source .venv/bin/activate
# En Windows (PowerShell):
# .\.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```

## Importación local de la ROM

Por derechos de autor, **nunca subir archivos de ROM al repositorio** (la carpeta `roms/` está en `.gitignore`).
Para importar localmente la ROM compatible con `StreetFighterIISpecialChampionEdition-Genesis-v0`:

1. Coloca el archivo de la ROM en la carpeta local `roms/`.
2. Ejecuta:
   ```bash
   # Con uv:
   uv run python -m stable_retro.import ./roms

   # O con el venv activado:
   python -m stable_retro.import ./roms
   ```

## Verificación de infraestructura, estados y *profiling* de FPS

Una vez importada localmente la ROM (`Street Fighter II' - Special Champion Edition (USA)`, SHA-1 `a5aad1d108046d9388e33247610dafb4c6516e0b`), puedes validar el entorno base y medir el escalamiento paralelo:

```bash
# 1. Smoke test headless (valida los 12 estados .state, rotación de rivales y RewardWrapper)
uv run python scripts/smoke_test_env.py

# 2. Pruebas unitarias de BaseEnv y SubprocVecEnv
uv run python tests/test_base_and_vec_env.py

# 3. Profiling de escalamiento paralelo ([1, 2, 4, 8, 12] procesos -> reports/fps_profile.json)
uv run python scripts/profile_fps.py

# 4. (Opcional) Regenerar o agregar estados .state para otros personajes o rivales
uv run python scripts/generate_states.py --agent Ryu --opponents Guile Ken ChunLi Zangief --difficulties 1 4 7
```

## Comparativa de métodos: PPO vs Rainbow DQN vs Aleatorio

Los tres métodos comparten entorno, presupuesto de timesteps, semilla y métricas.

### Entrenamiento

```bash
# Un solo método
python scripts/train.py --algo ppo     --timesteps 100000
python scripts/train.py --algo rainbow --timesteps 100000
python scripts/train.py --algo random  --timesteps 100000   # solo juega; dibuja la línea base

# Los tres con el mismo presupuesto
python scripts/train.py --algo all --timesteps 100000
```

Checkpoints generados en `checkpoints/`: `ppo_street_fighter_final.zip`, `rainbow_street_fighter_final.pt`.

### Curvas en TensorBoard

Cada método escribe los mismos tags (`combat/win_rate_window`, `combat/net_hp_diff_mean`,
`combat/damage_dealt_mean`, `combat/damage_taken_mean`, `rollout/ep_rew_mean`, ...) en
`logs/tensorboard/PPO_n`, `Rainbow_n` y `Random_n`, por lo que se superponen en un solo panel:

```bash
tensorboard --logdir logs/tensorboard
```

### Benchmark comparativo

```bash
python scripts/evaluate.py \
    --ppo-path checkpoints/ppo_street_fighter_final.zip \
    --rainbow-path checkpoints/rainbow_street_fighter_final.pt \
    --episodes 100 --output-dir reports
```

Genera en `reports/`: `comparison_results.json`, `comparison_results.csv`,
`comparison_table.md` y `comparison_plot.png` (este último requiere `matplotlib`,
opcional: `uv add matplotlib` o `pip install matplotlib`). El agente aleatorio se incluye
siempre salvo que se use `--no-random`.

> `--output` fue reemplazado por `--output-dir` y `--model-path` queda como alias obsoleto de `--ppo-path`.

### Rainbow DQN

Implementación propia (`src/hadoken_rl/agents/rainbow_dqn.py`) con los seis componentes:
Double DQN, Dueling, Noisy Nets, retornos n-step, Prioritized Replay y distribución C51.
Si se modifican `alpha`, `beta` o `win_reward` del reward wrapper, ajustar `v_min`/`v_max`
(por defecto ±250) para que el soporte de la distribución cubra el rango de retornos.
