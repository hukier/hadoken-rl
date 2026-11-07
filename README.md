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
│   ├── agents/               # Configuración y entrenamiento de PPO (y baseline)
│   └── evaluation/           # Win Rate, ΔHP neto, TensorBoard y benchmarks
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
