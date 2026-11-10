import sys
# Forzar a Python a encontrar la carpeta src
sys.path.insert(0, "src")

import stable_retro
from hadoken_rl.envs.action_wrapper import StreetFighterActionWrapper
from hadoken_rl.envs.observation_wrapper import StreetFighterRAMWrapper

def main():
    print("Iniciando emulador base...")
    # Carga el juego base importado previamente
    base_env = stable_retro.make(
        game="StreetFighterIISpecialChampionEdition-Genesis-v0",
        obs_type=stable_retro.Observations.RAM
    )
    
    print("Aplicando Wrappers...")
    env = StreetFighterActionWrapper(base_env)
    # Se le pasa el JSON que actualizamos en el paso anterior
    env = StreetFighterRAMWrapper(
        env, 
        layout_path="configs/ram_layout.json", 
        allow_unset=True
    )

    print("Espacio de acciones (Macro):", env.action_space)
    print("Espacio de observaciones (Vector):", env.observation_space)

    print("\nEjecutando reset()...")
    obs, info = env.reset()
    print("Forma de la observación inicial:", obs.shape)
    
    # La acción 12 corresponde a "HADOKEN" en tus DEFAULT_MACROS
    print("\nEjecutando step(12) [HADOKEN]...") 
    obs, reward, terminated, truncated, info = env.step(12)
    
    print("Forma de la observación post-step:", obs.shape)
    print("Recompensa obtenida:", reward)
    print("Macro registrada en info:", info.get("macro_action"))
    
    env.close()
    print("\n¡Verificación completada con éxito!")

if __name__ == "__main__":
    main()