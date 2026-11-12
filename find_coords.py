import stable_retro
import numpy as np

def main():
    print("Iniciando escaneo de coordenadas (X e Y)...")
    env = stable_retro.make(
        game="StreetFighterIISpecialChampionEdition-Genesis-v0", 
        state="Champion.Level1.RyuVsGuile"
    )
    env.reset()
    
    # El action_space de Genesis tiene 12 botones. 
    # Índices: [B, A, MODE, START, UP, DOWN, LEFT, RIGHT, C, Y, X, Z]
    nada = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    derecha = np.copy(nada); derecha[7] = 1  # Presionar RIGHT
    arriba = np.copy(nada); arriba[4] = 1    # Presionar UP
    
    ram_inicial = env.get_ram()
    
    # 1. Caminar hacia la derecha
    for _ in range(20): env.step(derecha)
    ram_caminando = env.get_ram()
    
    # 2. Saltar verticalmente
    for _ in range(30): env.step(arriba)
    ram_saltando = env.get_ram()
    
    # Buscamos +/- 256 bytes alrededor de la vida de Ryu (32832)
    rango_busqueda = range(32832 - 256, 32832 + 256)
    
    print("\n--- POSIBLES COORDENADAS X ---")
    for addr in rango_busqueda:
        # X cambia al caminar horizontalmente, pero no cambia al saltar verticalmente
        if ram_inicial[addr] != ram_caminando[addr] and ram_caminando[addr] == ram_saltando[addr]:
            print(f"Dirección: {addr} | Quieto: {ram_inicial[addr]} -> Movido a la derecha: {ram_caminando[addr]}")
            
    print("\n--- POSIBLES COORDENADAS Y ---")
    for addr in rango_busqueda:
        # Y NO cambia al caminar, pero cambia mucho al saltar
        if ram_inicial[addr] == ram_caminando[addr] and ram_caminando[addr] != ram_saltando[addr]:
            print(f"Dirección: {addr} | En suelo: {ram_caminando[addr]} -> En el aire: {ram_saltando[addr]}")

    env.close()

if __name__ == "__main__":
    main()