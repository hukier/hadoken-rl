import stable_retro
import numpy as np

def main():
    print("Iniciando emulador: Ryu se quedará quieto recibiendo golpes...")
    env = stable_retro.make(
        game="StreetFighterIISpecialChampionEdition-Genesis-v0", 
        state="Champion.Level1.RyuVsGuile"
    )
    env.reset()
    
    ram_inicial = env.get_ram()
    candidatos = {addr for addr in range(len(ram_inicial)) if ram_inicial[addr] == 176}
    
    # Creamos la acción "quedarse quieto": un vector de 12 ceros
    accion_nula = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    
    # 600 frames son ~10 segundos, suficiente para que Guile ataque
    for _ in range(600):
        env.step(accion_nula)  
        
    ram_final = env.get_ram()
    
    # Buscamos qué direcciones bajaron de 176
    cambiadas = {addr for addr in candidatos if 0 < ram_final[addr] < 176}
    
    print(f"\n[+] Direcciones que bajaron (¡Esta es la vida de Ryu / Jugador 1!):")
    for addr in cambiadas:
        print(f"    -> Decimal: {addr} (Hex: {hex(addr)}) | Vida restante: {ram_final[addr]}")
        
    # Por descarte, las que NO bajaron son las de Guile
    print(f"\n[+] Direcciones intactas (Probablemente la vida de Guile / Enemigo):")
    intactas = candidatos - cambiadas
    for addr in intactas:
        # Solo imprimimos las cercanas en memoria a las de Ryu
        if any(abs(addr - ryu_addr) < 2000 for ryu_addr in cambiadas):
             print(f"    -> Decimal: {addr} (Hex: {hex(addr)})")

    env.close()

if __name__ == "__main__":
    main()