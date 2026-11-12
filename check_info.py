import stable_retro

def main():
    env = stable_retro.make(
        game="StreetFighterIISpecialChampionEdition-Genesis-v0",
        state="Champion.Level1.RyuVsGuile"
    )
    obs, info = env.reset()
    print("\n[+] Variables disponibles en el diccionario 'info':")
    for key, value in info.items():
        print(f"    - {key}: {value}")
    env.close()

if __name__ == "__main__":
    main()