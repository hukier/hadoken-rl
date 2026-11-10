import sys
import json

# Forzar a Python a incluir la carpeta 'src' para encontrar tus módulos
sys.path.insert(0, "src")

from hadoken_rl.envs.observation_wrapper import load_layout, save_layout

PATH = "configs/ram_layout.json"
addr = {
    "agent_x": (0xFF8006, 2), "agent_y": (0xFF800A, 2),
    "enemy_x": (0xFF8286, 2), "enemy_y": (0xFF828A, 2), 
    "timer": (0xFF972A, 1)
}

feats = []
for f in load_layout(PATH)["features"]:
    if f.name in ("agent_animation", "enemy_animation"):
        continue
    if f.name in addr:
        f.address, f.size = addr[f.name]
    if f.name == "timer":
        f.bcd = True
    feats.append(f)

save_layout(feats, PATH)

d = json.load(open(PATH))
d["byte_swapped"] = True

with open(PATH, 'w', encoding='utf-8') as f:
    json.dump(d, f, indent=2)

print("features:", [f.name for f in feats])
print("¡Archivo actualizado correctamente!")