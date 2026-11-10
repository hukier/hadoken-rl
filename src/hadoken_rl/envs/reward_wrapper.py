"""Wrapper de Gymnasium para cálculo de recompensa y tracking de métricas de combate."""

import gymnasium as gym
from typing import Any, Dict, Tuple, Optional


class StreetFighterRewardWrapper(gym.Wrapper):
    """
    Wrapper que implementa Reward Shaping para Street Fighter II:
    R_t = alpha * dHP_oponente - beta * dHP_agente + R_resultado + R_distancia
    """

    def __init__(
        self,
        env: gym.Env,
        alpha: float = 1.0,
        beta: float = 1.0,
        win_reward: float = 50.0,
        loss_penalty: float = 50.0,
        distance_weight: float = 0.001,
        agent_hp_key: str = "health",
        enemy_hp_key: str = "enemy_health",
        agent_x_key: str = "x",
        enemy_x_key: str = "enemy_x",
    ):
        super().__init__(env)
        self.alpha = alpha
        self.beta = beta
        self.win_reward = win_reward
        self.loss_penalty = loss_penalty
        self.distance_weight = distance_weight

        # Nombres de variables según el data.json de la ROM en Stable-Retro
        self.agent_hp_key = agent_hp_key
        self.enemy_hp_key = enemy_hp_key
        self.agent_x_key = agent_x_key
        self.enemy_x_key = enemy_x_key

        # Variables internas de estado
        self.prev_agent_hp: Optional[float] = None
        self.prev_enemy_hp: Optional[float] = None
        self.total_agent_damage_taken = 0.0
        self.total_enemy_damage_dealt = 0.0

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)

        # Inicialización de barras de vida desde el diccionario info
        self.prev_agent_hp = float(info.get(self.agent_hp_key, 176.0))
        self.prev_enemy_hp = float(info.get(self.enemy_hp_key, 176.0))
        self.total_agent_damage_taken = 0.0
        self.total_enemy_damage_dealt = 0.0

        return obs, info

    def step(
        self, action: Any
    ) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        # 1. Obtención de variables actuales de HP
        curr_agent_hp = float(info.get(self.agent_hp_key, self.prev_agent_hp))
        curr_enemy_hp = float(info.get(self.enemy_hp_key, self.prev_enemy_hp))

        # 2. Variaciones de vida (daño neto instantáneo)
        # delta > 0 indica que la barra de vida disminuyó (recibió daño)
        d_agent = max(0.0, self.prev_agent_hp - curr_agent_hp)
        d_enemy = max(0.0, self.prev_enemy_hp - curr_enemy_hp)

        self.total_agent_damage_taken += d_agent
        self.total_enemy_damage_dealt += d_enemy

        # 3. Componente de daño ponderado
        shaping_reward = (self.alpha * d_enemy) - (self.beta * d_agent)

        # 4. Componente de proximidad espacial (R_distancia)
        r_distance = 0.0
        if self.agent_x_key in info and self.enemy_x_key in info:
            distance = abs(float(info[self.agent_x_key]) - float(info[self.enemy_x_key]))
            # Incentivo suave por presionar al rival (reducir distancia horizontal)
            r_distance = -self.distance_weight * distance

        # 5. Componente terminal (Victoria o Derrota)
        r_resultado = 0.0
        is_win = False
        if terminated or truncated:
            # En SF2 la vida máxima suele ser 176; si el rival llega a 0 es K.O.
            if curr_enemy_hp <= 0 and curr_agent_hp > 0:
                r_resultado = self.win_reward
                is_win = True
            elif curr_agent_hp <= 0 and curr_enemy_hp > 0:
                r_resultado = -self.loss_penalty
                is_win = False
            else:
                # Decisión por tiempo: gana quien conserve mayor porcentaje de vida
                is_win = curr_agent_hp > curr_enemy_hp
                r_resultado = self.win_reward if is_win else -self.loss_penalty

        # Recompensa total consolidada
        custom_reward = shaping_reward + r_resultado + r_distance

        # Actualizar históricos para el siguiente timestep
        self.prev_agent_hp = curr_agent_hp
        self.prev_enemy_hp = curr_enemy_hp

        # Registrar métricas en info para que los Callbacks las capturen
        info["step_reward"] = custom_reward
        info["net_hp_differential"] = curr_agent_hp - curr_enemy_hp
        info["damage_dealt"] = self.total_enemy_damage_dealt
        info["damage_taken"] = self.total_agent_damage_taken
        if terminated or truncated:
            info["is_win"] = is_win

        return obs, custom_reward, terminated, truncated, info