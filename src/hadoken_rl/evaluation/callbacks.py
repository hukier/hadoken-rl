"""Callbacks personalizados para Stable-Baselines3 y registro en TensorBoard."""

from collections import deque
from typing import Optional
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class StreetFighterMetricsCallback(BaseCallback):
    """
    Callback que extrae las métricas del RewardWrapper al finalizar cada episodio
    y las registra en TensorBoard:
      - combat/win_rate: Tasa de victoria en una ventana móvil de combates
      - combat/net_hp_diff_mean: Diferencial de vida promedio (HP_agente - HP_rival)
      - combat/damage_dealt_mean: Daño total infligido promedio por combate
      - combat/damage_taken_mean: Daño total recibido promedio por combate
    """

    def __init__(
        self,
        window_size: int = 50,
        log_freq: int = 1,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.window_size = window_size
        self.log_freq = log_freq

        # Buffers circulares para almacenar las métricas de los combates recientes
        self.win_history = deque(maxlen=window_size)
        self.net_hp_history = deque(maxlen=window_size)
        self.damage_dealt_history = deque(maxlen=window_size)
        self.damage_taken_history = deque(maxlen=window_size)

        self.episode_count = 0

    def _on_step(self) -> bool:
        # En Stable-Baselines3, 'dones' e 'infos' están disponibles en self.locals
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")

        if dones is None or infos is None:
            return True

        # Compatible con entornos vectorizados (SubprocVecEnv o DummyVecEnv)
        for done, info in zip(dones, infos):
            if not done:
                continue

            self.episode_count += 1

            # 1. Registrar victoria / derrota
            if "is_win" in info:
                self.win_history.append(1.0 if info["is_win"] else 0.0)

            # 2. Registrar diferencial de vida al final del asalto (K.O. o timeout)
            if "net_hp_differential" in info:
                self.net_hp_history.append(float(info["net_hp_differential"]))

            # 3. Registrar acumulados de daño
            if "damage_dealt" in info:
                self.damage_dealt_history.append(float(info["damage_dealt"]))
            if "damage_taken" in info:
                self.damage_taken_history.append(float(info["damage_taken"]))

            # Escribir métricas consolidadas en el logger de TensorBoard
            if self.episode_count % self.log_freq == 0 and len(self.win_history) > 0:
                win_rate = float(np.mean(self.win_history)) * 100.0
                mean_hp_diff = float(np.mean(self.net_hp_history))

                self.logger.record("combat/win_rate_window", win_rate)
                self.logger.record("combat/net_hp_diff_mean", mean_hp_diff)

                if len(self.damage_dealt_history) > 0:
                    self.logger.record(
                        "combat/damage_dealt_mean",
                        float(np.mean(self.damage_dealt_history)),
                    )
                if len(self.damage_taken_history) > 0:
                    self.logger.record(
                        "combat/damage_taken_mean",
                        float(np.mean(self.damage_taken_history)),
                    )

                self.logger.record("combat/episodes_completed", self.episode_count)

        return True

class CombatMetricsTracker:
    """
    Acumulador de métricas de combate independiente de Stable-Baselines3.

    Replica exactamente las ventanas móviles y los tags de TensorBoard de
    `StreetFighterMetricsCallback`, para que Rainbow DQN y el agente aleatorio
    (que no usan callbacks de SB3) publiquen las mismas curvas que PPO y
    puedan superponerse en TensorBoard.
    """

    def __init__(self, window_size: int = 50):
        self.win_history = deque(maxlen=window_size)
        self.net_hp_history = deque(maxlen=window_size)
        self.damage_dealt_history = deque(maxlen=window_size)
        self.damage_taken_history = deque(maxlen=window_size)
        self.episode_count = 0

    def update(self, info: dict) -> None:
        """Registrar el `info` final de un episodio terminado."""
        self.episode_count += 1
        if "is_win" in info:
            self.win_history.append(1.0 if info["is_win"] else 0.0)
        if "net_hp_differential" in info:
            self.net_hp_history.append(float(info["net_hp_differential"]))
        if "damage_dealt" in info:
            self.damage_dealt_history.append(float(info["damage_dealt"]))
        if "damage_taken" in info:
            self.damage_taken_history.append(float(info["damage_taken"]))

    def summary(self) -> dict:
        """Valores actuales listos para `writer.add_scalar(tag, valor, paso)`."""
        out = {}
        if len(self.win_history) > 0:
            out["combat/win_rate_window"] = float(np.mean(self.win_history)) * 100.0
        if len(self.net_hp_history) > 0:
            out["combat/net_hp_diff_mean"] = float(np.mean(self.net_hp_history))
        if len(self.damage_dealt_history) > 0:
            out["combat/damage_dealt_mean"] = float(np.mean(self.damage_dealt_history))
        if len(self.damage_taken_history) > 0:
            out["combat/damage_taken_mean"] = float(np.mean(self.damage_taken_history))
        if out:
            out["combat/episodes_completed"] = self.episode_count
        return out
