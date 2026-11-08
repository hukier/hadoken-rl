"""
Rainbow DQN (Hessel et al., 2018) en PyTorch puro, con interfaz tipo Stable-Baselines3.

Stable-Baselines3 solo trae DQN clásico, así que se implementan aquí los seis
componentes de Rainbow sobre una política MLP (la observación del proyecto es
un vector RAM, no píxeles):

  1. Double DQN              -> la red online elige la acción, la target la evalúa
  2. Dueling networks        -> cabezas separadas de valor V(s) y ventaja A(s, a)
  3. Noisy Nets              -> exploración con ruido paramétrico (sin epsilon-greedy)
  4. N-step returns          -> retornos a n pasos (n_step)
  5. Prioritized replay      -> muestreo proporcional al error TD, con pesos IS
  6. Distributional (C51)    -> se aprende la distribución del retorno en `n_atoms` átomos

Interfaz: `RainbowDQN(env).learn(total_timesteps)`, `.predict(obs, deterministic)`,
`.save(path)` y `RainbowDQN.load(path)`. `predict` devuelve `(accion, None)` igual
que SB3, por lo que `StreetFighterBenchmark` lo evalúa sin cambios.

Nota sobre `v_min` / `v_max`: el soporte de la distribución debe cubrir el rango de
retornos posibles. Con el reward shaping del proyecto (daño ±176 por combate, ±50 por
resultado) el valor por defecto de ±250 es seguro; si se cambian `alpha`, `beta` o
`win_reward` en `StreetFighterRewardWrapper`, hay que ajustarlos.
"""

from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Tuple, Union
import math

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hadoken_rl.agents._logging import next_run_dir
from hadoken_rl.evaluation.callbacks import CombatMetricsTracker


# --------------------------------------------------------------------------- #
# Capas y red
# --------------------------------------------------------------------------- #
class NoisyLinear(nn.Module):
    """Capa lineal con ruido gaussiano factorizado (Fortunato et al., 2018)."""

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("weight_epsilon", torch.zeros(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.zeros(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-bound, bound)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    @staticmethod
    def _scaled_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self) -> None:
        eps_in = self._scaled_noise(self.in_features)
        eps_out = self._scaled_noise(self.out_features)
        self.weight_epsilon.copy_(torch.outer(eps_out, eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # En modo eval se usa solo la media (política determinista).
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight, bias = self.weight_mu, self.bias_mu
        return F.linear(x, weight, bias)


class RainbowNetwork(nn.Module):
    """MLP compartido + cabezas Dueling ruidosas que producen una distribución C51 por acción."""

    def __init__(self, obs_dim: int, n_actions: int, n_atoms: int, hidden: int = 256):
        super().__init__()
        self.n_actions = n_actions
        self.n_atoms = n_atoms

        self.features = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            NoisyLinear(hidden, hidden), nn.ReLU(), NoisyLinear(hidden, n_atoms)
        )
        self.advantage_head = nn.Sequential(
            NoisyLinear(hidden, hidden), nn.ReLU(), NoisyLinear(hidden, n_actions * n_atoms)
        )

    def forward(self, x: torch.Tensor, log: bool = False) -> torch.Tensor:
        """Retorna (batch, n_actions, n_atoms): probabilidades (o log-probabilidades)."""
        feat = self.features(x)
        value = self.value_head(feat).view(-1, 1, self.n_atoms)
        adv = self.advantage_head(feat).view(-1, self.n_actions, self.n_atoms)
        logits = value + adv - adv.mean(dim=1, keepdim=True)
        return F.log_softmax(logits, dim=2) if log else F.softmax(logits, dim=2)

    def reset_noise(self) -> None:
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


# --------------------------------------------------------------------------- #
# Replay buffer priorizado
# --------------------------------------------------------------------------- #
class PrioritizedReplayBuffer:
    """
    Replay buffer con muestreo proporcional a la prioridad (Schaul et al., 2016).

    El muestreo es estratificado sobre la suma acumulada de prioridades, O(N) por
    lote; para los tamaños de buffer de este proyecto (<= ~1e6) es suficiente y
    evita la complejidad de un sum-tree.
    """

    def __init__(self, capacity: int, obs_dim: int, alpha: float = 0.5, eps: float = 1e-6):
        self.capacity = capacity
        self.alpha = alpha
        self.eps = eps
        self.size = 0
        self.pos = 0
        self.max_priority = 1.0

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.n_steps = np.ones(capacity, dtype=np.float32)
        self.priorities = np.zeros(capacity, dtype=np.float64)  # ya elevadas a alpha

    def __len__(self) -> int:
        return self.size

    def add(self, obs, action, reward, next_obs, done, n_steps) -> None:
        i = self.pos
        self.obs[i] = obs
        self.next_obs[i] = next_obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = float(done)
        self.n_steps[i] = n_steps
        # Transición nueva: prioridad máxima para garantizar que se muestree al menos una vez.
        self.priorities[i] = self.max_priority ** self.alpha
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float, rng: np.random.Generator) -> Dict[str, Any]:
        prios = self.priorities[: self.size]
        cumsum = np.cumsum(prios)
        total = cumsum[-1]

        # Muestreo estratificado: un punto uniforme dentro de cada uno de los batch_size segmentos.
        segments = (np.arange(batch_size) + rng.random(batch_size)) / batch_size * total
        idx = np.minimum(np.searchsorted(cumsum, segments), self.size - 1)

        probs = prios[idx] / total
        weights = (self.size * probs) ** (-beta)
        weights = (weights / weights.max()).astype(np.float32)

        return {
            "idx": idx,
            "obs": self.obs[idx],
            "next_obs": self.next_obs[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "dones": self.dones[idx],
            "n_steps": self.n_steps[idx],
            "weights": weights,
        }

    def update_priorities(self, idx: np.ndarray, td_errors: np.ndarray) -> None:
        raw = np.abs(td_errors) + self.eps
        self.priorities[idx] = raw ** self.alpha
        self.max_priority = max(self.max_priority, float(raw.max()))


# --------------------------------------------------------------------------- #
# Agente
# --------------------------------------------------------------------------- #
class RainbowDQN:
    """Agente Rainbow DQN para observaciones vectoriales y acciones discretas."""

    def __init__(
        self,
        env: Optional[gym.Env] = None,
        *,
        observation_space: Optional[gym.spaces.Space] = None,
        action_space: Optional[gym.spaces.Space] = None,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        n_step: int = 3,
        buffer_size: int = 100_000,
        batch_size: int = 64,
        learning_starts: int = 2_000,
        train_freq: int = 4,
        target_update_interval: int = 2_000,
        n_atoms: int = 51,
        v_min: float = -250.0,
        v_max: float = 250.0,
        hidden_size: int = 256,
        per_alpha: float = 0.5,
        per_beta_start: float = 0.4,
        noisy_std: float = 0.5,
        max_grad_norm: float = 10.0,
        tensorboard_log: Optional[Union[str, Path]] = None,
        metrics_window: int = 50,
        seed: Optional[int] = None,
        device: str = "auto",
        verbose: int = 1,
    ):
        self.env = env
        self.observation_space = observation_space if observation_space is not None else env.observation_space
        self.action_space = action_space if action_space is not None else env.action_space
        if not isinstance(self.action_space, gym.spaces.Discrete):
            raise ValueError("RainbowDQN requiere un espacio de acciones Discrete.")
        if self.observation_space.shape is None or len(self.observation_space.shape) == 0:
            raise ValueError("RainbowDQN requiere observaciones tipo Box con shape definido.")

        self.obs_shape = tuple(self.observation_space.shape)
        self.obs_dim = int(np.prod(self.obs_shape))
        self.n_actions = int(self.action_space.n)

        self.hparams = dict(
            learning_rate=learning_rate, gamma=gamma, n_step=n_step, buffer_size=buffer_size,
            batch_size=batch_size, learning_starts=learning_starts, train_freq=train_freq,
            target_update_interval=target_update_interval, n_atoms=n_atoms, v_min=v_min,
            v_max=v_max, hidden_size=hidden_size, per_alpha=per_alpha,
            per_beta_start=per_beta_start, noisy_std=noisy_std, max_grad_norm=max_grad_norm,
        )
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.n_step = n_step
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.target_update_interval = target_update_interval
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.per_beta_start = per_beta_start
        self.max_grad_norm = max_grad_norm
        self.tensorboard_log = tensorboard_log
        self.metrics_window = metrics_window
        self.verbose = verbose
        self.seed = seed
        self.num_timesteps = 0

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            self.action_space.seed(seed)
        self.rng = np.random.default_rng(seed)

        self.device = torch.device(
            "cuda" if (device == "auto" and torch.cuda.is_available()) else ("cpu" if device == "auto" else device)
        )

        self.online_net = RainbowNetwork(self.obs_dim, self.n_actions, n_atoms, hidden_size).to(self.device)
        self.target_net = RainbowNetwork(self.obs_dim, self.n_actions, n_atoms, hidden_size).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        for p in self.target_net.parameters():
            p.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=learning_rate, eps=1.5e-4)

        self.support = torch.linspace(v_min, v_max, n_atoms, device=self.device)
        self.delta_z = (v_max - v_min) / (n_atoms - 1)

        self.replay = PrioritizedReplayBuffer(buffer_size, self.obs_dim, alpha=per_alpha)

    # ------------------------------------------------------------------ #
    # Inferencia
    # ------------------------------------------------------------------ #
    def predict(self, obs: Any, deterministic: bool = True) -> Tuple[Any, None]:
        """
        Acción greedy respecto al valor esperado Q(s,a) = sum_z z * p(z|s,a).

        `deterministic=True` usa solo la media de las capas ruidosas; con False se
        muestrea ruido (exploración). Devuelve `(accion, None)` como SB3.
        """
        obs_arr = np.asarray(obs, dtype=np.float32)
        batched = obs_arr.ndim > len(self.obs_shape)
        x = torch.as_tensor(obs_arr.reshape(-1, self.obs_dim), device=self.device)

        was_training = self.online_net.training
        self.online_net.train(not deterministic)
        with torch.no_grad():
            q_values = (self.online_net(x) * self.support).sum(dim=2)
            actions = q_values.argmax(dim=1).cpu().numpy()
        self.online_net.train(was_training)

        return (actions if batched else int(actions[0])), None

    # ------------------------------------------------------------------ #
    # Entrenamiento
    # ------------------------------------------------------------------ #
    def _make_nstep(self, buf: Deque[tuple]) -> tuple:
        """Colapsa el buffer de n pasos en (s_0, a_0, R_n, s_n, terminated, k)."""
        ret = 0.0
        for i, (_, _, r, _, _) in enumerate(buf):
            ret += (self.gamma ** i) * r
        obs0, act0 = buf[0][0], buf[0][1]
        next_obs, terminated = buf[-1][3], buf[-1][4]
        return obs0, act0, ret, next_obs, terminated, len(buf)

    def _store(self, transition: tuple) -> None:
        obs0, act0, ret, next_obs, terminated, k = transition
        self.replay.add(
            np.asarray(obs0, dtype=np.float32).reshape(-1),
            act0, ret,
            np.asarray(next_obs, dtype=np.float32).reshape(-1),
            terminated, k,
        )

    def _train_step(self, beta: float) -> float:
        batch = self.replay.sample(self.batch_size, beta, self.rng)
        dev = self.device
        obs = torch.as_tensor(batch["obs"], device=dev)
        next_obs = torch.as_tensor(batch["next_obs"], device=dev)
        actions = torch.as_tensor(batch["actions"], device=dev)
        rewards = torch.as_tensor(batch["rewards"], device=dev)
        dones = torch.as_tensor(batch["dones"], device=dev)
        weights = torch.as_tensor(batch["weights"], device=dev)
        gamma_n = self.gamma ** torch.as_tensor(batch["n_steps"], device=dev)
        bsz = obs.shape[0]

        with torch.no_grad():
            # Double DQN: la red online elige a*, la target evalúa su distribución.
            next_probs_online = self.online_net(next_obs)
            next_actions = (next_probs_online * self.support).sum(dim=2).argmax(dim=1)
            self.target_net.reset_noise()
            next_probs = self.target_net(next_obs)[torch.arange(bsz, device=dev), next_actions]

            # Proyección categórica de la distribución objetivo sobre el soporte fijo.
            tz = rewards.unsqueeze(1) + (1.0 - dones).unsqueeze(1) * gamma_n.unsqueeze(1) * self.support.unsqueeze(0)
            tz = tz.clamp(self.v_min, self.v_max)
            b = (tz - self.v_min) / self.delta_z
            lower = b.floor().long()
            upper = b.ceil().long()
            # Si b es entero, lower == upper y la masa se perdería: se separan los índices.
            lower[(upper > 0) & (lower == upper)] -= 1
            upper[(lower < (self.n_atoms - 1)) & (lower == upper)] += 1

            target_dist = torch.zeros(bsz, self.n_atoms, device=dev)
            offset = (torch.arange(bsz, device=dev) * self.n_atoms).unsqueeze(1)
            target_dist.view(-1).index_add_(0, (lower + offset).view(-1), (next_probs * (upper.float() - b)).view(-1))
            target_dist.view(-1).index_add_(0, (upper + offset).view(-1), (next_probs * (b - lower.float())).view(-1))

        log_probs = self.online_net(obs, log=True)[torch.arange(bsz, device=dev), actions]
        per_sample_loss = -(target_dist * log_probs).sum(dim=1)  # cross-entropy (KL salvo constante)
        loss = (weights * per_sample_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.replay.update_priorities(batch["idx"], per_sample_loss.detach().cpu().numpy())
        self.online_net.reset_noise()
        return float(loss.item())

    def learn(self, total_timesteps: int, env: Optional[gym.Env] = None) -> "RainbowDQN":
        """Entrena `total_timesteps` pasos de entorno, logueando las mismas métricas que PPO."""
        from torch.utils.tensorboard import SummaryWriter

        env = env if env is not None else self.env
        if env is None:
            raise ValueError("Se requiere un entorno: RainbowDQN(env) o learn(..., env=env).")
        self.env = env

        writer = None
        if self.tensorboard_log is not None:
            run_dir = next_run_dir(self.tensorboard_log, "Rainbow")
            writer = SummaryWriter(log_dir=str(run_dir))
            if self.verbose:
                print(f"[Rainbow] Logs de TensorBoard en: {run_dir}")

        tracker = CombatMetricsTracker(window_size=self.metrics_window)
        ep_returns: Deque[float] = deque(maxlen=100)
        ep_lengths: Deque[int] = deque(maxlen=100)
        losses: Deque[float] = deque(maxlen=100)
        nstep_buf: Deque[tuple] = deque()

        obs, _ = env.reset(seed=self.seed)
        ep_return, ep_len = 0.0, 0
        start_step = self.num_timesteps
        self.online_net.train()

        for t in range(1, total_timesteps + 1):
            self.num_timesteps = start_step + t

            # Calentamiento aleatorio para poblar el buffer; luego explora el ruido de las NoisyLinear.
            if self.num_timesteps <= self.learning_starts:
                action = self.action_space.sample()
            else:
                self.online_net.reset_noise()
                action, _ = self.predict(obs, deterministic=False)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_return += float(reward)
            ep_len += 1

            nstep_buf.append((obs, action, float(reward), next_obs, bool(terminated)))
            if done:
                # Fin de episodio: vaciar el buffer n-step. Si fue truncado (no terminal) se
                # sigue haciendo bootstrap desde next_obs (terminated=False).
                while nstep_buf:
                    self._store(self._make_nstep(nstep_buf))
                    nstep_buf.popleft()
            elif len(nstep_buf) == self.n_step:
                self._store(self._make_nstep(nstep_buf))
                nstep_buf.popleft()

            obs = next_obs

            if done:
                tracker.update(info)
                ep_returns.append(ep_return)
                ep_lengths.append(ep_len)
                if writer is not None:
                    for tag, value in tracker.summary().items():
                        writer.add_scalar(tag, value, self.num_timesteps)
                    writer.add_scalar("rollout/ep_rew_mean", float(np.mean(ep_returns)), self.num_timesteps)
                    writer.add_scalar("rollout/ep_len_mean", float(np.mean(ep_lengths)), self.num_timesteps)
                obs, _ = env.reset()
                ep_return, ep_len = 0.0, 0

            # Actualización de la red
            if self.num_timesteps > self.learning_starts and len(self.replay) >= self.batch_size:
                if self.num_timesteps % self.train_freq == 0:
                    frac = min(1.0, t / max(1, total_timesteps))
                    beta = self.per_beta_start + frac * (1.0 - self.per_beta_start)
                    losses.append(self._train_step(beta))
                    if writer is not None and self.num_timesteps % 1000 < self.train_freq:
                        writer.add_scalar("train/loss", float(np.mean(losses)), self.num_timesteps)
                if self.num_timesteps % self.target_update_interval == 0:
                    self.target_net.load_state_dict(self.online_net.state_dict())

            if self.verbose and t % 10_000 == 0:
                msg = f"[Rainbow] {self.num_timesteps}/{start_step + total_timesteps} pasos"
                summary = tracker.summary()
                if "combat/win_rate_window" in summary:
                    msg += f" | win rate (ventana): {summary['combat/win_rate_window']:.1f}%"
                print(msg)

        if writer is not None:
            writer.flush()
            writer.close()
        return self

    # ------------------------------------------------------------------ #
    # Persistencia
    # ------------------------------------------------------------------ #
    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.online_net.state_dict(),
                "hparams": self.hparams,
                "obs_shape": self.obs_shape,
                "n_actions": self.n_actions,
                "num_timesteps": self.num_timesteps,
            },
            path,
        )

    @classmethod
    def load(cls, path: Union[str, Path], device: str = "auto") -> "RainbowDQN":
        """Carga un modelo guardado. Sirve para inferencia/evaluación (no restaura el replay buffer)."""
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=tuple(ckpt["obs_shape"]), dtype=np.float32)
        act_space = gym.spaces.Discrete(int(ckpt["n_actions"]))
        hp = dict(ckpt["hparams"])
        hp["buffer_size"] = 1  # el buffer no se restaura; evita reservar memoria al cargar
        agent = cls(observation_space=obs_space, action_space=act_space, device=device, verbose=0, **hp)
        agent.online_net.load_state_dict(ckpt["state_dict"])
        agent.target_net.load_state_dict(ckpt["state_dict"])
        agent.num_timesteps = int(ckpt.get("num_timesteps", 0))
        return agent
