"""
PPO-Lagrangian Trainer for Regime-Conditioned CMDP Trading

Implements:
- Rollout buffer with GAE computation for rewards and costs
- PPO clipped objective with Lagrangian penalty
- Primal-dual optimization for constraint satisfaction
- Shield-aware training (logs shield interventions)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset
import time
from pathlib import Path
import json

from config import Config, PPOConfig, TrainingConfig, ActionSpaceType
from networks import ActorCritic, SharedActorCritic, create_actor_critic
from trading_env import TradingCMDPEnv, VectorizedTradingEnv


@dataclass
class RolloutSample:
    """Single timestep sample from rollout. Not currently instantiated
    anywhere -- RolloutBuffer stores samples directly in its own arrays
    rather than as a list of these."""
    observation: np.ndarray
    action: np.ndarray
    safe_action: np.ndarray  # NOT the post-shield action -- see RolloutBuffer.safe_actions below
    reward: float
    cost_vector: np.ndarray  # K costs
    done: bool
    log_prob: float
    value: float
    cost_values: np.ndarray  # K cost value estimates
    shield_modified: bool
    shield_magnitude: float


class RolloutBuffer:
    """
    Buffer for storing rollout trajectories and computing advantages.

    Stores trajectories and computes:
    - GAE advantages for reward
    - GAE advantages for each cost (optional, can use simple returns)
    - Returns for value function training
    """

    def __init__(
            self,
            buffer_size: int,
            obs_dim: int,
            action_dim: int,
            n_costs: int,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            device: str = 'cpu'
    ):
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_costs = n_costs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device

        # Pre-allocate buffers
        self.observations = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        # NOTE: currently populated with the raw proposed action, not the
        # actual post-shield action the environment executed --
        # ShieldDiagnostics.to_dict() doesn't expose it. Not read by
        # update_policy(); the shield still genuinely constrains what gets
        # traded, just not what's recorded here.
        self.safe_actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.costs = np.zeros((buffer_size, n_costs), dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.cost_values = np.zeros((buffer_size, n_costs), dtype=np.float32)
        self.shield_modified = np.zeros(buffer_size, dtype=np.float32)
        self.shield_magnitudes = np.zeros(buffer_size, dtype=np.float32)

        # Computed after rollout
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)
        self.cost_advantages = np.zeros((buffer_size, n_costs), dtype=np.float32)
        self.cost_returns = np.zeros((buffer_size, n_costs), dtype=np.float32)

        self.ptr = 0
        self.path_start_idx = 0
        self.full = False

    def add(
            self,
            obs: np.ndarray,
            action: np.ndarray,
            safe_action: np.ndarray,
            reward: float,
            cost_vec: np.ndarray,
            done: bool,
            log_prob: float,
            value: float,
            cost_values: np.ndarray,
            shield_modified: bool = False,
            shield_magnitude: float = 0.0
    ):
        """Add a single timestep to the buffer."""
        assert self.ptr < self.buffer_size, "Buffer overflow"

        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.safe_actions[self.ptr] = safe_action
        self.rewards[self.ptr] = reward
        self.costs[self.ptr] = cost_vec
        self.dones[self.ptr] = float(done)
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        self.cost_values[self.ptr] = cost_values
        self.shield_modified[self.ptr] = float(shield_modified)
        self.shield_magnitudes[self.ptr] = shield_magnitude

        self.ptr += 1
        if self.ptr == self.buffer_size:
            self.full = True

    def finish_path(self, last_value: float = 0.0, last_cost_values: Optional[np.ndarray] = None):
        """
        Call at the end of a trajectory (or when buffer is full).
        Computes GAE advantages and returns for the path.
        """
        if last_cost_values is None:
            last_cost_values = np.zeros(self.n_costs, dtype=np.float32)

        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = self.rewards[path_slice]
        values = self.values[path_slice]
        dones = self.dones[path_slice]
        costs = self.costs[path_slice]
        cost_values = self.cost_values[path_slice]

        path_len = self.ptr - self.path_start_idx

        # Compute GAE for rewards
        self.advantages[path_slice], self.returns[path_slice] = self._compute_gae(
            rewards, values, dones, last_value, path_len
        )

        # Compute GAE for each cost
        for k in range(self.n_costs):
            self.cost_advantages[path_slice, k], self.cost_returns[path_slice, k] = self._compute_gae(
                costs[:, k], cost_values[:, k], dones, last_cost_values[k], path_len
            )

        self.path_start_idx = self.ptr

    def _compute_gae(
            self,
            rewards: np.ndarray,
            values: np.ndarray,
            dones: np.ndarray,
            last_value: float,
            path_len: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation."""
        advantages = np.zeros(path_len, dtype=np.float32)
        returns = np.zeros(path_len, dtype=np.float32)

        last_gae = 0.0
        for t in reversed(range(path_len)):
            if t == path_len - 1:
                next_value = last_value
                next_non_terminal = 1.0 - dones[t]
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - dones[t]

            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
            returns[t] = advantages[t] + values[t]

        return advantages, returns

    def get(self, normalize_advantages: bool = True) -> Dict[str, torch.Tensor]:
        """
        Get all data from buffer as tensors.
        Optionally normalizes advantages.
        """
        assert self.ptr == self.buffer_size or self.full, "Buffer not full"

        # Normalize reward advantages
        advantages = self.advantages[:self.ptr].copy()
        if normalize_advantages and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Normalize cost advantages (per cost)
        cost_advantages = self.cost_advantages[:self.ptr].copy()
        if normalize_advantages and len(cost_advantages) > 1:
            for k in range(self.n_costs):
                cost_adv_k = cost_advantages[:, k]
                if cost_adv_k.std() > 1e-8:
                    cost_advantages[:, k] = (cost_adv_k - cost_adv_k.mean()) / (cost_adv_k.std() + 1e-8)

        data = {
            'observations': torch.tensor(self.observations[:self.ptr], device=self.device),
            'actions': torch.tensor(self.actions[:self.ptr], device=self.device),
            'safe_actions': torch.tensor(self.safe_actions[:self.ptr], device=self.device),
            'log_probs': torch.tensor(self.log_probs[:self.ptr], device=self.device),
            'advantages': torch.tensor(advantages, device=self.device),
            'returns': torch.tensor(self.returns[:self.ptr], device=self.device),
            'cost_advantages': torch.tensor(cost_advantages, device=self.device),
            'cost_returns': torch.tensor(self.cost_returns[:self.ptr], device=self.device),
            'values': torch.tensor(self.values[:self.ptr], device=self.device),
            'cost_values': torch.tensor(self.cost_values[:self.ptr], device=self.device),
        }

        return data

    def reset(self):
        """Reset buffer for new rollout."""
        self.ptr = 0
        self.path_start_idx = 0
        self.full = False

    def get_statistics(self) -> Dict[str, float]:
        """Get buffer statistics for logging."""
        # Per-asset std of the RAW proposed action (pre-clip, pre-smoothing,
        # pre-shield) over this rollout -- low std relative to the action
        # bounds means the policy is saturating at its output bounds rather
        # than making graduated decisions (see PPOConfig.entropy_coef /
        # ActorCritic's initial log_std for the usual causes).
        action_std_per_asset = np.std(self.actions[:self.ptr], axis=0)
        return {
            'mean_reward': float(np.mean(self.rewards[:self.ptr])),
            'std_reward': float(np.std(self.rewards[:self.ptr])),
            'mean_costs': [float(np.mean(self.costs[:self.ptr, k])) for k in range(self.n_costs)],
            'shield_intervention_rate': float(np.mean(self.shield_modified[:self.ptr])),
            'mean_shield_magnitude': float(np.mean(self.shield_magnitudes[:self.ptr])),
            'episode_returns': float(np.sum(self.rewards[:self.ptr])),
            'action_std_per_asset': action_std_per_asset.tolist(),
            'mean_action_std': float(np.mean(action_std_per_asset)),
        }


@dataclass
class TrainingMetrics:
    """Metrics collected during training."""
    iteration: int
    total_timesteps: int

    # Losses
    policy_loss: float
    value_loss: float
    cost_value_losses: List[float]
    entropy: float

    # PPO statistics
    approx_kl: float
    clip_fraction: float

    # Dual variables
    lambdas: List[float]

    # Constraint statistics
    mean_costs: List[float]
    constraint_violations: List[float]

    # Performance
    mean_reward: float
    # Sum (NOT mean, despite the field this replaces being named mean_return)
    # of raw per-step rewards over the whole rollout -- RolloutBuffer.get_statistics's
    # 'episode_returns' is sum(rewards[:ptr]); the fold-2 autopsy traced the old
    # misleading name back to explaining the seemingly-huge ~-9000 magnitude
    # seen in early training logs (per-step reward O(1-10) x rollout_length).
    rollout_return_sum: float

    # Shield statistics
    shield_intervention_rate: float
    mean_shield_magnitude: float

    # Action saturation (raw proposed action, pre-clip/smoothing/shield)
    action_std_per_asset: List[float]
    mean_action_std: float

    # Entropy coefficient actually used this iteration (annealed -- see
    # PPOLagrangianTrainer._entropy_coef_for_iteration)
    entropy_coef: float

    # Timing
    rollout_time: float
    update_time: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'iteration': self.iteration,
            'total_timesteps': self.total_timesteps,
            'policy_loss': self.policy_loss,
            'value_loss': self.value_loss,
            'cost_value_losses': self.cost_value_losses,
            'entropy': self.entropy,
            'approx_kl': self.approx_kl,
            'clip_fraction': self.clip_fraction,
            'lambdas': self.lambdas,
            'mean_costs': self.mean_costs,
            'constraint_violations': self.constraint_violations,
            'mean_reward': self.mean_reward,
            'rollout_return_sum': self.rollout_return_sum,
            'shield_intervention_rate': self.shield_intervention_rate,
            'mean_shield_magnitude': self.mean_shield_magnitude,
            'action_std_per_asset': self.action_std_per_asset,
            'mean_action_std': self.mean_action_std,
            'entropy_coef': self.entropy_coef,
            'rollout_time': self.rollout_time,
            'update_time': self.update_time,
        }


class PPOLagrangianTrainer:
    """
    PPO-Lagrangian trainer for CMDP trading.

    Combines:
    - PPO with clipped objective for stable policy updates
    - Lagrangian relaxation for constraint satisfaction
    - Shield-aware training (actions are shielded before environment step)

    The Lagrangian objective is:
        L(π, λ) = E[Σ γ^t (r_t - λᵀ(c_t - d))]

    Dual update:
        λ ← max(0, λ + η_λ (Ĉ - d))
    """

    def __init__(
            self,
            env: Union[TradingCMDPEnv, VectorizedTradingEnv],
            config: Config,
            actor_critic: Optional[Union[ActorCritic, SharedActorCritic]] = None,
            device: str = 'cpu'
    ):
        self.env = env
        self.config = config
        self.ppo_config = config.ppo
        self.training_config = config.training
        self.device = device

        # Environment info
        self.obs_dim = env.observation_space.shape[0]
        self.n_assets = len(config.data.tickers)
        self.n_costs = 3  # dd, turnover, leverage (configurable)

        # Action space handling
        if config.action_space.action_type == ActionSpaceType.DISCRETE:
            self.action_dim = self.n_assets  # One action per asset
            self.n_action_levels = len(config.action_space.discrete_levels)
        else:
            self.action_dim = self.n_assets

        # Create actor-critic if not provided
        if actor_critic is None:
            self.actor_critic = create_actor_critic(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                n_cost_constraints=self.n_costs,
                config=config.ppo,
                action_type=config.action_space.action_type,
                n_discrete_levels=len(config.action_space.discrete_levels),
                shared=False
            )
        else:
            self.actor_critic = actor_critic

        self.actor_critic.to(device)

        # Optimizers
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(),
            lr=self.ppo_config.policy_lr,
            eps=1e-5
        )

        # Dual variables (Lagrange multipliers) - one per constraint.
        # constraint_limits are the d_k budgets the dual update compares
        # mean costs against (see ConstraintConfig.constraint_budgets for
        # why these default to 0.0 -- costs are already excess-over-budget
        # by the time they reach here, not raw unconstrained metrics).
        self.constraint_limits = np.array(
            config.constraints.constraint_budgets, dtype=np.float32
        )

        # lambda_init is either a single float (uniform warm-start, prior
        # behavior) or a length-n_costs list (per-constraint warm-start,
        # order [dd, turnover, leverage] -- see ConstraintConfig/
        # RiskManager.get_cost_vector).
        if isinstance(self.ppo_config.lambda_init, (list, tuple)):
            assert len(self.ppo_config.lambda_init) == self.n_costs, (
                f"lambda_init list must have {self.n_costs} entries (dd, turnover, "
                f"leverage), got {len(self.ppo_config.lambda_init)}"
            )
            self.lambdas = np.array(self.ppo_config.lambda_init, dtype=np.float32)
        else:
            self.lambdas = np.full(self.n_costs, self.ppo_config.lambda_init, dtype=np.float32)
        self.dual_lr = self.ppo_config.dual_lr
        self.lambda_max = self.ppo_config.lambda_max

        # PID-Lagrangian state (Stooke et al. 2020) -- only used when
        # ppo_config.dual_update_method == "pid"; see update_dual_variables().
        self._cost_integral = np.zeros(self.n_costs, dtype=np.float32)
        self._prev_violations = np.zeros(self.n_costs, dtype=np.float32)

        # Rollout buffer
        self.buffer = RolloutBuffer(
            buffer_size=self.training_config.rollout_length,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            n_costs=self.n_costs,
            gamma=self.ppo_config.gamma,
            gae_lambda=self.ppo_config.gae_lambda,
            device=device
        )

        # Training state
        self.total_timesteps = 0
        self.iteration = 0
        self.current_entropy_coef = self.ppo_config.entropy_coef
        self.metrics_history: List[TrainingMetrics] = []

        # Logging (will be set by caller if needed)
        self.log_dir = None

        # Seed the environment's own RNG (used for rolling-episode start-point
        # sampling in TradingCMDPEnv.reset()) so that, combined with
        # seed_everything() at the entry point, training is fully
        # reproducible for a fixed config.training.seed. Without this, the
        # first ever reset() leaves gymnasium to seed itself from OS entropy.
        self.env.reset(seed=self.training_config.seed)

    def collect_rollouts(self) -> Dict[str, float]:
        """
        Collect rollout data by running policy in environment.
        Returns buffer statistics.
        """
        self.buffer.reset()
        obs, info = self.env.reset()

        rollout_rewards = []
        rollout_costs = []

        for step in range(self.training_config.rollout_length):
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

                # Get action from policy
                action, log_prob, entropy = self.actor_critic.get_action(obs_tensor)
                value = self.actor_critic.reward_critic(obs_tensor).item()

                # Get cost value estimates
                cost_values = np.array([
                    critic(obs_tensor).item()
                    for critic in self.actor_critic.cost_critics
                ])

                action_np = action.cpu().numpy().squeeze()
                log_prob_np = log_prob.cpu().item()

            # Step environment (shield is applied internally)
            next_obs, reward, terminated, truncated, info = self.env.step(action_np)
            done = terminated or truncated

            # Extract cost vector and shield info. trading_env.py nests
            # shield diagnostics under info['shield'] (ShieldDiagnostics.to_dict()),
            # not flat top-level 'shield_modified'/'shield_magnitude' keys --
            # reading the wrong keys here silently zeroed out
            # shield_intervention_rate/mean_shield_magnitude in every logged
            # metric, even when the shield was actively intervening. info
            # has no 'shield' key at all when no shield is attached
            # (ShieldType.NONE), hence the empty-dict default.
            cost_vec = info.get('cost_vec', np.zeros(self.n_costs))
            shield_diag = info.get('shield', {})
            shield_modified = shield_diag.get('was_modified', False)
            shield_magnitude = shield_diag.get('modification_magnitude', 0.0)

            # Note: the environment doesn't expose the post-shield action via
            # info (ShieldDiagnostics.to_dict() omits it), so this is still
            # the raw proposed action. Only feeds RolloutBuffer.safe_actions,
            # which update_policy() doesn't currently train on.
            safe_action = action_np

            # Store in buffer
            self.buffer.add(
                obs=obs,
                action=action_np,
                safe_action=safe_action,
                reward=reward,
                cost_vec=cost_vec,
                done=done,
                log_prob=log_prob_np,
                value=value,
                cost_values=cost_values,
                shield_modified=shield_modified,
                shield_magnitude=shield_magnitude
            )

            rollout_rewards.append(reward)
            rollout_costs.append(cost_vec)

            obs = next_obs
            self.total_timesteps += 1

            if done:
                # Finish path with zero bootstrap
                self.buffer.finish_path(last_value=0.0, last_cost_values=np.zeros(self.n_costs))
                obs, info = self.env.reset()

        # If path didn't end, bootstrap from value function
        if not done:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                last_value = self.actor_critic.reward_critic(obs_tensor).item()
                last_cost_values = np.array([
                    critic(obs_tensor).item()
                    for critic in self.actor_critic.cost_critics
                ])
            self.buffer.finish_path(last_value=last_value, last_cost_values=last_cost_values)

        return self.buffer.get_statistics()

    def _entropy_coef_for_iteration(self, iteration: int) -> float:
        """entropy_coef annealed toward min_entropy_coef: entropy_decay was a
        dead config field (defined, never read) until this fix -- entropy_coef
        stayed fixed at its initial value for the whole run, a constant,
        undiminishing incentive toward higher action variance that the fold-2
        autopsy traced mean_action_std growing (not shrinking) across every
        Phase 2 experiment regardless of dual-update tuning."""
        decayed = self.ppo_config.entropy_coef * (self.ppo_config.entropy_decay ** iteration)
        return max(self.ppo_config.min_entropy_coef, decayed)

    def _log_std_ceiling_for_iteration(self, iteration: int) -> Optional[float]:
        """None (default) when log_std_decay_floor is unset -- log_std stays
        purely learned, matching prior behavior. When set, a DIRECT ceiling
        on the policy's (state-independent) log_std parameter, annealed from
        log_std_init down to log_std_decay_floor over training, using the
        same progress fraction as the entropy anneal so both dials tighten
        together rather than on independent, potentially-mismatched schedules."""
        if self.ppo_config.log_std_decay_floor is None or self.ppo_config.entropy_coef <= 0:
            return None
        progress = self._entropy_coef_for_iteration(iteration) / self.ppo_config.entropy_coef
        floor_progress = 1.0 - progress  # 0.0 at iter 0 -> ~1.0 once entropy_coef bottoms out
        return self.ppo_config.log_std_init + (
            self.ppo_config.log_std_decay_floor - self.ppo_config.log_std_init
        ) * floor_progress

    def update_policy(self) -> Dict[str, float]:
        """
        Perform PPO update with Lagrangian penalty.
        Returns update statistics.
        """
        self.current_entropy_coef = self._entropy_coef_for_iteration(self.iteration)

        data = self.buffer.get(normalize_advantages=True)

        # Compute combined advantage: A = A_r - λᵀ A_c
        combined_advantages = data['advantages'].clone()
        for k in range(self.n_costs):
            combined_advantages -= self.lambdas[k] * data['cost_advantages'][:, k]

        # Safety: clip extreme advantages and handle NaN
        combined_advantages = torch.nan_to_num(combined_advantages, nan=0.0)
        combined_advantages = torch.clamp(combined_advantages, -10.0, 10.0)

        # PPO update epochs
        batch_size = self.ppo_config.batch_size
        n_samples = len(data['observations'])
        indices = np.arange(n_samples)

        policy_losses = []
        value_losses = []
        cost_value_losses = [[] for _ in range(self.n_costs)]
        entropies = []
        approx_kls = []
        clip_fractions = []

        for epoch in range(self.ppo_config.n_epochs):
            np.random.shuffle(indices)

            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]

                # Get batch data
                obs_batch = data['observations'][batch_indices]
                actions_batch = data['actions'][batch_indices]
                old_log_probs_batch = data['log_probs'][batch_indices]
                advantages_batch = combined_advantages[batch_indices]
                returns_batch = data['returns'][batch_indices]
                cost_returns_batch = data['cost_returns'][batch_indices]

                # Evaluate actions under current policy
                new_log_probs, entropy, new_values, new_cost_values = self.actor_critic.evaluate_actions(
                    obs_batch, actions_batch
                )

                # Policy loss (PPO clipped objective)
                log_ratio = new_log_probs - old_log_probs_batch
                ratio = torch.exp(log_ratio)

                # Clipped surrogate
                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.ppo_config.clip_epsilon,
                    1.0 + self.ppo_config.clip_epsilon
                ) * advantages_batch

                policy_loss = -torch.min(surr1, surr2).mean()

                # Entropy bonus (annealed -- see _entropy_coef_for_iteration)
                entropy_loss = -self.current_entropy_coef * entropy.mean()

                # Value loss (reward critic)
                value_pred = self.actor_critic.reward_critic(obs_batch).squeeze()
                value_loss = 0.5 * nn.functional.mse_loss(
                    value_pred, returns_batch
                )

                # Cost value losses
                cost_value_loss = torch.tensor(0.0, device=self.device)
                for k, cost_critic in enumerate(self.actor_critic.cost_critics):
                    cost_pred = cost_critic(obs_batch).squeeze()
                    cost_target = cost_returns_batch[:, k]
                    loss_k = nn.functional.mse_loss(cost_pred, cost_target)
                    cost_value_loss += loss_k
                    cost_value_losses[k].append(loss_k.item())

                # Total loss
                total_loss = policy_loss + entropy_loss + value_loss + cost_value_loss

                # Skip update if loss is NaN
                if torch.isnan(total_loss):
                    print("Warning: NaN loss detected, skipping update")
                    continue

                # Optimization step
                self.optimizer.zero_grad()
                total_loss.backward()

                # Check for NaN gradients
                has_nan_grad = False
                for param in self.actor_critic.parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        has_nan_grad = True
                        param.grad = torch.nan_to_num(param.grad, nan=0.0)

                if has_nan_grad:
                    print("Warning: NaN gradients detected, zeroed them")

                # Gradient clipping
                if self.ppo_config.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(
                        self.actor_critic.parameters(),
                        self.ppo_config.max_grad_norm
                    )

                self.optimizer.step()

                # Track statistics
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.mean().item())

                # Approximate KL divergence
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    clip_frac = ((ratio - 1.0).abs() > self.ppo_config.clip_epsilon).float().mean().item()
                approx_kls.append(approx_kl)
                clip_fractions.append(clip_frac)

        # Optional direct log_std ceiling (see _log_std_ceiling_for_iteration) --
        # a no-op unless log_std_decay_floor is explicitly configured.
        log_std_ceiling = self._log_std_ceiling_for_iteration(self.iteration)
        if log_std_ceiling is not None and hasattr(self.actor_critic.policy, 'log_std'):
            with torch.no_grad():
                self.actor_critic.policy.log_std.clamp_(max=log_std_ceiling)

        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'cost_value_losses': [np.mean(losses) for losses in cost_value_losses],
            'entropy': np.mean(entropies),
            'approx_kl': np.mean(approx_kls),
            'clip_fraction': np.mean(clip_fractions),
        }

    def update_dual_variables(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update Lagrange multipliers based on constraint violations.

        additive (default): λ ← max(0, λ + η_λ (Ĉ - d))

        pid (Stooke et al. 2020): λ ← clip(Kp*violation + Ki*integral(violation)
        + Kd*d(violation)/dt, 0, lambda_max) -- reacts to the CURRENT violation
        and its trend, not only its accumulated history, so it can respond
        faster and damp instead of overshoot once violations start falling.

        Returns: (new_lambdas, constraint_violations)
        """
        # Compute mean costs from buffer
        mean_costs = np.mean(self.buffer.costs[:self.buffer.ptr], axis=0)

        # Constraint violations
        violations = mean_costs - self.constraint_limits

        if self.ppo_config.dual_update_method == "pid":
            # Integral term accumulates violation like the additive update,
            # but is clipped at 0 to prevent windup driving lambda negative
            # once violations turn negative (i.e. the constraint is being
            # satisfied) -- lambda itself is clipped >= 0 below regardless.
            self._cost_integral = np.maximum(0.0, self._cost_integral + violations)
            derivative = violations - self._prev_violations
            self._prev_violations = violations.copy()

            self.lambdas = np.clip(
                self.ppo_config.pid_kp * violations
                + self.ppo_config.pid_ki * self._cost_integral
                + self.ppo_config.pid_kd * derivative,
                0.0,
                self.lambda_max
            )
        else:
            self.lambdas = np.clip(
                self.lambdas + self.dual_lr * violations,
                0.0,
                self.lambda_max
            )

        return self.lambdas.copy(), violations

    def train(self, total_timesteps: Optional[int] = None) -> List[TrainingMetrics]:
        """
        Main training loop.

        Args:
            total_timesteps: Total timesteps to train (overrides config if provided)

        Returns:
            List of training metrics per iteration
        """
        if total_timesteps is None:
            total_timesteps = self.training_config.total_timesteps

        n_iterations = total_timesteps // self.training_config.rollout_length

        print(f"Starting PPO-Lagrangian training:")
        print(f"  Total timesteps: {total_timesteps}")
        print(f"  Rollout length: {self.training_config.rollout_length}")
        print(f"  Iterations: {n_iterations}")
        print(f"  Batch size: {self.ppo_config.batch_size}")
        print(f"  N epochs: {self.ppo_config.n_epochs}")
        print()

        for iteration in range(n_iterations):
            self.iteration = iteration

            # Collect rollouts
            rollout_start = time.time()
            rollout_stats = self.collect_rollouts()
            rollout_time = time.time() - rollout_start

            # Update policy
            update_start = time.time()
            update_stats = self.update_policy()
            update_time = time.time() - update_start

            # Update dual variables
            new_lambdas, violations = self.update_dual_variables()

            # Compile metrics
            metrics = TrainingMetrics(
                iteration=iteration,
                total_timesteps=self.total_timesteps,
                policy_loss=update_stats['policy_loss'],
                value_loss=update_stats['value_loss'],
                cost_value_losses=update_stats['cost_value_losses'],
                entropy=update_stats['entropy'],
                approx_kl=update_stats['approx_kl'],
                clip_fraction=update_stats['clip_fraction'],
                lambdas=new_lambdas.tolist(),
                mean_costs=rollout_stats['mean_costs'],
                constraint_violations=violations.tolist(),
                mean_reward=rollout_stats['mean_reward'],
                rollout_return_sum=rollout_stats['episode_returns'],
                shield_intervention_rate=rollout_stats['shield_intervention_rate'],
                mean_shield_magnitude=rollout_stats['mean_shield_magnitude'],
                action_std_per_asset=rollout_stats['action_std_per_asset'],
                mean_action_std=rollout_stats['mean_action_std'],
                entropy_coef=self.current_entropy_coef,
                rollout_time=rollout_time,
                update_time=update_time,
            )

            self.metrics_history.append(metrics)

            # Logging
            if iteration % 1 == 0:  # Log every iteration
                self._log_iteration(metrics)

            # Checkpointing (disabled - only save at end)
            # if self.training_config.save_frequency > 0 and iteration % 10 == 0:
            #     self.save_checkpoint(f"checkpoint_{iteration}.pt")

        # Final save
        self.save_checkpoint("final_model.pt")

        return self.metrics_history

    def _log_iteration(self, metrics: TrainingMetrics):
        """Log training progress."""
        print(f"Iteration {metrics.iteration} | Timesteps: {metrics.total_timesteps}")
        print(f"  Rewards: mean={metrics.mean_reward:.4f}, rollout_return_sum={metrics.rollout_return_sum:.4f}")
        print(f"  Losses: policy={metrics.policy_loss:.4f}, value={metrics.value_loss:.4f}")
        print(f"  PPO: entropy={metrics.entropy:.4f}, kl={metrics.approx_kl:.4f}, clip={metrics.clip_fraction:.3f}, "
              f"entropy_coef={metrics.entropy_coef:.5f}")
        print(f"  Costs: {[f'{c:.4f}' for c in metrics.mean_costs]}")
        print(f"  Lambdas: {[f'{l:.4f}' for l in metrics.lambdas]}")
        print(f"  Shield: rate={metrics.shield_intervention_rate:.3f}, mag={metrics.mean_shield_magnitude:.4f}")
        print(f"  Action std (raw, per-asset): {[f'{s:.4f}' for s in metrics.action_std_per_asset]} "
              f"(mean={metrics.mean_action_std:.4f})")
        print(f"  Time: rollout={metrics.rollout_time:.2f}s, update={metrics.update_time:.2f}s")
        print()

        # Save to JSON log
        if self.log_dir:
            log_path = self.log_dir / "training_log.jsonl"
            with open(log_path, 'a') as f:
                f.write(json.dumps(metrics.to_dict()) + '\n')

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        if self.log_dir is None:
            return

        checkpoint = {
            'iteration': self.iteration,
            'total_timesteps': self.total_timesteps,
            'model_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lambdas': self.lambdas,
            'config': self.config.to_dict(),
        }

        path = self.log_dir / filename
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.actor_critic.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lambdas = checkpoint['lambdas']
        self.iteration = checkpoint['iteration']
        self.total_timesteps = checkpoint['total_timesteps']

        print(f"Loaded checkpoint from {path}")
        print(f"  Iteration: {self.iteration}, Timesteps: {self.total_timesteps}")

    def get_policy(self) -> Union[ActorCritic, SharedActorCritic]:
        """Get trained policy."""
        return self.actor_critic


class VanillaPPOTrainer(PPOLagrangianTrainer):
    """
    Vanilla PPO trainer (no constraints).
    Used as baseline comparison.
    """

    def __init__(self, env, config, actor_critic=None, device='cpu'):
        super().__init__(env, config, actor_critic, device)
        # Disable dual variables
        self.lambdas = np.zeros(self.n_costs, dtype=np.float32)
        self.dual_lr = 0.0  # No dual updates

    def update_dual_variables(self):
        """No-op for vanilla PPO."""
        mean_costs = np.mean(self.buffer.costs[:self.buffer.ptr], axis=0)
        violations = mean_costs - self.constraint_limits
        return self.lambdas.copy(), violations


class RewardShapingPPOTrainer(PPOLagrangianTrainer):
    """
    PPO with reward shaping (soft penalties instead of Lagrangian).
    Used as baseline comparison.
    """

    def __init__(
            self,
            env,
            config,
            penalty_weights: Optional[np.ndarray] = None,
            actor_critic=None,
            device='cpu'
    ):
        super().__init__(env, config, actor_critic, device)

        # Fixed penalty weights (not learned)
        if penalty_weights is None:
            self.penalty_weights = np.array([1.0, 0.5, 0.5], dtype=np.float32)
        else:
            self.penalty_weights = penalty_weights

        # Disable dual updates
        self.lambdas = self.penalty_weights.copy()
        self.dual_lr = 0.0

    def update_dual_variables(self):
        """No dual updates - fixed penalty weights."""
        mean_costs = np.mean(self.buffer.costs[:self.buffer.ptr], axis=0)
        violations = mean_costs - self.constraint_limits
        return self.lambdas.copy(), violations


def create_trainer(
        env: Union[TradingCMDPEnv, VectorizedTradingEnv],
        config: Config,
        trainer_type: str = 'ppo_lagrangian',
        device: str = 'cpu'
) -> PPOLagrangianTrainer:
    """
    Factory function to create appropriate trainer.

    Args:
        env: Trading environment
        config: Configuration
        trainer_type: One of 'ppo_lagrangian', 'vanilla_ppo', 'reward_shaping'
        device: PyTorch device

    Returns:
        Trainer instance
    """
    if trainer_type == 'ppo_lagrangian':
        return PPOLagrangianTrainer(env, config, device=device)
    elif trainer_type == 'vanilla_ppo':
        return VanillaPPOTrainer(env, config, device=device)
    elif trainer_type == 'reward_shaping':
        return RewardShapingPPOTrainer(env, config, device=device)
    else:
        raise ValueError(f"Unknown trainer type: {trainer_type}")