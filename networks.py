"""
Neural Network Architectures for PPO-Lagrangian.
Includes policy networks (actor) and value networks (critics).
"""

from typing import List, Tuple, Optional, Dict
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical

from config import PPOConfig, ActionSpaceType


def create_mlp(
        input_dim: int,
        hidden_sizes: List[int],
        output_dim: int,
        activation: str = "tanh",
        output_activation: Optional[str] = None
) -> nn.Sequential:
    """Create a multi-layer perceptron."""
    activation_fn = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "elu": nn.ELU,
        "gelu": nn.GELU,
    }[activation]

    layers = []
    prev_dim = input_dim

    for hidden_dim in hidden_sizes:
        layers.extend([
            nn.Linear(prev_dim, hidden_dim),
            activation_fn()
        ])
        prev_dim = hidden_dim

    layers.append(nn.Linear(prev_dim, output_dim))

    if output_activation is not None:
        output_fn = {
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "softmax": lambda: nn.Softmax(dim=-1),
        }[output_activation]
        layers.append(output_fn())

    return nn.Sequential(*layers)


class ContinuousPolicy(nn.Module):
    """
    Continuous policy network for target portfolio weights.
    Outputs mean and log_std for Gaussian distribution.
    """

    def __init__(
            self,
            obs_dim: int,
            action_dim: int,
            hidden_sizes: List[int],
            activation: str = "tanh",
            log_std_init: float = -0.5,
            log_std_min: float = -20.0,
            log_std_max: float = 2.0,
            init_mean_bias: float = 0.5  # Start with stronger long bias (50%)
    ):
        super().__init__()

        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Shared feature extractor
        self.feature_net = create_mlp(
            obs_dim, hidden_sizes[:-1], hidden_sizes[-1], activation
        )

        # Mean head
        self.mean_net = nn.Linear(hidden_sizes[-1], action_dim)

        # Initialize with long bias (start with positive positions)
        with torch.no_grad():
            self.mean_net.bias.fill_(init_mean_bias)

        # Log std (state-independent or state-dependent)
        self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)

    def forward(
            self,
            obs: torch.Tensor,
            deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            action, log_prob, mean
        """
        # Check for NaN in input
        if torch.isnan(obs).any():
            obs = torch.nan_to_num(obs, nan=0.0)

        features = self.feature_net(obs)

        # Check for NaN in features
        if torch.isnan(features).any():
            features = torch.nan_to_num(features, nan=0.0)

        mean = self.mean_net(features)

        # Clamp mean to prevent extreme values and NaN
        mean = torch.clamp(mean, -10.0, 10.0)
        if torch.isnan(mean).any():
            mean = torch.nan_to_num(mean, nan=0.0)

        # Clamp log_std
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        # Create distribution
        dist = Normal(mean, std)

        if deterministic:
            action = mean
        else:
            action = dist.rsample()  # Reparameterization trick

        # Compute log probability
        log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob, mean

    def log_prob(
            self,
            obs: torch.Tensor,
            action: torch.Tensor
    ) -> torch.Tensor:
        """Compute log probability of action given observation."""
        # Check for NaN in input
        if torch.isnan(obs).any():
            obs = torch.nan_to_num(obs, nan=0.0)

        features = self.feature_net(obs)

        if torch.isnan(features).any():
            features = torch.nan_to_num(features, nan=0.0)

        mean = self.mean_net(features)
        mean = torch.clamp(mean, -10.0, 10.0)
        if torch.isnan(mean).any():
            mean = torch.nan_to_num(mean, nan=0.0)

        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)

        # Clamp log_prob to prevent extreme values
        log_prob = torch.clamp(log_prob, -100.0, 100.0)

        return log_prob

    def entropy(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute entropy of policy at given observation."""
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        dist = Normal(torch.zeros_like(std), std)
        return dist.entropy().sum(dim=-1)


class DiscretePolicy(nn.Module):
    """
    Discrete policy network for multi-discrete action space.
    One categorical distribution per asset.
    """

    def __init__(
            self,
            obs_dim: int,
            n_assets: int,
            n_levels: int,
            hidden_sizes: List[int],
            activation: str = "tanh"
    ):
        super().__init__()

        self.n_assets = n_assets
        self.n_levels = n_levels

        # Shared feature extractor
        self.feature_net = create_mlp(
            obs_dim, hidden_sizes[:-1], hidden_sizes[-1], activation
        )

        # Separate head for each asset
        self.action_heads = nn.ModuleList([
            nn.Linear(hidden_sizes[-1], n_levels)
            for _ in range(n_assets)
        ])

    def forward(
            self,
            obs: torch.Tensor,
            deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass.

        Returns:
            action (n_assets,), log_prob, logits list
        """
        features = self.feature_net(obs)

        actions = []
        log_probs = []
        logits_list = []

        for head in self.action_heads:
            logits = head(features)
            logits_list.append(logits)

            dist = Categorical(logits=logits)

            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                action = dist.sample()

            actions.append(action)
            log_probs.append(dist.log_prob(action))

        # Stack actions and sum log probs
        action = torch.stack(actions, dim=-1)
        log_prob = torch.stack(log_probs, dim=-1).sum(dim=-1)

        return action, log_prob, logits_list

    def log_prob(
            self,
            obs: torch.Tensor,
            action: torch.Tensor
    ) -> torch.Tensor:
        """Compute log probability of action given observation."""
        features = self.feature_net(obs)

        log_probs = []
        for i, head in enumerate(self.action_heads):
            logits = head(features)
            dist = Categorical(logits=logits)
            log_probs.append(dist.log_prob(action[..., i]))

        return torch.stack(log_probs, dim=-1).sum(dim=-1)

    def entropy(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute entropy of policy at given observation."""
        features = self.feature_net(obs)

        entropies = []
        for head in self.action_heads:
            logits = head(features)
            dist = Categorical(logits=logits)
            entropies.append(dist.entropy())

        return torch.stack(entropies, dim=-1).sum(dim=-1)


class ValueNetwork(nn.Module):
    """
    Value network (critic) for estimating state value.
    Used for both reward value and cost values.
    """

    def __init__(
            self,
            obs_dim: int,
            hidden_sizes: List[int],
            activation: str = "tanh"
    ):
        super().__init__()

        self.net = create_mlp(
            obs_dim, hidden_sizes, 1, activation
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Estimate value of observation."""
        return self.net(obs).squeeze(-1)


class ActorCritic(nn.Module):
    """
    Combined Actor-Critic network for PPO.
    Includes policy (actor), reward critic, and cost critics.
    """

    def __init__(
            self,
            obs_dim: int,
            action_dim: int,
            n_cost_constraints: int,
            config: PPOConfig,
            action_type: ActionSpaceType = ActionSpaceType.CONTINUOUS,
            n_discrete_levels: int = 5
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_cost_constraints = n_cost_constraints
        self.config = config
        self.action_type = action_type

        hidden_sizes = config.hidden_sizes
        activation = config.activation

        # Policy network
        if action_type == ActionSpaceType.CONTINUOUS:
            self.policy = ContinuousPolicy(
                obs_dim, action_dim, hidden_sizes, activation,
                log_std_init=config.log_std_init
            )
        else:
            self.policy = DiscretePolicy(
                obs_dim, action_dim, n_discrete_levels, hidden_sizes, activation
            )

        # Reward value network
        self.reward_critic = ValueNetwork(obs_dim, hidden_sizes, activation)

        # Cost value networks (one per constraint)
        self.cost_critics = nn.ModuleList([
            ValueNetwork(obs_dim, hidden_sizes, activation)
            for _ in range(n_cost_constraints)
        ])

    def get_action(
            self,
            obs: torch.Tensor,
            deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Get action from policy.

        Returns:
            action, log_prob, extra_info
        """
        action, log_prob, extra = self.policy(obs, deterministic)

        return action, log_prob, {'policy_extra': extra}

    def evaluate_actions(
            self,
            obs: torch.Tensor,
            action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Evaluate actions for PPO update.

        Returns:
            log_prob, entropy, reward_value, cost_values
        """
        log_prob = self.policy.log_prob(obs, action)
        entropy = self.policy.entropy(obs)
        reward_value = self.reward_critic(obs)
        cost_values = [critic(obs) for critic in self.cost_critics]

        return log_prob, entropy, reward_value, cost_values

    def get_values(
            self,
            obs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Get value estimates."""
        reward_value = self.reward_critic(obs)
        cost_values = [critic(obs) for critic in self.cost_critics]
        return reward_value, cost_values


class SharedActorCritic(nn.Module):
    """
    Actor-Critic with shared feature extraction.
    More parameter-efficient but may have optimization conflicts.
    """

    def __init__(
            self,
            obs_dim: int,
            action_dim: int,
            n_cost_constraints: int,
            config: PPOConfig,
            action_type: ActionSpaceType = ActionSpaceType.CONTINUOUS,
            n_discrete_levels: int = 5
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_cost_constraints = n_cost_constraints
        self.config = config
        self.action_type = action_type

        hidden_sizes = config.hidden_sizes
        activation = config.activation

        activation_fn = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "elu": nn.ELU,
        }[activation]

        # Shared feature extractor
        self.shared_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_sizes[0]),
            activation_fn(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            activation_fn()
        )

        feature_dim = hidden_sizes[1]

        # Policy heads
        if action_type == ActionSpaceType.CONTINUOUS:
            self.policy_mean = nn.Linear(feature_dim, action_dim)
            self.policy_log_std = nn.Parameter(torch.zeros(action_dim) - 0.5)
        else:
            self.policy_heads = nn.ModuleList([
                nn.Linear(feature_dim, n_discrete_levels)
                for _ in range(action_dim)
            ])

        # Value heads
        self.reward_value_head = nn.Linear(feature_dim, 1)
        self.cost_value_heads = nn.ModuleList([
            nn.Linear(feature_dim, 1)
            for _ in range(n_cost_constraints)
        ])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract shared features."""
        return self.shared_net(obs)

    def get_action(
            self,
            obs: torch.Tensor,
            deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        features = self.forward(obs)

        if self.action_type == ActionSpaceType.CONTINUOUS:
            mean = self.policy_mean(features)
            std = torch.exp(self.policy_log_std)
            dist = Normal(mean, std)

            if deterministic:
                action = mean
            else:
                action = dist.rsample()

            log_prob = dist.log_prob(action).sum(dim=-1)

        else:
            actions = []
            log_probs = []

            for head in self.policy_heads:
                logits = head(features)
                dist = Categorical(logits=logits)

                if deterministic:
                    action = logits.argmax(dim=-1)
                else:
                    action = dist.sample()

                actions.append(action)
                log_probs.append(dist.log_prob(action))

            action = torch.stack(actions, dim=-1)
            log_prob = torch.stack(log_probs, dim=-1).sum(dim=-1)

        return action, log_prob, {}

    def evaluate_actions(
            self,
            obs: torch.Tensor,
            action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        features = self.forward(obs)

        if self.action_type == ActionSpaceType.CONTINUOUS:
            mean = self.policy_mean(features)
            std = torch.exp(self.policy_log_std)
            dist = Normal(mean, std)

            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)

        else:
            log_probs = []
            entropies = []

            for i, head in enumerate(self.policy_heads):
                logits = head(features)
                dist = Categorical(logits=logits)
                log_probs.append(dist.log_prob(action[..., i]))
                entropies.append(dist.entropy())

            log_prob = torch.stack(log_probs, dim=-1).sum(dim=-1)
            entropy = torch.stack(entropies, dim=-1).sum(dim=-1)

        reward_value = self.reward_value_head(features).squeeze(-1)
        cost_values = [head(features).squeeze(-1) for head in self.cost_value_heads]

        return log_prob, entropy, reward_value, cost_values

    def get_values(
            self,
            obs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        features = self.forward(obs)
        reward_value = self.reward_value_head(features).squeeze(-1)
        cost_values = [head(features).squeeze(-1) for head in self.cost_value_heads]
        return reward_value, cost_values


def create_actor_critic(
        obs_dim: int,
        action_dim: int,
        n_cost_constraints: int,
        config: PPOConfig,
        action_type: ActionSpaceType,
        n_discrete_levels: int = 5,
        shared: bool = False
) -> nn.Module:
    """Factory function to create actor-critic network."""
    if shared:
        return SharedActorCritic(
            obs_dim, action_dim, n_cost_constraints,
            config, action_type, n_discrete_levels
        )
    else:
        return ActorCritic(
            obs_dim, action_dim, n_cost_constraints,
            config, action_type, n_discrete_levels
        )


if __name__ == "__main__":
    # Test networks
    print("Testing neural network architectures...")

    obs_dim = 64
    action_dim = 4
    n_constraints = 3
    batch_size = 32

    config = PPOConfig()

    # Test continuous policy
    print("\n--- Continuous Actor-Critic ---")
    ac = ActorCritic(
        obs_dim, action_dim, n_constraints, config,
        ActionSpaceType.CONTINUOUS
    )

    obs = torch.randn(batch_size, obs_dim)
    action, log_prob, _ = ac.get_action(obs)

    print(f"Action shape: {action.shape}")
    print(f"Log prob shape: {log_prob.shape}")

    log_prob, entropy, reward_v, cost_vs = ac.evaluate_actions(obs, action)
    print(f"Entropy shape: {entropy.shape}")
    print(f"Reward value shape: {reward_v.shape}")
    print(f"Cost values: {[v.shape for v in cost_vs]}")

    # Test discrete policy
    print("\n--- Discrete Actor-Critic ---")
    ac_discrete = ActorCritic(
        obs_dim, action_dim, n_constraints, config,
        ActionSpaceType.DISCRETE, n_discrete_levels=5
    )

    action, log_prob, _ = ac_discrete.get_action(obs)
    print(f"Discrete action shape: {action.shape}")
    print(f"Action values: {action[0]}")

    # Test shared network
    print("\n--- Shared Actor-Critic ---")
    ac_shared = SharedActorCritic(
        obs_dim, action_dim, n_constraints, config,
        ActionSpaceType.CONTINUOUS
    )

    action, log_prob, _ = ac_shared.get_action(obs)
    print(f"Shared action shape: {action.shape}")


    # Parameter count
    def count_params(model):
        return sum(p.numel() for p in model.parameters())


    print(f"\n--- Parameter counts ---")
    print(f"Separate AC: {count_params(ac):,}")
    print(f"Shared AC: {count_params(ac_shared):,}")