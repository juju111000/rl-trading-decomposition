"""
Trading Environment module.
Gymnasium-style CMDP environment for regime-conditioned trading.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config import (
    Config, ActionSpaceType, EnvironmentConfig,
    ConstraintConfig, ActionSpaceConfig
)
from portfolio_engine import PortfolioEngine, PortfolioState, RiskManager
from regime_model import RegimeModel
from policy_shield import PolicyShield, ShieldDiagnostics


class TradingCMDPEnv(gym.Env):
    """
    Constrained Markov Decision Process environment for trading.

    Observation:
        - Market features (per asset)
        - Portfolio state (weights, cash, drawdown, leverage)
        - Regime belief vector
        - Belief entropy

    Action:
        - Target portfolio weights (continuous) or exposure levels (discrete),
          passed through self.shield (if any) before being executed -- the
          shielded action, not the raw one, is what actually gets traded.

    Reward:
        - log(NAV_t / NAV_{t-1}) * environment.reward_scale, where NAV_t is
          marked once per step via PortfolioEngine.mark_to_market() AFTER
          this step's trade -- so transaction costs/slippage ARE reflected
          (they lower NAV before the price move is applied).
        - Optionally (environment.use_inactivity_penalty, default True): an
          additional shaping penalty subtracted when net exposure falls
          below 20%. This is reward shaping, not part of the CMDP cost
          vector below, and can be disabled for an ablation arm.

    Cost:
        - Vector of constraint violation costs [dd, turnover, leverage],
          each already expressed as excess-over-budget (e.g.
          cost_dd = max(0, drawdown - dd_limit)) -- see RiskManager.compute_costs().
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
            self,
            features: np.ndarray,  # Precomputed features (T, feature_dim)
            prices: Dict[str, np.ndarray],  # Price series per ticker
            tickers: List[str],
            config: Config,
            regime_model: Optional[RegimeModel] = None,
            shield: Optional[PolicyShield] = None,
            mode: str = "train"  # "train", "val", or "test"
    ):
        super().__init__()

        self.features = features
        self.prices = prices
        self.tickers = tickers
        self.config = config
        self.env_config = config.environment
        self.action_config = config.action_space
        self.constraint_config = config.constraints
        self.regime_model = regime_model
        self.shield = shield
        self.mode = mode

        self.n_assets = len(tickers)
        self.n_timesteps = len(features)
        self.feature_dim = features.shape[1]

        # Initialize components
        self.portfolio = PortfolioEngine(
            tickers=tickers,
            initial_capital=self.env_config.initial_capital,
            transaction_fee_rate=self.env_config.transaction_fee_rate,
            slippage_rate=self.env_config.slippage_rate,
            use_volatility_slippage=self.env_config.use_volatility_slippage,
            slippage_vol_multiplier=self.env_config.slippage_vol_multiplier
        )

        self.risk_manager = RiskManager(
            dd_limit=self.constraint_config.dd_limit,
            turnover_limit=self.constraint_config.turnover_limit,
            leverage_limit=self.constraint_config.leverage_limit,
            tail_quantile=self.constraint_config.tail_quantile,
            tail_window=self.constraint_config.tail_window
        )

        # Build observation and action spaces
        self._build_spaces()

        # Episode state
        self._current_step = 0
        self._episode_start = 0
        self._done = False

        # Logging
        self._episode_rewards: List[float] = []
        self._episode_costs: List[np.ndarray] = []
        self._shield_interventions: List[bool] = []

    def _build_spaces(self):
        """Define observation and action spaces."""
        # Observation space
        obs_dim = self.feature_dim

        if self.env_config.include_portfolio_state:
            obs_dim += PortfolioState.get_dim(self.n_assets)

        if self.env_config.include_regime_belief:
            obs_dim += self.config.regime.n_regimes

        if self.env_config.include_entropy:
            obs_dim += 1

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Action space
        if self.action_config.action_type == ActionSpaceType.DISCRETE:
            # Discrete: choose exposure level for each asset
            n_levels = len(self.action_config.discrete_levels)
            self.action_space = spaces.MultiDiscrete([n_levels] * self.n_assets)
        else:
            # Continuous: target weights
            low = self.action_config.weight_bounds[0]
            high = self.action_config.weight_bounds[1]
            self.action_space = spaces.Box(
                low=low,
                high=high,
                shape=(self.n_assets,),
                dtype=np.float32
            )

    def _get_prices(self, t: int) -> Dict[str, float]:
        """Get prices at time t."""
        return {ticker: self.prices[ticker][t] for ticker in self.tickers}

    def _get_observation(self) -> np.ndarray:
        """Construct observation vector."""
        t = self._current_step

        # Market features
        obs_parts = [self.features[t]]

        # Portfolio state
        if self.env_config.include_portfolio_state:
            prices = self._get_prices(t)
            state = self.portfolio.get_state(prices)
            obs_parts.append(state.to_array())

        # Regime belief
        if self.env_config.include_regime_belief and self.regime_model is not None:
            belief = self.regime_model.get_belief()
            obs_parts.append(belief)

        # Entropy
        if self.env_config.include_entropy and self.regime_model is not None:
            entropy = self.regime_model.get_normalized_entropy()
            obs_parts.append(np.array([entropy]))

        obs = np.concatenate(obs_parts).astype(np.float32)

        # Replace NaN/Inf with safe values
        obs = np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0)

        # Clip extreme values
        obs = np.clip(obs, -10.0, 10.0)

        return obs

    def _decode_action(self, action: Union[np.ndarray, int]) -> np.ndarray:
        """Convert action to target weights."""
        # Ensure action is an array
        action = np.atleast_1d(np.asarray(action)).flatten()

        if self.action_config.action_type == ActionSpaceType.DISCRETE:
            # Convert discrete indices to weight levels
            levels = np.array(self.action_config.discrete_levels)
            weights = levels[action]
        else:
            weights = action.copy()

        # Apply position size limits
        max_pos = self.action_config.max_position_size
        weights = np.clip(weights, -max_pos, max_pos)

        # Handle short constraint
        if not self.action_config.allow_short:
            weights = np.maximum(weights, 0)

        # Broker-style hard gross-exposure cap (margin reality, uniform
        # across every arm regardless of shield_type -- NOT the 1.5x
        # leverage_limit mandate, which stays a soft budget for the
        # Lagrangian/shield to learn; see ActionSpaceConfig.gross_leverage_cap).
        gross_cap = self.action_config.gross_leverage_cap
        if gross_cap is not None:
            gross = np.sum(np.abs(weights))
            if gross > gross_cap:
                weights = weights * (gross_cap / gross)

        return weights.astype(np.float32)

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        # Determine episode start
        if options and 'start_step' in options:
            self._episode_start = options['start_step']
        elif self.config.training.use_rolling_episodes and self.mode == "train":
            # Random starting point for training
            max_start = self.n_timesteps - self.config.training.episode_length - 1
            if max_start > 0:
                self._episode_start = self.np_random.integers(0, max_start)
            else:
                self._episode_start = 0
        else:
            self._episode_start = 0

        self._current_step = self._episode_start
        self._done = False

        # Reset components
        self.portfolio.reset()
        self.risk_manager.reset()

        if self.regime_model is not None:
            self.regime_model.reset()
            # Warm up regime model. Use enough history to fill a full
            # SMA(trend_ma_slow) window for HeuristicRegimeModel (other
            # model types just get extra, harmless warm-up steps).
            warmup_steps = max(20, getattr(self.config.regime, 'trend_ma_slow', 20))
            for t in range(max(0, self._episode_start - warmup_steps), self._current_step + 1):
                obs = self.features[t]
                self.regime_model.update(obs, prices=self._get_prices(t))

        # Clear episode logs
        self._episode_rewards = []
        self._episode_costs = []
        self._shield_interventions = []

        # Execution-smoothing state (see EnvironmentConfig.action_smoothing_beta):
        # w_{t-1} for the very first step of an episode is "flat" (no position
        # yet), matching the portfolio's actual starting state.
        self._smoothed_weights = np.zeros(self.n_assets, dtype=np.float32)

        # Decision-frequency state (see EnvironmentConfig.decision_frequency_days):
        # step 0 of every episode is always a decision day.
        self._days_since_decision = 0
        self._last_committed_weights = np.zeros(self.n_assets, dtype=np.float32)

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(
            self,
            action: Union[np.ndarray, int]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.

        Returns:
            observation, reward, terminated, truncated, info
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset().")

        t = self._current_step

        # Get current prices
        current_prices = self._get_prices(t)

        # Decision-frequency gate (see EnvironmentConfig.decision_frequency_days):
        # on a non-decision day, HOLD -- target = actual current weights, so
        # execute_trades sees zero delta and no trade happens (NOT "re-target
        # yesterday's decision", which would still trade to correct for price
        # drift). Applied uniformly to every arm, before the shield.
        decision_freq = max(1, self.env_config.decision_frequency_days)
        is_decision_day = (self._days_since_decision % decision_freq == 0)
        self._days_since_decision += 1

        if is_decision_day:
            # Decode action to target weights
            target_weights = self._decode_action(action)

            # Execution-model smoothing (see EnvironmentConfig.action_smoothing_beta):
            # w_t = (1-beta)*w_{t-1} + beta*a_t. Applied uniformly to every arm
            # BEFORE the shield (if any) ever sees the target -- an execution-lag
            # reality, not a safety mechanism, so shield=none gets the identical
            # treatment as a shielded arm.
            beta = self.env_config.action_smoothing_beta
            if beta is not None:
                target_weights = (1.0 - beta) * self._smoothed_weights + beta * target_weights
                target_weights = target_weights.astype(np.float32)
            self._smoothed_weights = target_weights

            # No-trade band (see EnvironmentConfig.no_trade_band): per asset,
            # only actually change a position if the move from its CURRENT
            # actual weight exceeds the band -- otherwise hold that asset.
            # Applied uniformly, before the shield.
            band = self.env_config.no_trade_band
            if band is not None:
                current_weights = self.portfolio.compute_weights(current_prices)
                weight_delta = target_weights - current_weights
                target_weights = np.where(
                    np.abs(weight_delta) > band, target_weights, current_weights
                ).astype(np.float32)

            self._last_committed_weights = target_weights
        else:
            target_weights = self.portfolio.compute_weights(current_prices)

        # Get regime state for shield
        if self.regime_model is not None:
            regime_belief = self.regime_model.get_belief()
            entropy = self.regime_model.get_entropy()
        else:
            regime_belief = np.ones(self.config.regime.n_regimes) / self.config.regime.n_regimes
            entropy = 0.0

        # Apply shield
        shield_diag = None
        if self.shield is not None:
            portfolio_state = self.portfolio.get_state(current_prices)
            safe_weights, shield_diag = self.shield.apply(
                target_weights, portfolio_state, regime_belief, entropy,
                self.portfolio.get_nav_history()
            )
            self._shield_interventions.append(shield_diag.was_modified)
        else:
            safe_weights = target_weights
            self._shield_interventions.append(False)

        # Ensure safe_weights is always a 1D array (handle scalar case)
        safe_weights = np.atleast_1d(np.asarray(safe_weights)).flatten()

        # Compute turnover before trading
        portfolio_state = self.portfolio.get_state(current_prices)
        turnover = self.portfolio.compute_turnover(safe_weights, current_prices)

        # Execute trades (mutates positions/cash; NAV is not recorded yet)
        transaction_cost, trade_info = self.portfolio.execute_trades(
            safe_weights, current_prices
        )

        # Advance to next time step
        self._current_step += 1

        # Check termination conditions
        terminated = False
        truncated = False

        if self._current_step >= self.n_timesteps - 1:
            truncated = True
            self._done = True
        elif (self._current_step - self._episode_start) >= self.config.training.episode_length:
            truncated = True
            self._done = True

        # Determine mark-to-market prices for this step
        if not self._done:
            new_prices = self._get_prices(self._current_step)

            # Update regime model
            if self.regime_model is not None:
                self.regime_model.update(self.features[self._current_step], prices=new_prices)
        else:
            new_prices = current_prices

        # Single NAV mark per step: reflects this step's transaction costs
        # AND any price movement together (cost-inclusive reward).
        log_return = self.portfolio.mark_to_market(
            new_prices, prev_prices=(current_prices if not self._done else None)
        )

        # Get updated state
        new_state = self.portfolio.get_state(new_prices)

        # Scale reward for better learning (daily returns are very small ~0.001)
        reward_scale = getattr(self.env_config, 'reward_scale', 100.0)
        reward = float(log_return * reward_scale)

        # Add inactivity penalty - penalize for holding too little position.
        # This is reward shaping (prevents the agent from learning to just
        # hold cash), not part of the CMDP cost/constraint machinery, and is
        # gated so it can be disabled for an ablation arm.
        if getattr(self.env_config, 'use_inactivity_penalty', True):
            current_exposure = abs(new_state.net_exposure)  # Total invested (absolute)
            min_exposure = 0.2  # Expect at least 20% invested
            inactivity_penalty_rate = getattr(self.env_config, 'inactivity_penalty', 0.01)

            if current_exposure < min_exposure:
                # Penalty proportional to how underinvested we are
                inactivity_penalty = inactivity_penalty_rate * (min_exposure - current_exposure) * reward_scale
                reward -= inactivity_penalty

        # Compute cost vector
        costs = self.risk_manager.compute_costs(
            new_state, turnover, log_return
        )
        cost_vector = self.risk_manager.get_cost_vector(costs)

        # Store episode data
        self._episode_rewards.append(reward)
        self._episode_costs.append(cost_vector)

        # Build info dict
        info = self._get_info()
        info.update({
            'cost_vec': cost_vector,
            'costs': costs,
            'turnover': turnover,
            'transaction_cost': transaction_cost,
            'portfolio_state': new_state,
            'log_return': log_return,
            'nav': new_state.nav,
            'drawdown': new_state.current_drawdown,
            'leverage': new_state.gross_leverage,
        })

        if shield_diag is not None:
            info['shield'] = shield_diag.to_dict()

        if self.regime_model is not None:
            info['regime_belief'] = regime_belief.tolist()
            info['regime_entropy'] = entropy
            info['regime_resolved'] = self.regime_model.is_resolved()

        obs = self._get_observation()

        return obs, reward, terminated, truncated, info

    def _get_info(self) -> Dict:
        """Get current info dict."""
        return {
            'step': self._current_step,
            'episode_start': self._episode_start,
            'mode': self.mode
        }

    def get_episode_statistics(self) -> Dict[str, float]:
        """Compute statistics for completed episode."""
        if len(self._episode_rewards) == 0:
            return {}

        rewards = np.array(self._episode_rewards)
        costs = np.array(self._episode_costs)

        stats = {
            'episode_return': float(np.sum(rewards)),
            'episode_length': len(rewards),
            'avg_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
        }

        # Cost statistics
        if costs.size > 0:
            stats['avg_cost_dd'] = float(np.mean(costs[:, 0]))
            stats['avg_cost_turnover'] = float(np.mean(costs[:, 1]))
            stats['avg_cost_leverage'] = float(np.mean(costs[:, 2]))
            stats['violation_rate_dd'] = float(np.mean(costs[:, 0] > 0))
            stats['violation_rate_turnover'] = float(np.mean(costs[:, 1] > 0))
            stats['violation_rate_leverage'] = float(np.mean(costs[:, 2] > 0))

        # Shield statistics
        if self._shield_interventions:
            stats['shield_intervention_rate'] = float(np.mean(self._shield_interventions))

        # Portfolio metrics
        metrics = self.portfolio.get_metrics()
        stats.update(metrics)

        return stats

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render current state (placeholder)."""
        if mode == "human":
            state = self.portfolio.get_state(self._get_prices(self._current_step))
            print(f"Step {self._current_step}: NAV=${state.nav:,.2f}, "
                  f"DD={state.current_drawdown:.2%}, Lev={state.gross_leverage:.2f}")
            return None
        return None

    def close(self):
        """Clean up resources."""
        pass


class VectorizedTradingEnv:
    """
    Vectorized environment for parallel rollouts.
    Wraps multiple TradingCMDPEnv instances.
    """

    def __init__(
            self,
            env_fns: List[callable],
            n_envs: int
    ):
        self.n_envs = n_envs
        self.envs = [fn() for fn in env_fns]

        # Get spaces from first env
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def reset(
            self,
            seed: Optional[int] = None
    ) -> Tuple[np.ndarray, List[Dict]]:
        """Reset all environments."""
        observations = []
        infos = []

        for i, env in enumerate(self.envs):
            env_seed = seed + i if seed is not None else None
            obs, info = env.reset(seed=env_seed)
            observations.append(obs)
            infos.append(info)

        return np.stack(observations), infos

    def step(
            self,
            actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """Step all environments."""
        observations = []
        rewards = []
        terminateds = []
        truncateds = []
        infos = []

        for i, (env, action) in enumerate(zip(self.envs, actions)):
            obs, reward, terminated, truncated, info = env.step(action)

            # Auto-reset on done
            if terminated or truncated:
                final_info = info.copy()
                final_info['terminal_observation'] = obs
                final_info['episode'] = env.get_episode_statistics()
                obs, _ = env.reset()
                info = final_info

            observations.append(obs)
            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(truncated)
            infos.append(info)

        return (
            np.stack(observations),
            np.array(rewards),
            np.array(terminateds),
            np.array(truncateds),
            infos
        )

    def close(self):
        """Close all environments."""
        for env in self.envs:
            env.close()


def make_env(
        features: np.ndarray,
        prices: Dict[str, np.ndarray],
        tickers: List[str],
        config: Config,
        regime_model: Optional[RegimeModel] = None,
        shield: Optional[PolicyShield] = None,
        mode: str = "train",
        seed: int = 42
) -> TradingCMDPEnv:
    """Factory function to create trading environment."""
    env = TradingCMDPEnv(
        features=features,
        prices=prices,
        tickers=tickers,
        config=config,
        regime_model=regime_model,
        shield=shield,
        mode=mode
    )
    env.reset(seed=seed)
    return env


if __name__ == "__main__":
    # Test environment
    from data_loader import SyntheticDataGenerator
    from features import FeatureBuilder, FeatureConfig
    from regime_model import HeuristicRegimeModel
    from policy_shield import create_shield
    from config import Config

    print("Testing TradingCMDPEnv...")

    # Generate data
    gen = SyntheticDataGenerator(n_assets=4, n_days=500)
    data, _ = gen.generate_regime_switching_data()

    # Get tickers (preserve column order, not set()'s hash-dependent order)
    tickers = list(dict.fromkeys(col[0] for col in data.columns))

    # Compute features
    feature_config = FeatureConfig()
    builder = FeatureBuilder(feature_config)
    features_df = builder.compute_features(data)
    features = features_df.values

    # Get prices
    prices = {t: data[(t, 'Close')].values for t in tickers}

    # Create config
    config = Config()
    config.training.episode_length = 100

    # Create regime model
    regime_config = config.regime
    regime_model = HeuristicRegimeModel(regime_config)
    regime_model.fit(data)

    # Create shield
    shield = create_shield(
        config.shield, config.constraints, len(tickers)
    )

    # Create environment
    env = TradingCMDPEnv(
        features=features,
        prices=prices,
        tickers=tickers,
        config=config,
        regime_model=regime_model,
        shield=shield
    )

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Run test episode
    obs, info = env.reset(seed=42)
    print(f"\nInitial observation shape: {obs.shape}")

    total_reward = 0
    done = False
    step = 0

    while not done and step < 50:
        # Random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step += 1

        if step % 10 == 0:
            print(f"Step {step}: reward={reward:.4f}, "
                  f"NAV=${info['nav']:,.0f}, DD={info['drawdown']:.2%}")

    stats = env.get_episode_statistics()
    print(f"\nEpisode statistics:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")