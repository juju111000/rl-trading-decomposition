"""
Configuration module for Regime-Conditioned Shielded PPO-Lagrangian Trading System.
Uses dataclasses for type-safe, validated configuration.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Literal, Union
from enum import Enum
import yaml
import json


class ActionSpaceType(Enum):
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"


class RegimeModelType(Enum):
    HEURISTIC = "heuristic"
    HMM = "hmm"
    LEARNED = "learned"


class BeliefFormationType(Enum):
    """How HeuristicRegimeModel turns daily regime scores into a belief
    vector. All three share the same trend/vol scoring; see regime_model.py."""
    SOFTMAX_TEMPERATURE = "softmax_temperature"  # (A) softmax(scores / T), then EMA-smoothed over days
    HARD_EMA = "hard_ema"  # (B) argmax(scores) -> onehot -> EMA belief
    HARD_EMA_HYSTERESIS = "hard_ema_hysteresis"  # (C) B, but the argmax must persist hysteresis_days before it flips


class ShieldType(Enum):
    NONE = "none"  # No shield attached -- create_shield() returns None; the environment executes the raw proposed action unmodified
    RULE_BASED = "rule_based"
    PROJECTION = "projection"
    UNCERTAINTY_AWARE = "uncertainty_aware"


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    tickers: List[str] = field(default_factory=lambda: ["^GSPC", "^GDAXI", "000001.SS", "CL=F"])
    start_date: str = "2005-01-01"
    end_date: str = "2024-12-31"
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    frequency: str = "1d"
    handle_missing: Literal["ffill", "drop", "interpolate"] = "ffill"
    use_adjusted_close: bool = True
    cache_dir: str = "./data_cache"


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    return_windows: List[int] = field(default_factory=lambda: [1, 5, 20])
    volatility_windows: List[int] = field(default_factory=lambda: [20, 60])
    ma_windows: List[int] = field(default_factory=lambda: [20, 50, 200])
    roc_windows: List[int] = field(default_factory=lambda: [20])
    use_atr: bool = True
    atr_window: int = 14
    use_cross_asset_features: bool = True
    correlation_window: int = 60
    normalization_window: int = 252  # ~1 trading year
    clip_zscore: float = 5.0


@dataclass
class RegimeConfig:
    """Configuration for regime inference module."""
    model_type: RegimeModelType = RegimeModelType.HEURISTIC
    n_regimes: int = 3  # e.g., bull, bear, sideways
    # Heuristic model params
    vol_percentile_thresholds: List[float] = field(default_factory=lambda: [0.33, 0.67])
    trend_ma_fast: int = 20
    trend_ma_slow: int = 60
    # HMM params
    hmm_n_iter: int = 100
    hmm_covariance_type: str = "full"
    # Belief smoothing (used by all three HeuristicRegimeModel belief-formation
    # candidates below -- as the softmax-EMA alpha for A, and as the hard-EMA
    # alpha for B/C)
    belief_ema_alpha: float = 0.1
    # Belief formation mechanism (HeuristicRegimeModel only): how daily
    # trend/vol scores become a belief vector. CANONICAL DEFAULT: candidate C
    # (hard EMA + hysteresis), selected via the pre-registered train-segment
    # sweep in results/stage2_diagnostics/recalibration_ranking.csv -- the
    # original always-on softmax (candidate A, temperature=1.0) left entropy
    # above 0.7 on 87-100% of test days across all 3 real windows, including
    # the calm 2015-17 window; C(alpha=0.1) resolved calm-period entropy to
    # a decisive read (median 0.034) while reducing regime-flip chatter
    # relative to its non-hysteresis counterpart B(alpha=0.1). Set
    # belief_formation=SOFTMAX_TEMPERATURE explicitly to recover the
    # original behavior for comparison.
    belief_formation: BeliefFormationType = BeliefFormationType.HARD_EMA_HYSTERESIS
    softmax_temperature: float = 1.0  # (A) softmax(scores / T); T=1.0 == original behavior; unused by the canonical C default
    hysteresis_days: int = 3  # (C) consecutive days the argmax must persist before a hard regime flip commits


@dataclass
class ConstraintConfig:
    """Configuration for CMDP constraints."""
    # Drawdown constraints (RELAXED for better learning)
    dd_limit: float = 0.20  # Soft constraint threshold (20%)
    dd_stop: float = 0.30  # Hard stop threshold (30%)
    # Turnover constraints (RELAXED)
    turnover_limit: float = 1.0  # Soft limit per step
    turnover_cap: float = 2.0  # Hard cap per step
    # Leverage constraints
    leverage_limit: float = 1.5  # Soft limit
    leverage_cap: float = 2.0  # Hard cap
    # Tail risk (optional)
    use_tail_cost: bool = False
    tail_quantile: float = 0.05  # VaR threshold
    tail_window: int = 20

    # Dual-update budgets (d_k in the CMDP formulation: lambda_k is pushed up
    # whenever mean_cost_k > d_k, down otherwise). These are 0.0 by design,
    # NOT because the per-constraint budgets above are unset: dd_limit /
    # turnover_limit / leverage_limit above ARE the real budgets, and
    # RiskManager.compute_costs() (portfolio_engine.py) already subtracts
    # them when computing each step's cost, e.g.
    # cost_dd = max(0, current_drawdown - dd_limit). By the time
    # PPOLagrangianTrainer.update_dual_variables() (trainer.py) sees these
    # costs, they're already "excess over budget", so comparing them against
    # d_k=0 downstream is the mathematically correct, not-yet-double-counted
    # thing to do. Set a nonzero value here only if you want the dual update
    # to additionally tolerate some average excess-over-budget before
    # lambda_k starts climbing (e.g. 0.01 to permit brief, minor
    # overshoots) -- trainer.py reads this directly as constraint_limits.
    constraint_budgets: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass
class ShieldConfig:
    """Configuration for policy shield."""
    shield_type: ShieldType = ShieldType.UNCERTAINTY_AWARE
    # Uncertainty-aware tightening
    entropy_tightening_kappa: float = 0.3  # Reduced from 0.5
    # Regime-aware adjustments (RELAXED)
    high_vol_position_cap: float = 0.5  # Increased from 0.3
    high_vol_leverage_cap: float = 1.5  # Increased from 1.0
    # Transition detection
    entropy_spike_threshold: float = 0.8  # Relative to max entropy

    # Component toggles for UncertaintyAwareShield, for the shield-component
    # ablation (main.SHIELD_COMPONENT_VARIANTS: full / no_entropy_term /
    # no_bear_cap / no_dd_scaling). All default True, i.e. the "full" shield.
    use_entropy_tightening: bool = True  # if False, leverage/turnover/position caps use their base values (tighten=0)
    use_bear_cap: bool = True  # if False, the high-confidence-Bear conservative cap (0.2) never applies
    use_dd_scaling: bool = True  # if False, drawdown never scales the position (dd_scale always 1.0)

    # Drawdown anchor for the shield's own DD-based rules (dd_stop checks and
    # UncertaintyAwareShield's continuous dd_scale) -- NOT the RiskManager/
    # reward cost signal the dual mechanism trains against, which stays
    # peak-anchored regardless of this setting.
    # "peak": current/original behavior -- anchored to the all-time NAV peak
    #     (PortfolioState.current_drawdown), so a single deep drawdown keeps
    #     the shield's position cap depressed indefinitely even long after
    #     NAV has recovered and stabilized above the crash (the "absorbing
    #     trap": the shield never releases because the OLD peak is still the
    #     reference point).
    # "trailing": anchored to the max NAV over the last dd_anchor_window
    #     trading days instead -- once a crash is further in the past than
    #     the window, the anchor forgets it and the drawdown (and therefore
    #     the shield's position cap) releases even if NAV never fully
    #     reclaimed the old all-time high.
    dd_anchor: Literal["peak", "trailing"] = "peak"
    dd_anchor_window: int = 63  # ~1 trading quarter; only used when dd_anchor="trailing"


@dataclass
class ActionSpaceConfig:
    """Configuration for action space."""
    action_type: ActionSpaceType = ActionSpaceType.CONTINUOUS  # Use continuous for trading
    # Discrete action space
    discrete_levels: List[float] = field(default_factory=lambda: [-1.0, -0.5, 0.0, 0.5, 1.0])
    # Continuous action space
    weight_bounds: tuple = (-1.0, 1.0)
    # Portfolio constraints
    allow_short: bool = True
    max_position_size: float = 0.5
    # Broker-style hard gross-exposure cap (sum of |weight| across assets),
    # applied uniformly to every arm's decoded action regardless of shield_type
    # -- a margin-account reality, not a learned or shield-based constraint.
    # None disables it (existing behavior). This is NOT the 1.5x leverage_limit
    # mandate in ConstraintConfig -- that stays a soft budget the
    # Lagrangian/shield must learn to respect; this cap only prevents the
    # physically-impossible (a broker would never let gross exposure exceed
    # this regardless of what the model wants).
    gross_leverage_cap: Optional[float] = None


@dataclass
class PPOConfig:
    """Configuration for PPO-Lagrangian algorithm."""
    # Network architecture
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 256])
    activation: str = "tanh"
    # PPO hyperparameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    clip_value_loss: bool = True
    value_clip_epsilon: float = 0.2
    # Entropy bonus (increased for more exploration)
    entropy_coef: float = 0.05  # Increased from 0.01
    entropy_decay: float = 0.998
    min_entropy_coef: float = 0.005
    # Learning rates
    policy_lr: float = 1e-4  # Reduced for stability
    value_lr: float = 5e-4
    dual_lr: float = 0.005  # Reduced for stability
    # Training
    n_epochs: int = 10
    batch_size: int = 64
    max_grad_norm: float = 0.5
    # Dual variable. A single float applies uniformly to all 3 costs
    # (existing behavior); a length-3 list gives each its own warm-start,
    # order [dd, turnover, leverage] (see portfolio_engine.RiskManager.get_cost_vector) --
    # e.g. for "the policy tolerates high lambda but not FAST-MOVING lambda"
    # (Phase 3 Step 5 finding): start high, let dual_lr move slowly from there.
    lambda_init: Union[float, List[float]] = 0.0
    lambda_max: float = 10.0  # Reduced from 100 to prevent over-penalization
    # Cost critic
    use_cost_critic: bool = True

    # Dual update mechanism. "additive" is the original pure integral update
    # (lambda += dual_lr * violation, see PPOLagrangianTrainer.update_dual_variables).
    # "pid" is PID-Lagrangian (Stooke et al. 2020, "Responsive Safety in RL by
    # PID Lagrangian Methods"): lambda = max(0, Kp*violation + Ki*integral(violation)
    # + Kd*d(violation)/dt), which reacts to the CURRENT violation (P) and its
    # trend (D) instead of only its accumulated history (I) -- intended to
    # respond faster than a pure-integral update at a given learning rate, and
    # to damp rather than overshoot once violations start falling.
    dual_update_method: Literal["additive", "pid"] = "additive"
    pid_kp: float = 0.0
    pid_ki: float = 0.0
    pid_kd: float = 0.0

    # Policy action-noise controls (networks.ContinuousPolicy). log_std_init
    # now actually reaches the network (previously hardcoded -0.5 there,
    # unreachable from config) -- default unchanged, so this alone changes
    # no behavior. log_std_decay_floor is an optional DIRECT lever on action
    # noise, separate from/in addition to the entropy bonus's indirect
    # pressure (see PPOLagrangianTrainer._entropy_coef_for_iteration): when
    # set, the trainer anneals a ceiling on the state-independent log_std
    # parameter from log_std_init down to this floor over training, using
    # the same progress fraction as the entropy anneal. None (default)
    # disables it -- log_std is purely learned, matching prior behavior.
    log_std_init: float = -0.5
    log_std_decay_floor: Optional[float] = None


@dataclass
class TrainingConfig:
    """Configuration for training loop."""
    total_timesteps: int = 1_000_000
    rollout_length: int = 2048
    eval_frequency: int = 10_000
    save_frequency: int = 50_000
    log_frequency: int = 1000
    # Episode structure
    episode_length: int = 252  # ~1 trading year
    use_rolling_episodes: bool = True
    rolling_step: int = 63  # ~1 quarter
    # Early stopping
    patience: int = 10
    min_improvement: float = 0.01
    # Random seed
    seed: int = 42


@dataclass
class EnvironmentConfig:
    """Configuration for trading environment."""
    initial_capital: float = 1_000_000.0
    transaction_fee_rate: float = 0.001  # 10 bps
    slippage_rate: float = 0.0005  # 5 bps
    use_volatility_slippage: bool = True
    slippage_vol_multiplier: float = 0.1
    # Reward scaling (daily returns are ~0.001, scale up for better learning)
    reward_scale: float = 100.0
    # Inactivity penalty (penalize for holding less than 20% position). This
    # is reward shaping on top of the raw log-return reward -- not part of
    # the CMDP's cost/constraint machinery -- so it's gated behind its own
    # flag (default ON, matching prior always-on behavior) to allow a clean
    # ablation arm without it.
    use_inactivity_penalty: bool = True
    inactivity_penalty: float = 0.02  # Penalty rate per day for being underinvested
    # Execution-model smoothing, applied uniformly to every arm's decoded
    # target weights BEFORE the shield (if any) ever sees them -- an
    # execution-lag/broker-throttling reality, not a safety mechanism, so it
    # must not act differently for shield=none vs shielded arms:
    #   w_t = (1 - beta) * w_{t-1} + beta * a_t
    # None (default) disables it (existing behavior, a_t executed as-is).
    action_smoothing_beta: Optional[float] = None
    # No-trade band: per-asset, only execute the change from CURRENT actual
    # weight to the (post-smoothing) target if |target - current| exceeds
    # this threshold; otherwise hold that asset's position unchanged this
    # step. Applied uniformly to every arm, same as action_smoothing_beta --
    # an execution-cost-avoidance reality (real brokers/desks don't bother
    # trading a position 0.3% away from target), not a safety mechanism.
    # None (default) disables it.
    no_trade_band: Optional[float] = None
    # Decision frequency: the policy's action is decoded into a genuinely
    # NEW target only every decision_frequency_days steps (default 1 = every
    # day, existing behavior); on the days in between, the position is HELD
    # (target = actual current weights, so execute_trades sees zero delta --
    # not "re-target a stale value", which would still trade to correct for
    # price drift). NAV/rewards/costs are still computed every day regardless.
    # Applied uniformly to every arm.
    decision_frequency_days: int = 1
    # Observation
    include_portfolio_state: bool = True
    include_regime_belief: bool = True
    include_entropy: bool = True
    observation_history: int = 1  # Number of past observations


@dataclass
class EvaluationConfig:
    """Configuration for evaluation and metrics."""
    # Risk-free rate for Sharpe
    risk_free_rate: float = 0.02
    # Rolling windows for metrics
    sharpe_window: int = 252
    volatility_window: int = 20
    # Benchmark
    benchmark_ticker: str = "^GSPC"
    # Output
    results_dir: str = "./results"
    save_plots: bool = True
    save_csv: bool = True


@dataclass
class Config:
    """Master configuration combining all sub-configs."""
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    constraints: ConstraintConfig = field(default_factory=ConstraintConfig)
    shield: ShieldConfig = field(default_factory=ShieldConfig)
    action_space: ActionSpaceConfig = field(default_factory=ActionSpaceConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # Experiment metadata
    experiment_name: str = "regime_cmdp_trading"
    description: str = ""

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls._from_dict(data)

    @classmethod
    def from_json(cls, path: str) -> "Config":
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict) -> "Config":
        """Create Config from dictionary."""
        config = cls()

        if 'data' in data:
            config.data = DataConfig(**data['data'])
        if 'features' in data:
            config.features = FeatureConfig(**data['features'])
        if 'regime' in data:
            regime_data = data['regime'].copy()
            if 'model_type' in regime_data:
                regime_data['model_type'] = RegimeModelType(regime_data['model_type'])
            if 'belief_formation' in regime_data:
                regime_data['belief_formation'] = BeliefFormationType(regime_data['belief_formation'])
            config.regime = RegimeConfig(**regime_data)
        if 'constraints' in data:
            config.constraints = ConstraintConfig(**data['constraints'])
        if 'shield' in data:
            shield_data = data['shield'].copy()
            if 'shield_type' in shield_data:
                shield_data['shield_type'] = ShieldType(shield_data['shield_type'])
            config.shield = ShieldConfig(**shield_data)
        if 'action_space' in data:
            action_data = data['action_space'].copy()
            if 'action_type' in action_data:
                action_data['action_type'] = ActionSpaceType(action_data['action_type'])
            config.action_space = ActionSpaceConfig(**action_data)
        if 'ppo' in data:
            config.ppo = PPOConfig(**data['ppo'])
        if 'training' in data:
            config.training = TrainingConfig(**data['training'])
        if 'environment' in data:
            config.environment = EnvironmentConfig(**data['environment'])
        if 'evaluation' in data:
            config.evaluation = EvaluationConfig(**data['evaluation'])
        if 'experiment_name' in data:
            config.experiment_name = data['experiment_name']
        if 'description' in data:
            config.description = data['description']

        return config

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'experiment_name': self.experiment_name,
            'description': self.description,
            'data': self.data.__dict__,
            'features': self.features.__dict__,
            'regime': {**self.regime.__dict__, 'model_type': self.regime.model_type.value,
                       'belief_formation': self.regime.belief_formation.value},
            'constraints': self.constraints.__dict__,
            'shield': {**self.shield.__dict__, 'shield_type': self.shield.shield_type.value},
            'action_space': {**self.action_space.__dict__, 'action_type': self.action_space.action_type.value},
            'ppo': self.ppo.__dict__,
            'training': self.training.__dict__,
            'environment': self.environment.__dict__,
            'evaluation': self.evaluation.__dict__,
        }

    def save_yaml(self, path: str):
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def save_json(self, path: str):
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def get_default_config() -> Config:
    """Return default configuration."""
    return Config()


def get_ablation_configs() -> dict:
    """Return configurations for ablation studies."""
    configs = {}

    # Baseline: Vanilla PPO (no shield, no constraints)
    vanilla_ppo = Config()
    vanilla_ppo.experiment_name = "vanilla_ppo"
    vanilla_ppo.shield.shield_type = ShieldType.RULE_BASED
    vanilla_ppo.constraints.dd_limit = 1.0  # Effectively disabled
    vanilla_ppo.constraints.turnover_limit = 100.0
    vanilla_ppo.constraints.leverage_limit = 100.0
    configs['vanilla_ppo'] = vanilla_ppo

    # Shield-only PPO (hard constraints only, no dual learning)
    shield_only = Config()
    shield_only.experiment_name = "shield_only_ppo"
    shield_only.ppo.dual_lr = 0.0  # Disable dual updates
    configs['shield_only'] = shield_only

    # PPO-Lagrangian only (no shield)
    lagrangian_only = Config()
    lagrangian_only.experiment_name = "ppo_lagrangian_only"
    lagrangian_only.shield.shield_type = ShieldType.RULE_BASED
    lagrangian_only.constraints.dd_stop = 1.0  # Disable hard stop
    lagrangian_only.constraints.turnover_cap = 100.0
    lagrangian_only.constraints.leverage_cap = 100.0
    configs['lagrangian_only'] = lagrangian_only

    # Full method
    full_method = Config()
    full_method.experiment_name = "full_regime_cmdp"
    configs['full_method'] = full_method

    # Ablation: No regime conditioning
    no_regime = Config()
    no_regime.experiment_name = "no_regime_conditioning"
    no_regime.environment.include_regime_belief = False
    no_regime.environment.include_entropy = False
    configs['no_regime'] = no_regime

    return configs