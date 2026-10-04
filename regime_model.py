"""
Regime inference module.
Provides belief vectors over latent market regimes.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.special import softmax

from config import RegimeConfig, RegimeModelType, BeliefFormationType


class RegimeModel(ABC):
    """
    Abstract base class for regime inference models.
    All regime models output a belief vector over K regimes.
    """

    def __init__(self, config: RegimeConfig):
        self.config = config
        self.n_regimes = config.n_regimes
        self._current_belief: Optional[np.ndarray] = None

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> None:
        """Fit the regime model to historical data."""
        pass

    @abstractmethod
    def update(
            self,
            observation: np.ndarray,
            prices: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Update belief given new observation.

        Args:
            observation: Current market features (used by feature-vector-
                based models, e.g. HMM/Learned).
            prices: Optional dict of ticker -> current price. Models that
                compute signals directly from named OHLCV data (e.g.
                HeuristicRegimeModel's SMA crossover) use this instead of
                guessing positions in `observation`.

        Returns:
            Belief vector b_t ∈ R^K, sum to 1.
        """
        pass

    @abstractmethod
    def get_belief(self) -> np.ndarray:
        """Return current belief vector."""
        pass

    def get_entropy(self) -> float:
        """
        Compute entropy of current belief.
        H(b) = -sum_k b_k * log(b_k)
        High entropy = uncertain about regime (likely transition).
        """
        belief = self.get_belief()
        # Avoid log(0)
        belief = np.clip(belief, 1e-10, 1.0)
        return -np.sum(belief * np.log(belief))

    def get_max_entropy(self) -> float:
        """Return maximum possible entropy (uniform distribution)."""
        return np.log(self.n_regimes)

    def get_normalized_entropy(self) -> float:
        """Return entropy normalized to [0, 1]."""
        return self.get_entropy() / self.get_max_entropy()

    def is_resolved(self) -> bool:
        """Whether the current belief reflects a settled regime call rather
        than one mid-transition. Base implementation: models without a
        committed/pending hysteresis state machine are always resolved --
        see HeuristicRegimeModel's override for the one case where this
        actually varies."""
        return True

    def reset(self) -> None:
        """Reset to uniform prior."""
        self._current_belief = np.ones(self.n_regimes) / self.n_regimes


class HeuristicRegimeModel(RegimeModel):
    """
    Rule-based regime model using trend and volatility indicators computed
    directly from the market ticker's own Close price (named OHLCV data),
    never from positional indices into the generic per-step feature vector.

    Regimes:
        0: Bull (SMA(trend_ma_fast) > SMA(trend_ma_slow), low volatility)
        1: Bear (SMA(trend_ma_fast) < SMA(trend_ma_slow), high volatility)
        2: Sideways (no clear trend, medium volatility)
    """

    def __init__(self, config: RegimeConfig):
        super().__init__(config)
        self._vol_thresholds: Optional[Tuple[float, float]] = None
        self._market_ticker: Optional[str] = None
        self._close_history: List[float] = []
        self._ema_alpha = config.belief_ema_alpha
        self._trend_ma_fast = config.trend_ma_fast
        self._trend_ma_slow = config.trend_ma_slow
        # Hard-assignment state for candidates B/C (belief_formation in
        # {HARD_EMA, HARD_EMA_HYSTERESIS}); unused by candidate A.
        self._committed_regime: Optional[int] = None
        self._pending_regime: Optional[int] = None
        self._pending_count: int = 0
        self.reset()

    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit volatility thresholds from historical data, and seed the
        close-price history used for the SMA crossover trend signal.
        Uses first ticker (in config-declared order) as market proxy.
        """
        # Preserve column order (first occurrence) rather than set()'s
        # hash-dependent order: market_ticker below is relied on as the
        # primary / benchmark asset, so it must match the config-declared order.
        tickers = list(dict.fromkeys(col[0] for col in data.columns))
        self._market_ticker = tickers[0]
        close = data[(self._market_ticker, 'Close')]

        # Seed the rolling close-price buffer so update() has a full
        # SMA(trend_ma_slow) window available as soon as fit() has run.
        self._close_history = close.tail(self._trend_ma_slow).tolist()

        # Compute historical volatility
        returns = np.log(close / close.shift(1)).dropna()
        rolling_vol = returns.rolling(20).std() * np.sqrt(252)
        rolling_vol = rolling_vol.dropna()

        # Set volatility thresholds based on percentiles
        low_pct, high_pct = self.config.vol_percentile_thresholds
        self._vol_thresholds = (
            rolling_vol.quantile(low_pct),
            rolling_vol.quantile(high_pct)
        )

        print(f"HeuristicRegimeModel fitted:")
        print(f"  Vol thresholds: {self._vol_thresholds}")

    def update(
            self,
            observation: np.ndarray,
            prices: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Update belief from the market ticker's current Close price.

        Trend: SMA(trend_ma_fast) vs SMA(trend_ma_slow) crossover on the
        market ticker's own close-price history (positive = bullish, fast
        SMA above slow SMA). Volatility: realized (annualized) volatility
        of daily log returns over the same history, compared against the
        percentile thresholds fit() computed.

        `observation` (the generic per-step feature vector) is accepted for
        interface compatibility with other RegimeModel implementations but
        is not used here -- `prices` (ticker -> current Close) is what
        actually drives this model.
        """
        if (
                prices is not None
                and self._market_ticker is not None
                and self._market_ticker in prices
        ):
            self._close_history.append(float(prices[self._market_ticker]))
            max_history = self._trend_ma_slow + 5
            if len(self._close_history) > max_history:
                self._close_history = self._close_history[-max_history:]

        close_series = pd.Series(self._close_history)

        # Trend: SMA(fast) vs SMA(slow) crossover.
        if len(close_series) >= self._trend_ma_slow:
            sma_fast = close_series.iloc[-self._trend_ma_fast:].mean()
            sma_slow = close_series.iloc[-self._trend_ma_slow:].mean()
            trend_signal = (sma_fast - sma_slow) / sma_slow if sma_slow > 0 else 0.0
        else:
            # Not enough history yet (e.g. right after an episode reset) --
            # neutral trend rather than a guess.
            trend_signal = 0.0

        # Volatility: realized annualized vol of recent daily returns.
        daily_returns = np.log(close_series / close_series.shift(1)).dropna()
        if len(daily_returns) >= 5:
            current_vol = daily_returns.tail(20).std() * np.sqrt(252)
            if np.isnan(current_vol):
                current_vol = 0.15
        else:
            current_vol = 0.15

        scores = self._compute_scores(trend_signal, current_vol)

        formation = self.config.belief_formation
        if formation == BeliefFormationType.SOFTMAX_TEMPERATURE:
            self._current_belief = self._update_softmax_temperature(scores)
        elif formation == BeliefFormationType.HARD_EMA:
            self._current_belief = self._update_hard_ema(scores)
        elif formation == BeliefFormationType.HARD_EMA_HYSTERESIS:
            self._current_belief = self._update_hard_ema_hysteresis(scores)
        else:
            raise ValueError(f"Unknown belief_formation: {formation}")

        return self._current_belief

    def _compute_scores(self, trend_signal: float, current_vol: float) -> np.ndarray:
        """Shared regime scoring (soft assignment) used by all three belief-
        formation candidates: (A) softmax+temperature reads these directly;
        (B)/(C) take argmax(scores) as the hard daily assignment."""
        scores = np.zeros(self.n_regimes)

        if self._vol_thresholds is None:
            # Default thresholds if not fitted
            low_vol, high_vol = 0.10, 0.25
        else:
            low_vol, high_vol = self._vol_thresholds

        # Bull regime: positive trend (fast SMA above slow SMA) + low vol
        trend_bull = max(0, trend_signal * 10)  # Scale factor
        vol_bull = max(0, (low_vol - current_vol) / low_vol) if low_vol > 0 else 0
        scores[0] = trend_bull + vol_bull

        # Bear regime: negative trend (fast SMA below slow SMA) + high vol
        trend_bear = max(0, -trend_signal * 10)
        vol_bear = max(0, (current_vol - high_vol) / high_vol) if high_vol > 0 else 0
        scores[1] = trend_bear + vol_bear

        # Sideways: neither strong trend nor extreme vol
        trend_sideways = 1 - abs(trend_signal * 5)
        vol_sideways = 1 - abs(current_vol - (low_vol + high_vol) / 2) / ((high_vol - low_vol) / 2 + 1e-6)
        scores[2] = max(0, trend_sideways) + max(0, vol_sideways)

        return scores

    def _blend_onehot_ema(self, regime_idx: int) -> np.ndarray:
        """Shared EMA-of-onehot update used by candidates B and C."""
        onehot = np.zeros(self.n_regimes)
        onehot[regime_idx] = 1.0

        if self._current_belief is not None:
            belief = self._ema_alpha * onehot + (1 - self._ema_alpha) * self._current_belief
        else:
            belief = onehot

        return belief / belief.sum()

    def _update_softmax_temperature(self, scores: np.ndarray) -> np.ndarray:
        """(A) Original mechanism, parameterized by temperature T:
        softmax(scores / T), then EMA-smoothed over days. T=1.0 reproduces
        the original always-on behavior exactly."""
        temperature = self.config.softmax_temperature
        new_belief = softmax(scores / temperature)

        if self._current_belief is not None:
            belief = self._ema_alpha * new_belief + (1 - self._ema_alpha) * self._current_belief
        else:
            belief = new_belief

        return belief / belief.sum()

    def _update_hard_ema(self, scores: np.ndarray) -> np.ndarray:
        """(B) Spec-style: hard daily regime assignment z_t = argmax(scores),
        then b_t = (1-alpha)*b_{t-1} + alpha*onehot(z_t)."""
        z_t = int(np.argmax(scores))
        return self._blend_onehot_ema(z_t)

    def _update_hard_ema_hysteresis(self, scores: np.ndarray) -> np.ndarray:
        """(C) Same as (B), but the committed regime only flips once the
        argmax has differed from it for hysteresis_days consecutive days --
        reduces single-day chatter driving the EMA around."""
        raw_regime = int(np.argmax(scores))
        hysteresis_days = self.config.hysteresis_days

        if self._committed_regime is None:
            self._committed_regime = raw_regime
            self._pending_regime = None
            self._pending_count = 0
        elif raw_regime == self._committed_regime:
            self._pending_regime = None
            self._pending_count = 0
        else:
            if raw_regime == self._pending_regime:
                self._pending_count += 1
            else:
                self._pending_regime = raw_regime
                self._pending_count = 1

            if self._pending_count >= hysteresis_days:
                self._committed_regime = raw_regime
                self._pending_regime = None
                self._pending_count = 0

        return self._blend_onehot_ema(self._committed_regime)

    def get_belief(self) -> np.ndarray:
        if self._current_belief is None:
            return np.ones(self.n_regimes) / self.n_regimes
        return self._current_belief.copy()

    def is_resolved(self) -> bool:
        """For belief_formation=HARD_EMA_HYSTERESIS: False while the argmax
        has flipped away from the committed regime but hasn't yet persisted
        hysteresis_days to actually commit (self._pending_regime is not
        None) -- i.e. mid-transition. Always True for the other two
        formation candidates, which have no committed/pending state."""
        if self.config.belief_formation != BeliefFormationType.HARD_EMA_HYSTERESIS:
            return True
        return self._pending_regime is None

    def reset(self) -> None:
        super().reset()
        # Cleared per episode: training uses random episode start points, so
        # a previous episode's close-price history would otherwise leak into
        # a new episode covering a completely different point in time.
        # _market_ticker/_vol_thresholds come from fit() and persist across
        # episodes, same as before.
        self._close_history = []
        self._committed_regime = None
        self._pending_regime = None
        self._pending_count = 0


class HMMRegimeModel(RegimeModel):
    """
    Hidden Markov Model for regime inference.
    Uses Gaussian emissions on return/volatility features.
    """

    def __init__(self, config: RegimeConfig):
        super().__init__(config)
        self._hmm = None
        self._fitted = False
        self._observation_buffer: List[np.ndarray] = []
        self.reset()

    def fit(self, data: pd.DataFrame) -> None:
        """Fit HMM on historical returns and volatility."""
        try:
            from hmmlearn import hmm
        except ImportError:
            raise ImportError("hmmlearn required. Install with: pip install hmmlearn")

        # Get first ticker's data
        # Preserve column order (first occurrence) rather than set()'s
        # hash-dependent order: tickers[0] below is relied on as the primary
        # / benchmark market proxy, so it must match the config-declared order.
        tickers = list(dict.fromkeys(col[0] for col in data.columns))
        market_ticker = tickers[0]
        close = data[(market_ticker, 'Close')]

        # Prepare features for HMM
        returns = np.log(close / close.shift(1)).dropna()
        volatility = returns.rolling(20).std().dropna()

        # Align
        aligned_idx = returns.index.intersection(volatility.index)
        X = np.column_stack([
            returns.loc[aligned_idx].values,
            volatility.loc[aligned_idx].values
        ])

        # Fit HMM
        self._hmm = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=self.config.hmm_covariance_type,
            n_iter=self.config.hmm_n_iter,
            random_state=42
        )

        self._hmm.fit(X)
        self._fitted = True

        print(f"HMMRegimeModel fitted:")
        print(f"  Means: {self._hmm.means_}")
        print(f"  Transition matrix:\n{self._hmm.transmat_}")

    def update(
            self,
            observation: np.ndarray,
            prices: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """Update belief using HMM filtering. `prices` is accepted for
        interface compatibility (see RegimeModel.update) but unused here."""
        if not self._fitted or self._hmm is None:
            # Return uniform if not fitted
            return np.ones(self.n_regimes) / self.n_regimes

        # Use first two features (return, volatility) for HMM
        hmm_obs = observation[:2].reshape(1, -1)

        self._observation_buffer.append(hmm_obs)

        # Keep buffer manageable
        max_buffer = 100
        if len(self._observation_buffer) > max_buffer:
            self._observation_buffer = self._observation_buffer[-max_buffer:]

        # Compute filtered belief using forward algorithm
        obs_sequence = np.vstack(self._observation_buffer)

        try:
            # Get state probabilities
            _, posteriors = self._hmm.score_samples(obs_sequence)
            self._current_belief = posteriors[-1]  # Latest belief
        except Exception:
            # Fallback to uniform on error
            self._current_belief = np.ones(self.n_regimes) / self.n_regimes

        return self._current_belief

    def get_belief(self) -> np.ndarray:
        if self._current_belief is None:
            return np.ones(self.n_regimes) / self.n_regimes
        return self._current_belief.copy()

    def reset(self) -> None:
        super().reset()
        self._observation_buffer = []


class LearnedRegimeModel(RegimeModel):
    """
    Neural network-based regime inference.
    Can be jointly trained with the policy or pretrained.
    """

    def __init__(self, config: RegimeConfig, input_dim: int = 32):
        super().__init__(config)
        self.input_dim = input_dim
        self._model = None
        self._build_model()
        self.reset()

    def _build_model(self):
        """Build neural network for regime classification."""
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError("PyTorch required. Install with: pip install torch")

        class RegimeNetwork(nn.Module):
            def __init__(self, input_dim: int, n_regimes: int, hidden_dim: int = 64):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, n_regimes),
                    nn.Softmax(dim=-1)
                )

            def forward(self, x):
                return self.network(x)

        self._model = RegimeNetwork(self.input_dim, self.n_regimes)

    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit using unsupervised clustering or semi-supervised learning.
        For simplicity, we use K-means on features to get pseudo-labels.
        """
        # This is a placeholder - in practice you'd use proper training
        print("LearnedRegimeModel: Using untrained network (pseudo-labels from clustering)")

    def update(
            self,
            observation: np.ndarray,
            prices: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """Forward pass through network. `prices` is accepted for interface
        compatibility (see RegimeModel.update) but unused here."""
        import torch

        if self._model is None:
            return np.ones(self.n_regimes) / self.n_regimes

        # Ensure correct input dimension
        if len(observation) < self.input_dim:
            observation = np.pad(observation, (0, self.input_dim - len(observation)))
        elif len(observation) > self.input_dim:
            observation = observation[:self.input_dim]

        with torch.no_grad():
            x = torch.FloatTensor(observation).unsqueeze(0)
            belief = self._model(x).numpy()[0]

        # EMA smoothing
        if self._current_belief is not None:
            alpha = self.config.belief_ema_alpha
            self._current_belief = alpha * belief + (1 - alpha) * self._current_belief
        else:
            self._current_belief = belief

        return self._current_belief

    def get_belief(self) -> np.ndarray:
        if self._current_belief is None:
            return np.ones(self.n_regimes) / self.n_regimes
        return self._current_belief.copy()

    def get_parameters(self):
        """Return model parameters for joint training."""
        if self._model is not None:
            return self._model.parameters()
        return []


def create_regime_model(config: RegimeConfig, input_dim: int = 32) -> RegimeModel:
    """Factory function to create regime model based on config."""
    if config.model_type == RegimeModelType.HEURISTIC:
        return HeuristicRegimeModel(config)
    elif config.model_type == RegimeModelType.HMM:
        return HMMRegimeModel(config)
    elif config.model_type == RegimeModelType.LEARNED:
        return LearnedRegimeModel(config, input_dim)
    else:
        raise ValueError(f"Unknown regime model type: {config.model_type}")


class RegimeAnalyzer:
    """
    Utility class for analyzing regime dynamics.
    Useful for understanding model behavior and generating paper figures.
    """

    def __init__(self, regime_model: RegimeModel):
        self.model = regime_model
        self.belief_history: List[np.ndarray] = []
        self.entropy_history: List[float] = []
        self.timestamps: List[int] = []

    def record(self, t: int):
        """Record current belief state."""
        self.belief_history.append(self.model.get_belief().copy())
        self.entropy_history.append(self.model.get_entropy())
        self.timestamps.append(t)

    def get_regime_sequence(self) -> np.ndarray:
        """Return sequence of most likely regimes."""
        beliefs = np.array(self.belief_history)
        return np.argmax(beliefs, axis=1)

    def get_transition_points(self, threshold: float = 0.3) -> List[int]:
        """
        Find time points where regime transitions likely occurred.
        Based on entropy spikes.
        """
        entropy = np.array(self.entropy_history)
        max_entropy = self.model.get_max_entropy()

        # Find points where entropy exceeds threshold
        high_entropy_mask = entropy > threshold * max_entropy

        # Find transitions (changes in dominant regime)
        regimes = self.get_regime_sequence()
        regime_changes = np.where(np.diff(regimes) != 0)[0] + 1

        # Combine both signals
        transition_candidates = set(regime_changes.tolist())
        for i, is_high in enumerate(high_entropy_mask):
            if is_high:
                transition_candidates.add(i)

        return sorted(list(transition_candidates))

    def to_dataframe(self) -> pd.DataFrame:
        """Convert history to DataFrame for analysis."""
        beliefs = np.array(self.belief_history)
        df = pd.DataFrame(
            beliefs,
            columns=[f'regime_{i}' for i in range(beliefs.shape[1])],
            index=self.timestamps
        )
        df['entropy'] = self.entropy_history
        df['dominant_regime'] = np.argmax(beliefs, axis=1)
        return df

    def reset(self):
        """Clear history."""
        self.belief_history = []
        self.entropy_history = []
        self.timestamps = []


if __name__ == "__main__":
    # Test regime models
    from data_loader import SyntheticDataGenerator
    from features import FeatureBuilder, FeatureConfig

    print("Testing regime models...")

    # Generate data
    gen = SyntheticDataGenerator(n_assets=2, n_days=500)
    data, true_regimes = gen.generate_regime_switching_data()

    # Compute features
    feature_config = FeatureConfig()
    builder = FeatureBuilder(feature_config)
    features = builder.compute_features(data)

    # Test heuristic model
    print("\n--- Heuristic Model ---")
    regime_config = RegimeConfig(model_type=RegimeModelType.HEURISTIC)
    model = HeuristicRegimeModel(regime_config)
    model.fit(data)

    analyzer = RegimeAnalyzer(model)
    tickers = list(dict.fromkeys(col[0] for col in data.columns))

    for t in range(100, 200):
        obs = features.iloc[t].values
        current_prices = {ticker: data[(ticker, 'Close')].iloc[t] for ticker in tickers}
        belief = model.update(obs, prices=current_prices)
        analyzer.record(t)

    print(f"Entropy range: [{min(analyzer.entropy_history):.3f}, {max(analyzer.entropy_history):.3f}]")
    print(f"Transitions detected: {len(analyzer.get_transition_points())}")

    # Compare with true regimes
    predicted = analyzer.get_regime_sequence()
    true_segment = true_regimes[100:200]
    accuracy = np.mean(predicted == true_segment)
    print(f"Regime prediction accuracy: {accuracy:.2%}")