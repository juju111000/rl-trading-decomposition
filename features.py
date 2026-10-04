"""
Feature engineering module.
Computes technical indicators and prepares features for the trading environment.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

from config import FeatureConfig

# Rolling windows longer than this use min_periods=ROLLING_MIN_PERIODS
# rather than requiring the full window, so every rolling feature's warmup
# is bounded by the same ~20-real-observation floor as normalization's own
# min_periods=20 -- matching main.FEATURE_WARMUP_BUFFER_DAYS (~40 calendar
# days). Without this, e.g. the 200-day MA would need 200 real trading days
# of warmup (~280 calendar days), far beyond that buffer. Early-window
# values computed from fewer than the full window's observations are
# noisier / less representative of the intended long-term lookback --
# a deliberate tradeoff for bounded, buffer-covered warmup instead of
# dropping folds whose real data can't supply a full 200-day runway
# (see DATA_NOTES.md discussion of the walk-forward window constraints).
ROLLING_MIN_PERIODS = 20


class FeatureBuilder:
    """
    Builds feature vectors from OHLCV data.
    Handles per-asset features and cross-asset features.
    """

    def __init__(self, config: FeatureConfig):
        self.config = config
        self.feature_names: List[str] = []
        self._normalization_params: Dict[str, Tuple[float, float]] = {}

    def compute_features(
            self,
            data: pd.DataFrame,
            fit_normalization: bool = True
    ) -> pd.DataFrame:
        """
        Compute all features from aligned OHLCV data.

        Args:
            data: Multi-indexed DataFrame with (ticker, OHLCV) columns.
            fit_normalization: If True, compute normalization params from this data.

        Returns:
            DataFrame with computed features.
        """
        # Preserve column order (first occurrence) rather than set()'s
        # hash-dependent order, so results are deterministic run-to-run.
        tickers = list(dict.fromkeys(col[0] for col in data.columns))

        all_features = []
        self.feature_names = []

        # Per-asset features
        for ticker in tickers:
            ticker_data = data[ticker]
            ticker_features = self._compute_asset_features(ticker_data, ticker)
            all_features.append(ticker_features)

        # Cross-asset features
        if self.config.use_cross_asset_features and len(tickers) > 1:
            cross_features = self._compute_cross_asset_features(data, tickers)
            all_features.append(cross_features)

        # Combine all features
        features = pd.concat(all_features, axis=1)

        # Handle NaN from rolling windows. ffill only -- never bfill: that
        # would backfill leading warmup NaN from LATER rows, which is
        # forward-looking leakage (audit #14). The caller (see
        # main.load_and_prepare_data) is responsible for requesting enough
        # real prior history (a warmup buffer before the nominal start) and
        # trimming whatever leading NaN survives that buffer.
        features = features.ffill()

        # Normalize
        if fit_normalization:
            features = self._fit_normalize(features)
        else:
            features = self._apply_normalize(features)

        return features

    def _compute_asset_features(
            self,
            ohlcv: pd.DataFrame,
            ticker: str
    ) -> pd.DataFrame:
        """Compute features for a single asset."""
        features = {}
        close = ohlcv['Close']
        high = ohlcv['High']
        low = ohlcv['Low']
        volume = ohlcv['Volume'] if 'Volume' in ohlcv.columns else pd.Series(0, index=close.index)

        # Log returns at different horizons
        for w in self.config.return_windows:
            feat_name = f"{ticker}_ret_{w}d"
            features[feat_name] = np.log(close / close.shift(w))
            self.feature_names.append(feat_name)

        # Rolling volatility
        for w in self.config.volatility_windows:
            feat_name = f"{ticker}_vol_{w}d"
            returns = np.log(close / close.shift(1))
            min_periods = min(w, ROLLING_MIN_PERIODS)
            features[feat_name] = returns.rolling(w, min_periods=min_periods).std() * np.sqrt(252)  # Annualized
            self.feature_names.append(feat_name)

        # Moving averages (as ratio to current price)
        for w in self.config.ma_windows:
            feat_name = f"{ticker}_ma_ratio_{w}d"
            min_periods = min(w, ROLLING_MIN_PERIODS)
            ma = close.rolling(w, min_periods=min_periods).mean()
            features[feat_name] = close / ma - 1  # Deviation from MA
            self.feature_names.append(feat_name)

        # Rate of change (momentum)
        for w in self.config.roc_windows:
            feat_name = f"{ticker}_roc_{w}d"
            features[feat_name] = (close - close.shift(w)) / close.shift(w)
            self.feature_names.append(feat_name)

        # ATR (Average True Range)
        if self.config.use_atr:
            feat_name = f"{ticker}_atr"
            tr = pd.concat([
                high - low,
                abs(high - close.shift(1)),
                abs(low - close.shift(1))
            ], axis=1).max(axis=1)
            atr = tr.rolling(self.config.atr_window).mean()
            features[feat_name] = atr / close  # Normalized by price
            self.feature_names.append(feat_name)

        # RSI (Relative Strength Index)
        feat_name = f"{ticker}_rsi"
        features[feat_name] = self._compute_rsi(close, 14)
        self.feature_names.append(feat_name)

        # MACD signal
        feat_name = f"{ticker}_macd_signal"
        features[feat_name] = self._compute_macd_signal(close)
        self.feature_names.append(feat_name)

        # Volume features (if available)
        if volume.sum() > 0:
            feat_name = f"{ticker}_vol_ratio"
            vol_ma = volume.rolling(20).mean()

            # If a ticker's 20-day trailing average volume is ever exactly
            # zero (e.g. a multi-year real-zero-volume gap in the source --
            # this hit ^STOXX50E badly enough to break fold 1 before it was
            # dropped from the universe, see DATA_NOTES.md's git history),
            # this produces a genuine NaN block. That is intentional: it is
            # not a rolling-window warmup gap, so it must NOT be silently
            # bfilled or neutral-filled here -- let main.load_and_prepare_data's
            # leading-NaN trim (or its loud non-leading-NaN error) handle it,
            # the same as any other data-quality problem.
            features[feat_name] = volume / vol_ma.replace(0, np.nan)
            self.feature_names.append(feat_name)

        # High-Low range (volatility proxy)
        feat_name = f"{ticker}_hl_range"
        features[feat_name] = (high - low) / close
        self.feature_names.append(feat_name)

        return pd.DataFrame(features)

    def _compute_cross_asset_features(
            self,
            data: pd.DataFrame,
            tickers: List[str]
    ) -> pd.DataFrame:
        """Compute cross-asset correlation and relative features."""
        features = {}

        # Get returns for all assets
        returns = {}
        for ticker in tickers:
            returns[ticker] = np.log(data[(ticker, 'Close')] / data[(ticker, 'Close')].shift(1))
        returns_df = pd.DataFrame(returns)

        # Rolling correlation between pairs
        for i, t1 in enumerate(tickers):
            for t2 in tickers[i + 1:]:
                feat_name = f"corr_{t1}_{t2}"
                corr_min_periods = min(self.config.correlation_window, ROLLING_MIN_PERIODS)
                corr = returns_df[t1].rolling(
                    self.config.correlation_window, min_periods=corr_min_periods
                ).corr(returns_df[t2])
                features[feat_name] = corr
                self.feature_names.append(feat_name)

        # Market (first ticker) vs others relative strength
        market_ticker = tickers[0]
        market_return = returns_df[market_ticker]

        for ticker in tickers[1:]:
            feat_name = f"rel_strength_{ticker}_vs_{market_ticker}"
            rel_strength = (returns_df[ticker].rolling(20).mean() -
                            market_return.rolling(20).mean())
            features[feat_name] = rel_strength
            self.feature_names.append(feat_name)

        # Average correlation (market stress indicator)
        if len(tickers) > 2:
            corr_cols = [c for c in features.keys() if c.startswith('corr_')]
            if corr_cols:
                features['avg_correlation'] = pd.DataFrame(
                    {c: features[c] for c in corr_cols}
                ).mean(axis=1)
                self.feature_names.append('avg_correlation')

        return pd.DataFrame(features)

    def _compute_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Compute RSI indicator."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return (rsi - 50) / 50  # Normalize to [-1, 1]

    def _compute_macd_signal(
            self,
            prices: pd.Series,
            fast: int = 12,
            slow: int = 26,
            signal: int = 9
    ) -> pd.Series:
        """Compute MACD signal line crossover."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        # Return normalized MACD histogram
        hist = macd - signal_line
        return hist / prices * 100  # As percentage of price

    def _fit_normalize(self, features: pd.DataFrame) -> pd.DataFrame:
        """Fit normalization parameters and apply."""
        self._normalization_params = {}

        normalized = features.copy()
        for col in features.columns:
            # Use rolling normalization to avoid lookahead
            rolling_mean = features[col].rolling(
                self.config.normalization_window, min_periods=20
            ).mean()
            rolling_std = features[col].rolling(
                self.config.normalization_window, min_periods=20
            ).std()

            # For initial period, use expanding window
            expanding_mean = features[col].expanding(min_periods=20).mean()
            expanding_std = features[col].expanding(min_periods=20).std()

            mean = rolling_mean.fillna(expanding_mean)
            std = rolling_std.fillna(expanding_std).replace(0, 1)

            zscore = (features[col] - mean) / std
            normalized[col] = zscore.clip(-self.config.clip_zscore, self.config.clip_zscore)

            # Store final params for test set application
            self._normalization_params[col] = (
                features[col].mean(),
                features[col].std()
            )

        # Rolling/expanding mean+std both need >= min_periods=20 observations,
        # so the z-score is NaN for the first 19 rows of the WHOLE series.
        # Never backfilled (audit #14 -- see the ffill-only note above): the
        # caller trims any leading NaN that survives its warmup buffer
        # instead (see main.load_and_prepare_data).
        return normalized

    def _apply_normalize(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply stored normalization parameters."""
        normalized = features.copy()
        for col in features.columns:
            if col in self._normalization_params:
                mean, std = self._normalization_params[col]
                std = std if std > 0 else 1
                zscore = (features[col] - mean) / std
                normalized[col] = zscore.clip(-self.config.clip_zscore, self.config.clip_zscore)
        return normalized

    def get_feature_dim(self) -> int:
        """Return total feature dimension."""
        return len(self.feature_names)


class FeatureExtractor(ABC):
    """Abstract base class for feature extractors."""

    @abstractmethod
    def extract(self, data: pd.DataFrame, t: int) -> np.ndarray:
        """Extract features at time t."""
        pass

    @abstractmethod
    def get_dim(self) -> int:
        """Return feature dimension."""
        pass


class PrecomputedFeatureExtractor(FeatureExtractor):
    """
    Feature extractor that uses precomputed features.
    Efficient for training where features are computed once.
    """

    def __init__(self, features: pd.DataFrame):
        self.features = features.values
        self.feature_dim = features.shape[1]
        self.index = features.index

    def extract(self, data: pd.DataFrame, t: int) -> np.ndarray:
        """Get precomputed features at time t."""
        if t < 0 or t >= len(self.features):
            raise IndexError(f"Time index {t} out of range [0, {len(self.features)})")
        return self.features[t].astype(np.float32)

    def get_dim(self) -> int:
        return self.feature_dim


class OnlineFeatureExtractor(FeatureExtractor):
    """
    Feature extractor that computes features on-the-fly.
    Useful for live trading or when memory is constrained.
    """

    def __init__(self, config: FeatureConfig, lookback: int = 300):
        self.config = config
        self.lookback = lookback
        self.builder = FeatureBuilder(config)
        self._feature_dim: Optional[int] = None

    def extract(self, data: pd.DataFrame, t: int) -> np.ndarray:
        """Compute features at time t using lookback window."""
        start = max(0, t - self.lookback + 1)
        window_data = data.iloc[start:t + 1]

        features = self.builder.compute_features(window_data, fit_normalization=True)

        if self._feature_dim is None:
            self._feature_dim = features.shape[1]

        return features.iloc[-1].values.astype(np.float32)

    def get_dim(self) -> int:
        if self._feature_dim is None:
            raise ValueError("Feature dimension unknown. Call extract() first.")
        return self._feature_dim


if __name__ == "__main__":
    # Test with synthetic data
    from data_loader import SyntheticDataGenerator

    print("Testing feature builder...")
    gen = SyntheticDataGenerator(n_assets=3, n_days=500)
    data, _ = gen.generate_regime_switching_data()

    config = FeatureConfig()
    builder = FeatureBuilder(config)

    features = builder.compute_features(data)
    print(f"Feature shape: {features.shape}")
    print(f"Feature names ({len(builder.feature_names)}):")
    for name in builder.feature_names[:10]:
        print(f"  {name}")
    print("  ...")

    print(f"\nFeature statistics:")
    print(features.describe())