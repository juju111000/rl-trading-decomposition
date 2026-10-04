"""
Data loading and preprocessing module.
Handles downloading OHLCV data, aligning calendars, and preparing datasets.
"""

import os
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import warnings

try:
    import yfinance as yf
except ImportError:
    yf = None
    warnings.warn("yfinance not installed. Install with: pip install yfinance")

from config import DataConfig


class DataLoader:
    """
    Loads and preprocesses financial data for multiple assets.
    Handles calendar alignment, missing data, and train/val/test splits.
    """

    def __init__(self, config: DataConfig):
        self.config = config
        self.raw_data: Dict[str, pd.DataFrame] = {}
        self.aligned_data: Optional[pd.DataFrame] = None
        self.returns: Optional[pd.DataFrame] = None
        self._splits: Optional[Dict[str, pd.DataFrame]] = None

    def download_data(self, force_download: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Download OHLCV data for all tickers.

        Args:
            force_download: If True, bypass cache and download fresh data.

        Returns:
            Dictionary mapping ticker to DataFrame with OHLCV data.
        """
        if yf is None:
            raise ImportError("yfinance required. Install with: pip install yfinance")

        os.makedirs(self.config.cache_dir, exist_ok=True)
        # Cache key includes the ticker set, not just the date range -- two
        # configs requesting the same dates with DIFFERENT tickers (e.g.
        # swapping ^STOXX50E for ^GDAXI) must not silently load each
        # other's cached raw data.
        ticker_slug = "-".join(sorted(t.replace("^", "").replace("=", "") for t in self.config.tickers))
        cache_path = os.path.join(
            self.config.cache_dir,
            f"raw_data_{self.config.start_date}_{self.config.end_date}_{ticker_slug}.pkl"
        )

        # Try loading from cache
        if not force_download and os.path.exists(cache_path):
            print(f"Loading cached data from {cache_path}")
            with open(cache_path, 'rb') as f:
                self.raw_data = pickle.load(f)
            return self.raw_data

        print(f"Downloading data for {self.config.tickers}...")

        for ticker in self.config.tickers:
            print(f"  Downloading {ticker}...")
            try:
                df = yf.download(
                    ticker,
                    start=self.config.start_date,
                    end=self.config.end_date,
                    progress=False
                )

                if df.empty:
                    warnings.warn(f"No data received for {ticker}")
                    continue

                # Handle multi-level columns from yfinance
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                # Ensure required columns exist
                required_cols = ['Open', 'High', 'Low', 'Close']
                missing = [c for c in required_cols if c not in df.columns]
                if missing:
                    warnings.warn(f"Missing columns for {ticker}: {missing}")
                    continue

                # Handle Volume (some indices don't have it)
                if 'Volume' not in df.columns:
                    df['Volume'] = 0

                # Use Adj Close if available and configured
                if self.config.use_adjusted_close and 'Adj Close' in df.columns:
                    df['Close'] = df['Adj Close']

                self.raw_data[ticker] = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                print(f"    Downloaded {len(df)} rows for {ticker}")

            except Exception as e:
                warnings.warn(f"Failed to download {ticker}: {e}")
                continue

        # Cache the data
        with open(cache_path, 'wb') as f:
            pickle.dump(self.raw_data, f)
        print(f"Cached data to {cache_path}")

        return self.raw_data

    def align_data(self) -> pd.DataFrame:
        """
        Align all assets to a common trading calendar.

        Returns:
            Multi-indexed DataFrame with aligned OHLCV data for all assets.
        """
        if not self.raw_data:
            raise ValueError("No data loaded. Call download_data() first.")

        # Get intersection of all dates (common trading days)
        common_dates = None
        for ticker, df in self.raw_data.items():
            dates = set(df.index)
            if common_dates is None:
                common_dates = dates
            else:
                if self.config.handle_missing == "drop":
                    common_dates = common_dates.intersection(dates)
                else:
                    common_dates = common_dates.union(dates)

        common_dates = sorted(list(common_dates))
        print(f"Common trading days: {len(common_dates)}")

        # Create aligned multi-index DataFrame
        aligned_dfs = []
        for ticker in self.raw_data.keys():
            df = self.raw_data[ticker].reindex(common_dates)

            # Treat non-positive OHLC as bad data, the same way as a missing
            # observation -- e.g. CL=F (WTI crude) printed a negative Close
            # on 2020-04-20 and negative Open/Low on 2020-04-21 during the
            # historic negative-oil-price event. The rest of the pipeline
            # (log returns, log-NAV, position-based portfolio accounting)
            # assumes strictly positive prices; a negative/zero price here
            # would silently produce NaN deep inside training/evaluation
            # instead of failing loudly, so mask it before that can happen.
            price_cols = [c for c in ('Open', 'High', 'Low', 'Close') if c in df.columns]
            bad_price_mask = (df[price_cols] <= 0).any(axis=1)
            n_bad = int(bad_price_mask.sum())
            if n_bad > 0:
                bad_dates = ", ".join(d.strftime("%Y-%m-%d") for d in df.index[bad_price_mask])
                print(
                    f"WARNING: masked {n_bad} non-positive OHLC row(s) for {ticker}: "
                    f"{bad_dates} -- see DATA_NOTES.md for the data decision this reflects."
                )
            df[price_cols] = df[price_cols].where(df[price_cols] > 0)

            # Handle missing data
            if self.config.handle_missing == "ffill":
                df = df.ffill()
            elif self.config.handle_missing == "interpolate":
                df = df.interpolate(method='linear')

            # Drop any remaining NaN rows (e.g., at the start)
            df = df.dropna()

            # Add ticker level to columns
            df.columns = pd.MultiIndex.from_product([[ticker], df.columns])
            aligned_dfs.append(df)

        # Concatenate all tickers
        self.aligned_data = pd.concat(aligned_dfs, axis=1)

        # Keep only rows where all assets have data
        self.aligned_data = self.aligned_data.dropna()
        print(f"Aligned data shape: {self.aligned_data.shape}")

        return self.aligned_data

    def compute_returns(self) -> pd.DataFrame:
        """
        Compute log returns from Close prices.

        Returns:
            DataFrame with log returns for each asset.
        """
        if self.aligned_data is None:
            raise ValueError("Data not aligned. Call align_data() first.")

        returns_dict = {}
        for ticker in self.raw_data.keys():
            close = self.aligned_data[(ticker, 'Close')]
            returns_dict[ticker] = np.log(close / close.shift(1))

        self.returns = pd.DataFrame(returns_dict).dropna()
        return self.returns

    def get_train_val_test_split(self) -> Dict[str, pd.DataFrame]:
        """
        Split data into train/validation/test sets.

        Returns:
            Dictionary with 'train', 'val', 'test' DataFrames.
        """
        if self.aligned_data is None:
            raise ValueError("Data not aligned. Call align_data() first.")

        n = len(self.aligned_data)
        train_end = int(n * self.config.train_ratio)
        val_end = int(n * (self.config.train_ratio + self.config.val_ratio))

        self._splits = {
            'train': self.aligned_data.iloc[:train_end].copy(),
            'val': self.aligned_data.iloc[train_end:val_end].copy(),
            'test': self.aligned_data.iloc[val_end:].copy()
        }

        print(f"Data split:")
        print(f"  Train: {len(self._splits['train'])} samples "
              f"({self._splits['train'].index[0]} to {self._splits['train'].index[-1]})")
        print(f"  Val:   {len(self._splits['val'])} samples "
              f"({self._splits['val'].index[0]} to {self._splits['val'].index[-1]})")
        print(f"  Test:  {len(self._splits['test'])} samples "
              f"({self._splits['test'].index[0]} to {self._splits['test'].index[-1]})")

        return self._splits

    def get_walk_forward_splits(
            self,
            train_window: int = 504,  # ~2 years
            val_window: int = 63,  # ~3 months
            test_window: int = 63,  # ~3 months
            step: int = 63  # ~3 months
    ) -> List[Dict[str, pd.DataFrame]]:
        """
        Generate walk-forward validation splits.

        Args:
            train_window: Number of days for training.
            val_window: Number of days for validation.
            test_window: Number of days for testing.
            step: Step size between folds.

        Returns:
            List of dictionaries, each with 'train', 'val', 'test' DataFrames.
        """
        if self.aligned_data is None:
            raise ValueError("Data not aligned. Call align_data() first.")

        splits = []
        total_window = train_window + val_window + test_window
        n = len(self.aligned_data)

        start = 0
        while start + total_window <= n:
            train_end = start + train_window
            val_end = train_end + val_window
            test_end = val_end + test_window

            splits.append({
                'train': self.aligned_data.iloc[start:train_end].copy(),
                'val': self.aligned_data.iloc[train_end:val_end].copy(),
                'test': self.aligned_data.iloc[val_end:test_end].copy(),
                'fold_start': self.aligned_data.index[start],
                'fold_end': self.aligned_data.index[min(test_end - 1, n - 1)]
            })

            start += step

        print(f"Generated {len(splits)} walk-forward folds")
        return splits

    def prepare_dataset(
            self,
            force_download: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Full pipeline: download, align, compute returns, and split data.

        Returns:
            Tuple of (aligned_data, returns, splits_dict)
        """
        self.download_data(force_download=force_download)
        self.align_data()
        self.compute_returns()
        splits = self.get_train_val_test_split()

        return self.aligned_data, self.returns, splits

    def get_price_data(self, split: str = 'train') -> pd.DataFrame:
        """Get Close prices for a specific split."""
        if self._splits is None:
            raise ValueError("No splits created. Call get_train_val_test_split() first.")

        data = self._splits[split]
        prices = {}
        for ticker in self.raw_data.keys():
            prices[ticker] = data[(ticker, 'Close')]
        return pd.DataFrame(prices)

    def get_ohlcv(self, ticker: str, split: str = 'train') -> pd.DataFrame:
        """Get OHLCV data for a specific ticker and split."""
        if self._splits is None:
            raise ValueError("No splits created. Call get_train_val_test_split() first.")

        return self._splits[split][ticker]


class SyntheticDataGenerator:
    """
    Generate synthetic market data for testing and debugging.
    Useful when real data download fails or for controlled experiments.
    """

    def __init__(
            self,
            n_assets: int = 4,
            n_days: int = 5000,
            seed: int = 42,
            start_date: str = '2010-01-01'
    ):
        self.n_assets = n_assets
        self.n_days = n_days
        self.seed = seed
        self.start_date = start_date
        self.rng = np.random.RandomState(seed)

    def generate_regime_switching_data(
            self,
            n_regimes: int = 3,
            regime_persistence: float = 0.98
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Generate synthetic price data with regime-switching dynamics.

        Args:
            n_regimes: Number of market regimes.
            regime_persistence: Probability of staying in current regime.

        Returns:
            Tuple of (OHLCV DataFrame, regime labels array)
        """
        # Regime parameters (mean return, volatility)
        regime_params = [
            (0.0003, 0.01),  # Bull: positive drift, low vol
            (-0.0002, 0.025),  # Bear: negative drift, high vol
            (0.0, 0.015)  # Sideways: no drift, medium vol
        ]

        # Generate regime sequence (Markov chain)
        regimes = np.zeros(self.n_days, dtype=int)
        regimes[0] = self.rng.randint(n_regimes)

        for t in range(1, self.n_days):
            if self.rng.random() < regime_persistence:
                regimes[t] = regimes[t - 1]
            else:
                # Transition to a different regime
                other_regimes = [r for r in range(n_regimes) if r != regimes[t - 1]]
                regimes[t] = self.rng.choice(other_regimes)

        # Generate returns for each asset with some correlation
        asset_names = [f"ASSET_{i}" for i in range(self.n_assets)]

        # Base correlation structure
        corr_matrix = np.eye(self.n_assets) * 0.5 + 0.5
        L = np.linalg.cholesky(corr_matrix)

        all_data = {}
        dates = pd.date_range(start=self.start_date, periods=self.n_days, freq='B')

        for i, asset in enumerate(asset_names):
            # Generate correlated innovations
            z = self.rng.randn(self.n_days)

            # Apply regime-specific parameters
            returns = np.zeros(self.n_days)
            for t in range(self.n_days):
                mu, sigma = regime_params[regimes[t]]
                # Add asset-specific variation
                mu_asset = mu * (0.8 + 0.4 * self.rng.random())
                sigma_asset = sigma * (0.8 + 0.4 * self.rng.random())
                returns[t] = mu_asset + sigma_asset * z[t]

            # Generate prices from returns
            log_prices = np.cumsum(returns) + np.log(100)  # Start at 100
            close = np.exp(log_prices)

            # Generate OHLC from Close
            daily_vol = np.abs(returns) + 0.005
            high = close * (1 + daily_vol * self.rng.uniform(0.2, 1.0, self.n_days))
            low = close * (1 - daily_vol * self.rng.uniform(0.2, 1.0, self.n_days))
            open_price = low + (high - low) * self.rng.uniform(0.3, 0.7, self.n_days)

            # Volume (higher in volatile periods)
            base_volume = 1e6
            volume = base_volume * (1 + 2 * daily_vol / daily_vol.mean()) * self.rng.uniform(0.5, 1.5, self.n_days)

            all_data[asset] = pd.DataFrame({
                'Open': open_price,
                'High': high,
                'Low': low,
                'Close': close,
                'Volume': volume.astype(int)
            }, index=dates)

        # Create multi-index DataFrame
        dfs = []
        for asset, df in all_data.items():
            df.columns = pd.MultiIndex.from_product([[asset], df.columns])
            dfs.append(df)

        combined = pd.concat(dfs, axis=1)

        return combined, regimes

    def generate_simple_trending_data(self) -> pd.DataFrame:
        """Generate simple trending data for basic testing."""
        dates = pd.date_range(start=self.start_date, periods=self.n_days, freq='B')

        all_data = {}
        for i in range(self.n_assets):
            asset = f"ASSET_{i}"

            # Trending with mean reversion
            trend = 0.0001 * (1 + 0.5 * np.sin(np.arange(self.n_days) * 2 * np.pi / 252))
            noise = self.rng.randn(self.n_days) * 0.015
            returns = trend + noise

            log_prices = np.cumsum(returns) + np.log(100)
            close = np.exp(log_prices)

            all_data[asset] = pd.DataFrame({
                'Open': close * (1 + self.rng.randn(self.n_days) * 0.002),
                'High': close * (1 + np.abs(self.rng.randn(self.n_days)) * 0.01),
                'Low': close * (1 - np.abs(self.rng.randn(self.n_days)) * 0.01),
                'Close': close,
                'Volume': (1e6 * self.rng.uniform(0.5, 1.5, self.n_days)).astype(int)
            }, index=dates)

        dfs = []
        for asset, df in all_data.items():
            df.columns = pd.MultiIndex.from_product([[asset], df.columns])
            dfs.append(df)

        return pd.concat(dfs, axis=1)


if __name__ == "__main__":
    # Test with synthetic data
    print("Testing synthetic data generation...")
    gen = SyntheticDataGenerator(n_assets=4, n_days=2000)
    synth_data, regimes = gen.generate_regime_switching_data()
    print(f"Generated synthetic data: {synth_data.shape}")
    print(f"Regime distribution: {np.bincount(regimes)}")

    # Test DataLoader with real data (if yfinance available)
    if yf is not None:
        print("\nTesting real data loading...")
        config = DataConfig(
            tickers=["^GSPC", "^GDAXI"],
            start_date="2020-01-01",
            end_date="2023-12-31"
        )
        loader = DataLoader(config)
        try:
            aligned, returns, splits = loader.prepare_dataset()
            print(f"Loaded real data: {aligned.shape}")
        except Exception as e:
            print(f"Real data loading failed: {e}")