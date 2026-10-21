"""
Portfolio engine module.
Handles NAV computation, position tracking, risk metrics, and transaction costs.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class Position:
    """Represents a position in a single asset."""
    ticker: str
    shares: float = 0.0
    avg_cost: float = 0.0

    @property
    def is_long(self) -> bool:
        return self.shares > 0

    @property
    def is_short(self) -> bool:
        return self.shares < 0

    def market_value(self, price: float) -> float:
        return self.shares * price

    def unrealized_pnl(self, price: float) -> float:
        return self.shares * (price - self.avg_cost)


@dataclass
class PortfolioState:
    """
    Complete portfolio state at a point in time.
    Used as part of the observation in the RL environment.
    """
    nav: float
    cash: float
    weights: np.ndarray  # Current portfolio weights
    positions: Dict[str, Position]
    peak_nav: float  # For drawdown calculation
    current_drawdown: float
    gross_leverage: float
    net_exposure: float

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for observation."""
        return np.concatenate([
            self.weights,
            [self.cash / self.nav if self.nav > 0 else 1.0],  # Cash ratio
            [self.current_drawdown],
            [self.gross_leverage],
            [self.net_exposure]
        ]).astype(np.float32)

    @staticmethod
    def get_dim(n_assets: int) -> int:
        """Return dimension of portfolio state array."""
        return n_assets + 4  # weights + cash_ratio + dd + leverage + net_exposure


class PortfolioEngine:
    """
    Core portfolio management engine.
    Tracks positions, computes NAV, handles trades, and calculates risk metrics.
    """

    def __init__(
            self,
            tickers: List[str],
            initial_capital: float = 1_000_000.0,
            transaction_fee_rate: float = 0.001,
            slippage_rate: float = 0.0005,
            use_volatility_slippage: bool = True,
            slippage_vol_multiplier: float = 0.1
    ):
        self.tickers = tickers
        self.n_assets = len(tickers)
        self.initial_capital = initial_capital
        self.transaction_fee_rate = transaction_fee_rate
        self.slippage_rate = slippage_rate
        self.use_volatility_slippage = use_volatility_slippage
        self.slippage_vol_multiplier = slippage_vol_multiplier

        # State
        self._cash: float = initial_capital
        self._positions: Dict[str, Position] = {t: Position(ticker=t) for t in tickers}
        self._nav_history: List[float] = [initial_capital]
        self._peak_nav: float = initial_capital
        self._weight_history: List[np.ndarray] = []

        # Rolling volatility for slippage calculation
        self._return_buffer: Dict[str, List[float]] = {t: [] for t in tickers}
        self._vol_estimates: Dict[str, float] = {t: 0.02 for t in tickers}  # Default 2% DAILY vol

    def reset(self) -> PortfolioState:
        """Reset portfolio to initial state."""
        self._cash = self.initial_capital
        self._positions = {t: Position(ticker=t) for t in self.tickers}
        self._nav_history = [self.initial_capital]
        self._peak_nav = self.initial_capital
        self._weight_history = []
        self._return_buffer = {t: [] for t in self.tickers}
        self._vol_estimates = {t: 0.02 for t in self.tickers}

        return self.get_state(self._get_dummy_prices())

    def _get_dummy_prices(self) -> Dict[str, float]:
        """Return dummy prices for initialization."""
        return {t: 100.0 for t in self.tickers}

    def get_state(self, prices: Dict[str, float]) -> PortfolioState:
        """Get current portfolio state given market prices."""
        nav = self.compute_nav(prices)
        weights = self.compute_weights(prices)

        return PortfolioState(
            nav=nav,
            cash=self._cash,
            weights=weights,
            positions=self._positions.copy(),
            peak_nav=self._peak_nav,
            current_drawdown=self.compute_drawdown(nav),
            gross_leverage=self.compute_gross_leverage(prices),
            net_exposure=self.compute_net_exposure(prices)
        )

    def compute_nav(self, prices: Dict[str, float]) -> float:
        """Compute Net Asset Value."""
        market_value = sum(
            self._positions[t].market_value(prices[t])
            for t in self.tickers
        )
        return self._cash + market_value

    def compute_weights(self, prices: Dict[str, float]) -> np.ndarray:
        """Compute current portfolio weights."""
        nav = self.compute_nav(prices)
        if nav <= 0:
            return np.zeros(self.n_assets)

        weights = np.array([
            self._positions[t].market_value(prices[t]) / nav
            for t in self.tickers
        ])
        return weights

    def compute_drawdown(self, nav: Optional[float] = None) -> float:
        """Compute current drawdown from peak."""
        if nav is None:
            nav = self._nav_history[-1] if self._nav_history else self.initial_capital

        if self._peak_nav <= 0:
            return 0.0

        dd = 1 - nav / self._peak_nav
        return max(0.0, dd)

    def compute_gross_leverage(self, prices: Dict[str, float]) -> float:
        """Compute gross leverage (sum of absolute weights)."""
        weights = self.compute_weights(prices)
        return np.sum(np.abs(weights))

    def compute_net_exposure(self, prices: Dict[str, float]) -> float:
        """Compute net exposure (sum of weights, can be negative)."""
        weights = self.compute_weights(prices)
        return np.sum(weights)

    def compute_turnover(
            self,
            target_weights: np.ndarray,
            prices: Dict[str, float]
    ) -> float:
        """Compute turnover required to reach target weights."""
        current_weights = self.compute_weights(prices)
        return np.sum(np.abs(target_weights - current_weights))

    def execute_trades(
            self,
            target_weights: np.ndarray,
            prices: Dict[str, float],
            volatilities: Optional[Dict[str, float]] = None
    ) -> Tuple[float, Dict[str, any]]:
        """
        Execute trades to reach target portfolio weights.

        Mutates positions and cash (fees/slippage are deducted from cash here).
        Does NOT touch nav_history or peak_nav -- call mark_to_market() once
        afterwards to record the resulting NAV for this step.

        Args:
            target_weights: Target portfolio weights for each asset.
            prices: Current market prices.
            volatilities: Optional volatility estimates for slippage.

        Returns:
            Tuple of (transaction_cost, trade_info_dict)
        """
        # Ensure target_weights is an array (handle scalar case for single asset)
        target_weights = np.atleast_1d(np.asarray(target_weights)).flatten()

        # Get current state
        old_nav = self.compute_nav(prices)
        current_weights = self.compute_weights(prices)

        # Calculate required trades
        trade_info = {'trades': [], 'slippage': 0.0, 'fees': 0.0}
        total_slippage = 0.0
        total_fees = 0.0

        for i, ticker in enumerate(self.tickers):
            current_weight = current_weights[i]
            target_weight = target_weights[i]
            weight_delta = target_weight - current_weight

            if abs(weight_delta) < 1e-6:
                continue

            # Calculate dollar amount to trade
            trade_dollars = weight_delta * old_nav
            price = prices[ticker]
            shares_to_trade = trade_dollars / price

            # Calculate transaction costs
            fee = abs(trade_dollars) * self.transaction_fee_rate

            # Calculate slippage. Audit defect #8: the volatility term uses
            # DAILY-scale volatility (the 0.02 default is a 2% daily vol), and is
            # applied only when use_volatility_slippage is set. Previously the
            # running estimate was annualized (x sqrt(252), ~16x too large) and
            # the term was charged even with use_volatility_slippage=False.
            base_slippage = abs(trade_dollars) * self.slippage_rate
            if self.use_volatility_slippage:
                if volatilities and ticker in volatilities:
                    vol = volatilities[ticker]  # caller-supplied: must be daily-scale
                else:
                    vol = self._vol_estimates.get(ticker, 0.02)
                vol_slippage = abs(trade_dollars) * vol * self.slippage_vol_multiplier
            else:
                vol_slippage = 0.0
            slippage = base_slippage + vol_slippage

            # Execute trade
            old_shares = self._positions[ticker].shares
            self._positions[ticker].shares += shares_to_trade

            # Update average cost for new positions
            if shares_to_trade > 0:  # Buying
                if old_shares >= 0:  # Adding to long or opening long
                    old_cost_basis = old_shares * self._positions[ticker].avg_cost
                    new_cost_basis = shares_to_trade * price
                    total_shares = old_shares + shares_to_trade
                    if total_shares > 0:
                        self._positions[ticker].avg_cost = (old_cost_basis + new_cost_basis) / total_shares
                else:  # Covering short
                    if self._positions[ticker].shares >= 0:  # Fully covered, now long
                        self._positions[ticker].avg_cost = price
            else:  # Selling
                if old_shares <= 0:  # Adding to short or opening short
                    old_cost_basis = abs(old_shares) * self._positions[ticker].avg_cost
                    new_cost_basis = abs(shares_to_trade) * price
                    total_shares = abs(old_shares) + abs(shares_to_trade)
                    if total_shares > 0:
                        self._positions[ticker].avg_cost = (old_cost_basis + new_cost_basis) / total_shares

            # Update cash
            self._cash -= trade_dollars  # Negative for buys, positive for sells
            self._cash -= fee + slippage

            total_fees += fee
            total_slippage += slippage

            trade_info['trades'].append({
                'ticker': ticker,
                'shares': shares_to_trade,
                'price': price,
                'fee': fee,
                'slippage': slippage
            })

        trade_info['fees'] = total_fees
        trade_info['slippage'] = total_slippage
        self._weight_history.append(target_weights.copy())

        transaction_cost = total_fees + total_slippage

        return transaction_cost, trade_info

    def mark_to_market(
            self,
            prices: Dict[str, float],
            prev_prices: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Record NAV at `prices` as the single history entry for this step.

        Call exactly once per environment step, after execute_trades(), so
        that nav_history gains exactly one entry per step -- reflecting both
        this step's transaction costs and any price movement together. Updates
        peak NAV and, if prev_prices is given, rolling per-asset volatility
        estimates.

        Args:
            prices: Prices to mark the portfolio at (after this step's trade).
            prev_prices: Prior prices, used to update volatility estimates.
                Pass None (e.g. on the terminal step, where there is no new
                price to move to) to skip that update.

        Returns:
            Log return relative to the previous nav_history entry (cost- and
            price-inclusive).
        """
        new_nav = self.compute_nav(prices)

        if prev_prices is not None:
            for ticker in self.tickers:
                if ticker in prev_prices and ticker in prices:
                    ret = np.log(prices[ticker] / prev_prices[ticker])
                    self._return_buffer[ticker].append(ret)

                    # Keep last 20 returns for vol estimation
                    if len(self._return_buffer[ticker]) > 20:
                        self._return_buffer[ticker] = self._return_buffer[ticker][-20:]

                    if len(self._return_buffer[ticker]) >= 5:
                        # DAILY-scale (audit defect #8; previously x sqrt(252))
                        self._vol_estimates[ticker] = float(np.std(self._return_buffer[ticker]))

        # Update peak NAV
        if new_nav > self._peak_nav:
            self._peak_nav = new_nav

        old_nav = self._nav_history[-1]
        self._nav_history.append(new_nav)

        if old_nav > 0:
            return np.log(new_nav / old_nav)
        return 0.0

    def get_nav_history(self) -> np.ndarray:
        """Return NAV history as numpy array."""
        return np.array(self._nav_history)

    def get_returns_history(self) -> np.ndarray:
        """Return log returns history."""
        nav = self.get_nav_history()
        if len(nav) < 2:
            return np.array([])
        return np.diff(np.log(nav))

    def get_metrics(self) -> Dict[str, float]:
        """Compute performance metrics."""
        nav = self.get_nav_history()
        returns = self.get_returns_history()

        if len(returns) == 0:
            return {}

        # Annualization factor (assuming daily data)
        ann_factor = 252

        # Cumulative return
        total_return = (nav[-1] / nav[0]) - 1

        # Annualized return
        n_days = len(returns)
        ann_return = (1 + total_return) ** (ann_factor / n_days) - 1

        # Volatility
        volatility = np.std(returns) * np.sqrt(ann_factor)

        # Sharpe ratio (assuming 0 risk-free rate for simplicity)
        sharpe = ann_return / volatility if volatility > 0 else 0

        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) * np.sqrt(ann_factor) if len(downside_returns) > 0 else volatility
        sortino = ann_return / downside_std if downside_std > 0 else 0

        # Maximum drawdown
        peak = np.maximum.accumulate(nav)
        drawdowns = (peak - nav) / peak
        max_drawdown = np.max(drawdowns)

        # Calmar ratio
        calmar = ann_return / max_drawdown if max_drawdown > 0 else 0

        return {
            'total_return': total_return,
            'ann_return': ann_return,
            'volatility': volatility,
            'sharpe': sharpe,
            'sortino': sortino,
            'max_drawdown': max_drawdown,
            'calmar': calmar,
            'final_nav': nav[-1]
        }


class RiskManager:
    """
    Computes risk metrics and cost signals for CMDP constraints.
    """

    def __init__(
            self,
            dd_limit: float = 0.10,
            turnover_limit: float = 0.5,
            leverage_limit: float = 1.5,
            tail_quantile: float = 0.05,
            tail_window: int = 20
    ):
        self.dd_limit = dd_limit
        self.turnover_limit = turnover_limit
        self.leverage_limit = leverage_limit
        self.tail_quantile = tail_quantile
        self.tail_window = tail_window

        self._return_history: List[float] = []

    def compute_costs(
            self,
            state: PortfolioState,
            turnover: float,
            latest_return: float
    ) -> Dict[str, float]:
        """
        Compute constraint violation costs.

        Returns dict with:
            - cost_dd: drawdown violation cost
            - cost_turnover: turnover violation cost
            - cost_leverage: leverage violation cost
            - cost_tail: tail loss cost (if enabled)
        """
        costs = {}

        # Drawdown cost
        dd_violation = max(0, state.current_drawdown - self.dd_limit)
        costs['cost_dd'] = dd_violation

        # Turnover cost
        turnover_violation = max(0, turnover - self.turnover_limit)
        costs['cost_turnover'] = turnover_violation

        # Leverage cost
        leverage_violation = max(0, state.gross_leverage - self.leverage_limit)
        costs['cost_leverage'] = leverage_violation

        # Tail loss cost
        self._return_history.append(latest_return)
        if len(self._return_history) > self.tail_window:
            self._return_history = self._return_history[-self.tail_window:]

        if len(self._return_history) >= self.tail_window:
            sorted_returns = np.sort(self._return_history)
            var_idx = int(self.tail_quantile * len(sorted_returns))
            var_threshold = sorted_returns[var_idx]

            # Cost if current return is in tail
            if latest_return < var_threshold:
                costs['cost_tail'] = abs(latest_return - var_threshold)
            else:
                costs['cost_tail'] = 0.0
        else:
            costs['cost_tail'] = 0.0

        return costs

    def get_cost_vector(self, costs: Dict[str, float]) -> np.ndarray:
        """Convert cost dict to numpy array."""
        return np.array([
            costs['cost_dd'],
            costs['cost_turnover'],
            costs['cost_leverage']
        ], dtype=np.float32)

    def reset(self):
        """Reset internal state."""
        self._return_history = []


if __name__ == "__main__":
    # Test portfolio engine
    print("Testing PortfolioEngine...")

    tickers = ['AAPL', 'GOOGL', 'MSFT']
    engine = PortfolioEngine(tickers, initial_capital=100000)

    prices = {'AAPL': 150.0, 'GOOGL': 2800.0, 'MSFT': 300.0}

    # Initial state
    state = engine.get_state(prices)
    print(f"Initial NAV: ${state.nav:,.2f}")
    print(f"Initial weights: {state.weights}")

    # Execute trades (mutates positions/cash only -- no NAV recorded yet)
    target_weights = np.array([0.4, 0.3, 0.3])
    tc, info = engine.execute_trades(target_weights, prices)
    print(f"\nAfter trading (pre-mark):")
    print(f"Transaction cost: ${tc:,.2f}")

    # Simulate price changes and mark-to-market: one NAV record for this step,
    # reflecting both the trade above and this price move.
    new_prices = {'AAPL': 155.0, 'GOOGL': 2750.0, 'MSFT': 310.0}
    ret = engine.mark_to_market(new_prices, prev_prices=prices)

    state = engine.get_state(new_prices)
    print(f"\nAfter price change:")
    print(f"NAV: ${state.nav:,.2f}")
    print(f"Weights: {state.weights}")
    print(f"Return: {ret:.4f}")
    print(f"Drawdown: {state.current_drawdown:.2%}")

    # Get metrics
    metrics = engine.get_metrics()
    print(f"\nMetrics: {metrics}")