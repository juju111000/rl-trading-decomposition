"""
Logging and Visualization Module

Implements:
- TensorBoard integration for training metrics
- CSV logging for experiment tracking
- Matplotlib plots for analysis
- Performance dashboards
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import json
import numpy as np
import pandas as pd
from datetime import datetime
import warnings

# Optional imports with graceful fallback
try:
    from torch.utils.tensorboard import SummaryWriter

    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    warnings.warn("TensorBoard not available. Install with: pip install tensorboard")

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.gridspec import GridSpec

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("Matplotlib not available. Install with: pip install matplotlib")


class TensorBoardLogger:
    """TensorBoard logging wrapper."""

    def __init__(self, log_dir: str, experiment_name: Optional[str] = None):
        if not HAS_TENSORBOARD:
            raise ImportError("TensorBoard not available")

        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.log_dir = Path(log_dir) / experiment_name
        self.writer = SummaryWriter(str(self.log_dir))
        self.step = 0

    def log_scalar(self, tag: str, value: float, step: Optional[int] = None):
        """Log scalar value."""
        if step is None:
            step = self.step
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: Optional[int] = None):
        """Log multiple scalars under one tag."""
        if step is None:
            step = self.step
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_histogram(self, tag: str, values: np.ndarray, step: Optional[int] = None):
        """Log histogram of values."""
        if step is None:
            step = self.step
        self.writer.add_histogram(tag, values, step)

    def log_training_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log training metrics from PPO trainer."""
        if step is None:
            step = self.step

        # Losses
        if 'policy_loss' in metrics:
            self.log_scalar('loss/policy', metrics['policy_loss'], step)
        if 'value_loss' in metrics:
            self.log_scalar('loss/value', metrics['value_loss'], step)
        if 'entropy' in metrics:
            self.log_scalar('loss/entropy', metrics['entropy'], step)

        # PPO stats
        if 'approx_kl' in metrics:
            self.log_scalar('ppo/approx_kl', metrics['approx_kl'], step)
        if 'clip_fraction' in metrics:
            self.log_scalar('ppo/clip_fraction', metrics['clip_fraction'], step)

        # Rewards and costs
        if 'mean_reward' in metrics:
            self.log_scalar('performance/mean_reward', metrics['mean_reward'], step)
        if 'rollout_return_sum' in metrics:
            self.log_scalar('performance/rollout_return_sum', metrics['rollout_return_sum'], step)

        # Constraints
        if 'mean_costs' in metrics:
            for i, cost in enumerate(metrics['mean_costs']):
                self.log_scalar(f'costs/cost_{i}', cost, step)
        if 'lambdas' in metrics:
            for i, lam in enumerate(metrics['lambdas']):
                self.log_scalar(f'dual/lambda_{i}', lam, step)
        if 'constraint_violations' in metrics:
            for i, viol in enumerate(metrics['constraint_violations']):
                self.log_scalar(f'violations/constraint_{i}', viol, step)

        # Shield
        if 'shield_intervention_rate' in metrics:
            self.log_scalar('shield/intervention_rate', metrics['shield_intervention_rate'], step)
        if 'mean_shield_magnitude' in metrics:
            self.log_scalar('shield/mean_magnitude', metrics['mean_shield_magnitude'], step)

        self.step = step + 1

    def log_evaluation_metrics(self, metrics: Dict[str, Any], prefix: str = 'eval'):
        """Log evaluation metrics."""
        step = self.step

        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                self.log_scalar(f'{prefix}/{key}', value, step)

    def close(self):
        """Close writer."""
        self.writer.close()


class CSVLogger:
    """Simple CSV logger for experiment tracking."""

    def __init__(self, log_dir: str, filename: str = 'training_log.csv'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.filepath = self.log_dir / filename
        self.columns = None
        self._initialized = False

    def log(self, metrics: Dict[str, Any]):
        """Log metrics to CSV."""
        # Flatten nested structures
        flat_metrics = self._flatten_dict(metrics)

        if not self._initialized:
            self.columns = list(flat_metrics.keys())
            with open(self.filepath, 'w') as f:
                f.write(','.join(self.columns) + '\n')
            self._initialized = True

        # Write row
        values = [str(flat_metrics.get(col, '')) for col in self.columns]
        with open(self.filepath, 'a') as f:
            f.write(','.join(values) + '\n')

    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep).items())
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    items.append((f"{new_key}_{i}", item))
            else:
                items.append((new_key, v))
        return dict(items)

    def load(self) -> pd.DataFrame:
        """Load logged data as DataFrame."""
        return pd.read_csv(self.filepath)


class ExperimentLogger:
    """Combined logger with TensorBoard and CSV."""

    def __init__(
            self,
            log_dir: str,
            experiment_name: Optional[str] = None,
            use_tensorboard: bool = True
    ):
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.csv_logger = CSVLogger(str(self.log_dir))

        self.tb_logger = None
        if use_tensorboard and HAS_TENSORBOARD:
            self.tb_logger = TensorBoardLogger(str(log_dir), experiment_name)

    def log_training(self, metrics: Dict[str, Any], step: int):
        """Log training metrics."""
        metrics['step'] = step
        self.csv_logger.log(metrics)

        if self.tb_logger:
            self.tb_logger.log_training_metrics(metrics, step)

    def log_evaluation(self, metrics: Dict[str, Any], prefix: str = 'eval'):
        """Log evaluation metrics."""
        self.csv_logger.log({f'{prefix}_{k}': v for k, v in metrics.items()})

        if self.tb_logger:
            self.tb_logger.log_evaluation_metrics(metrics, prefix)

    def save_config(self, config: Dict[str, Any]):
        """Save configuration."""
        with open(self.log_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2, default=str)

    def close(self):
        """Close loggers."""
        if self.tb_logger:
            self.tb_logger.close()


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_training_curves(
        log_path: Union[str, Path],
        save_path: Optional[str] = None,
        figsize: tuple = (14, 10)
):
    """
    Plot training curves from CSV log.

    Args:
        log_path: Path to training log CSV
        save_path: Optional path to save figure
        figsize: Figure size
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib not available")

    df = pd.read_csv(log_path)

    fig, axes = plt.subplots(3, 2, figsize=figsize)

    # Rewards
    ax = axes[0, 0]
    if 'mean_reward' in df.columns:
        ax.plot(df['step'] if 'step' in df.columns else df.index, df['mean_reward'])
    ax.set_title('Mean Reward per Step')
    ax.set_xlabel('Step')
    ax.set_ylabel('Reward')
    ax.grid(True, alpha=0.3)

    # Losses
    ax = axes[0, 1]
    if 'policy_loss' in df.columns:
        ax.plot(df.index, df['policy_loss'], label='Policy')
    if 'value_loss' in df.columns:
        ax.plot(df.index, df['value_loss'], label='Value')
    ax.set_title('Losses')
    ax.set_xlabel('Iteration')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Dual variables (lambdas)
    ax = axes[1, 0]
    lambda_cols = [c for c in df.columns if c.startswith('lambdas_')]
    for col in lambda_cols:
        ax.plot(df.index, df[col], label=col.replace('lambdas_', 'λ_'))
    ax.set_title('Lagrange Multipliers')
    ax.set_xlabel('Iteration')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Cost violations
    ax = axes[1, 1]
    cost_cols = [c for c in df.columns if c.startswith('mean_costs_')]
    for col in cost_cols:
        ax.plot(df.index, df[col], label=col.replace('mean_costs_', 'cost_'))
    ax.set_title('Mean Costs')
    ax.set_xlabel('Iteration')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Shield interventions
    ax = axes[2, 0]
    if 'shield_intervention_rate' in df.columns:
        ax.plot(df.index, df['shield_intervention_rate'])
    ax.set_title('Shield Intervention Rate')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Rate')
    ax.grid(True, alpha=0.3)

    # PPO stats
    ax = axes[2, 1]
    if 'entropy' in df.columns:
        ax.plot(df.index, df['entropy'], label='Entropy')
    if 'approx_kl' in df.columns:
        ax2 = ax.twinx()
        ax2.plot(df.index, df['approx_kl'], color='orange', label='Approx KL')
        ax2.set_ylabel('KL', color='orange')
    ax.set_title('PPO Stats')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Entropy')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")

    return fig


def plot_evaluation_results(
        nav_history: List[float],
        drawdown_history: List[float],
        regime_beliefs: Optional[List[np.ndarray]] = None,
        dates: Optional[List] = None,
        title: str = 'Evaluation Results',
        save_path: Optional[str] = None,
        figsize: tuple = (14, 12)
):
    """
    Plot comprehensive evaluation results.

    Args:
        nav_history: Portfolio NAV over time
        drawdown_history: Drawdown over time
        regime_beliefs: Optional regime belief vectors
        dates: Optional date labels
        title: Plot title
        save_path: Optional save path
        figsize: Figure size
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib not available")

    n_plots = 3 if regime_beliefs is None else 4
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)

    x = dates if dates is not None else range(len(nav_history))

    # NAV
    ax = axes[0]
    ax.plot(x, nav_history, linewidth=1.5)
    ax.set_ylabel('NAV ($)')
    ax.set_title(f'{title} - Portfolio Value')
    ax.grid(True, alpha=0.3)

    # Add horizontal line at initial value
    ax.axhline(y=nav_history[0], color='gray', linestyle='--', alpha=0.5)

    # Drawdown
    ax = axes[1]
    ax.fill_between(x, 0, drawdown_history, alpha=0.7, color='red')
    ax.set_ylabel('Drawdown')
    ax.set_title('Drawdown')
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    # Returns
    ax = axes[2]
    returns = np.diff(np.log(nav_history))
    ax.bar(x[1:] if dates is not None else range(len(returns)), returns, width=1, alpha=0.7)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_ylabel('Log Return')
    ax.set_title('Daily Returns')
    ax.grid(True, alpha=0.3)

    # Regime beliefs
    if regime_beliefs is not None:
        ax = axes[3]
        beliefs_array = np.array(regime_beliefs)
        n_regimes = beliefs_array.shape[1]

        colors = ['green', 'red', 'gray'][:n_regimes]
        labels = ['Bull', 'Bear', 'Sideways'][:n_regimes]

        ax.stackplot(x[:len(beliefs_array)], beliefs_array.T,
                     labels=labels, colors=colors, alpha=0.7)
        ax.set_ylabel('Belief')
        ax.set_title('Regime Beliefs')
        ax.legend(loc='upper right')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    # Format x-axis if dates provided
    if dates is not None and HAS_MATPLOTLIB:
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)

    axes[-1].set_xlabel('Date' if dates is not None else 'Step')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved evaluation plot to {save_path}")

    return fig


def plot_comparison_table(
        comparison_df: pd.DataFrame,
        metrics: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        figsize: tuple = (12, 6)
):
    """
    Create visual comparison table/heatmap.

    Args:
        comparison_df: DataFrame from compare_methods()
        metrics: Metrics to display
        save_path: Optional save path
        figsize: Figure size
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib not available")

    if metrics is None:
        # Extract mean columns
        metrics = [c.replace('_mean', '') for c in comparison_df.columns if c.endswith('_mean')]

    # Get mean values
    mean_cols = [f'{m}_mean' for m in metrics if f'{m}_mean' in comparison_df.columns]
    data = comparison_df[mean_cols].copy()
    data.columns = [c.replace('_mean', '') for c in data.columns]

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = ax.imshow(data.values, cmap='RdYlGn', aspect='auto')

    # Set ticks
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels(data.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Value')

    # Add text annotations
    for i in range(len(data.index)):
        for j in range(len(data.columns)):
            val = data.iloc[i, j]
            text = f'{val:.3f}' if abs(val) < 100 else f'{val:.1f}'
            ax.text(j, i, text, ha='center', va='center',
                    color='white' if abs(val) > data.values.max() * 0.7 else 'black')

    ax.set_title('Method Comparison')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")

    return fig


def plot_constraint_analysis(
        cost_history: np.ndarray,
        lambda_history: np.ndarray,
        constraint_names: List[str] = None,
        constraint_limits: List[float] = None,
        save_path: Optional[str] = None,
        figsize: tuple = (14, 8)
):
    """
    Plot constraint satisfaction over training.

    Args:
        cost_history: Array of shape (n_iterations, n_costs)
        lambda_history: Array of shape (n_iterations, n_costs)
        constraint_names: Names for each constraint
        constraint_limits: Limit values for each constraint
        save_path: Optional save path
        figsize: Figure size
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib not available")

    n_costs = cost_history.shape[1]

    if constraint_names is None:
        constraint_names = [f'Constraint {i}' for i in range(n_costs)]
    if constraint_limits is None:
        constraint_limits = [0.0] * n_costs

    fig, axes = plt.subplots(n_costs, 2, figsize=figsize)
    if n_costs == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_costs):
        # Cost/violation over time
        ax = axes[i, 0]
        ax.plot(cost_history[:, i], label='Cost')
        ax.axhline(y=constraint_limits[i], color='red', linestyle='--',
                   label=f'Limit ({constraint_limits[i]})')
        ax.set_ylabel('Cost Value')
        ax.set_title(f'{constraint_names[i]} - Cost')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Lambda over time
        ax = axes[i, 1]
        ax.plot(lambda_history[:, i], color='orange')
        ax.set_ylabel('λ')
        ax.set_title(f'{constraint_names[i]} - Dual Variable')
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel('Iteration')
    axes[-1, 1].set_xlabel('Iteration')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved constraint analysis to {save_path}")

    return fig


def plot_regime_analysis(
        nav_history: np.ndarray,
        regime_beliefs: np.ndarray,
        entropies: np.ndarray,
        dates: Optional[List] = None,
        regime_names: List[str] = None,
        save_path: Optional[str] = None,
        figsize: tuple = (14, 10)
):
    """
    Plot regime-aware performance analysis.

    Args:
        nav_history: Portfolio NAV over time
        regime_beliefs: Regime belief matrix (T, K)
        entropies: Belief entropy over time
        dates: Optional date labels
        regime_names: Names for regimes
        save_path: Optional save path
        figsize: Figure size
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib not available")

    n_regimes = regime_beliefs.shape[1]
    if regime_names is None:
        regime_names = ['Bull', 'Bear', 'Sideways'][:n_regimes]

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(4, 2, figure=fig)

    x = dates if dates is not None else range(len(nav_history))

    # NAV with regime coloring
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(x, nav_history, 'k-', linewidth=1)

    # Color background by dominant regime
    dominant = np.argmax(regime_beliefs, axis=1)
    colors = ['lightgreen', 'lightcoral', 'lightgray'][:n_regimes]

    prev_regime = dominant[0]
    start_idx = 0
    for i, reg in enumerate(dominant):
        if reg != prev_regime or i == len(dominant) - 1:
            ax1.axvspan(x[start_idx], x[i], alpha=0.3, color=colors[prev_regime])
            start_idx = i
            prev_regime = reg

    ax1.set_ylabel('NAV')
    ax1.set_title('Portfolio Value with Regime Background')
    ax1.grid(True, alpha=0.3)

    # Regime beliefs stacked
    ax2 = fig.add_subplot(gs[1, :])
    ax2.stackplot(x[:len(regime_beliefs)], regime_beliefs.T,
                  labels=regime_names, colors=colors, alpha=0.8)
    ax2.set_ylabel('Belief')
    ax2.set_ylim(0, 1)
    ax2.legend(loc='upper right')
    ax2.set_title('Regime Beliefs')
    ax2.grid(True, alpha=0.3)

    # Entropy
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(x[:len(entropies)], entropies)
    ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Transition threshold')
    ax3.set_ylabel('Entropy')
    ax3.set_title('Belief Entropy')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Returns by regime
    ax4 = fig.add_subplot(gs[2, 1])
    returns = np.diff(np.log(nav_history))

    regime_returns = {i: [] for i in range(n_regimes)}
    for i, ret in enumerate(returns):
        if i < len(dominant):
            regime_returns[dominant[i]].append(ret)

    positions = range(n_regimes)
    means = [np.mean(regime_returns[i]) if regime_returns[i] else 0 for i in range(n_regimes)]
    stds = [np.std(regime_returns[i]) if regime_returns[i] else 0 for i in range(n_regimes)]

    ax4.bar(positions, means, yerr=stds, color=colors, capsize=5)
    ax4.set_xticks(positions)
    ax4.set_xticklabels(regime_names)
    ax4.set_ylabel('Mean Return')
    ax4.set_title('Returns by Regime')
    ax4.axhline(y=0, color='black', linewidth=0.5)
    ax4.grid(True, alpha=0.3)

    # Transition analysis
    ax5 = fig.add_subplot(gs[3, :])

    # Find high entropy periods (transitions)
    transition_mask = entropies > 0.5

    # Plot returns with transition highlighting
    ax5.bar(range(len(returns)), returns, width=1, alpha=0.5, color='blue', label='Normal')

    transition_returns = returns.copy()
    transition_returns[~transition_mask[:len(returns)]] = 0
    ax5.bar(range(len(returns)), transition_returns, width=1, alpha=0.8, color='red', label='During transition')

    ax5.axhline(y=0, color='black', linewidth=0.5)
    ax5.set_ylabel('Return')
    ax5.set_xlabel('Step')
    ax5.set_title('Returns During Regime Transitions')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved regime analysis to {save_path}")

    return fig


def create_results_dashboard(
        eval_results: Dict[str, Any],
        training_log_path: Optional[str] = None,
        output_dir: str = './results',
        experiment_name: str = 'experiment'
):
    """
    Create comprehensive results dashboard.

    Args:
        eval_results: Evaluation results dictionary
        training_log_path: Optional path to training log CSV
        output_dir: Output directory for plots
        experiment_name: Name for saved files
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available - skipping dashboard creation")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training curves
    if training_log_path:
        try:
            plot_training_curves(
                training_log_path,
                save_path=str(output_dir / f'{experiment_name}_training.png')
            )
        except Exception as e:
            print(f"Could not plot training curves: {e}")

    # Evaluation results
    if 'nav_history' in eval_results and 'drawdown_history' in eval_results:
        plot_evaluation_results(
            eval_results['nav_history'],
            eval_results['drawdown_history'],
            regime_beliefs=eval_results.get('regime_beliefs'),
            save_path=str(output_dir / f'{experiment_name}_evaluation.png')
        )

    # Comparison table
    if 'comparison' in eval_results:
        plot_comparison_table(
            eval_results['comparison'],
            save_path=str(output_dir / f'{experiment_name}_comparison.png')
        )

    print(f"Dashboard created in {output_dir}")


# ============================================================================
# Report Generation
# ============================================================================

def generate_latex_table(
        comparison_df: pd.DataFrame,
        caption: str = 'Method Comparison',
        label: str = 'tab:comparison'
) -> str:
    """
    Generate LaTeX table from comparison DataFrame.

    Args:
        comparison_df: DataFrame from compare_methods()
        caption: Table caption
        label: LaTeX label

    Returns:
        LaTeX table string
    """
    # Extract mean columns and format
    mean_cols = [c for c in comparison_df.columns if c.endswith('_mean')]

    latex = "\\begin{table}[htbp]\n"
    latex += "\\centering\n"
    latex += f"\\caption{{{caption}}}\n"
    latex += f"\\label{{{label}}}\n"

    # Column format
    n_cols = len(mean_cols) + 1  # methods + metrics
    latex += "\\begin{tabular}{l" + "c" * len(mean_cols) + "}\n"
    latex += "\\toprule\n"

    # Header
    headers = ['Method'] + [c.replace('_mean', '').replace('_', ' ').title() for c in mean_cols]
    latex += " & ".join(headers) + " \\\\\n"
    latex += "\\midrule\n"

    # Data rows
    for idx, row in comparison_df.iterrows():
        values = [str(idx)]
        for col in mean_cols:
            val = row[col]
            std_col = col.replace('_mean', '_std')
            if std_col in comparison_df.columns:
                std = row[std_col]
                values.append(f"${val:.3f} \\pm {std:.3f}$")
            else:
                values.append(f"${val:.3f}$")
        latex += " & ".join(values) + " \\\\\n"

    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"

    return latex


def generate_summary_report(
        results: Dict[str, Any],
        output_path: str
) -> str:
    """
    Generate markdown summary report.

    Args:
        results: Results dictionary
        output_path: Path to save report

    Returns:
        Markdown report string
    """
    report = "# Experiment Results Summary\n\n"
    report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    # Configuration
    if 'config' in results:
        report += "## Configuration\n\n"
        report += "```json\n"
        report += json.dumps(results['config'], indent=2, default=str)
        report += "\n```\n\n"

    # Training summary
    if 'training_metrics' in results:
        report += "## Training Summary\n\n"
        metrics = results['training_metrics']
        if isinstance(metrics, list) and len(metrics) > 0:
            final = metrics[-1]
            report += f"- Total timesteps: {final.get('total_timesteps', 'N/A')}\n"
            report += f"- Final policy loss: {final.get('policy_loss', 'N/A'):.4f}\n"
            report += f"- Final value loss: {final.get('value_loss', 'N/A'):.4f}\n"
            report += f"- Final lambdas: {final.get('lambdas', 'N/A')}\n\n"

    # Evaluation summary
    if 'evaluation' in results:
        report += "## Evaluation Results\n\n"
        eval_results = results['evaluation']

        report += "| Metric | Value |\n"
        report += "|--------|-------|\n"

        for key, value in eval_results.items():
            if isinstance(value, (int, float)):
                report += f"| {key} | {value:.4f} |\n"
        report += "\n"

    # Comparison
    if 'comparison' in results:
        report += "## Method Comparison\n\n"
        comparison = results['comparison']
        if isinstance(comparison, pd.DataFrame):
            report += comparison.to_markdown() + "\n\n"

    # Save report
    with open(output_path, 'w') as f:
        f.write(report)

    print(f"Report saved to {output_path}")

    return report