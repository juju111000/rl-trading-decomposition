"""
Regression test for determinism (A.2).

Two independent process runs of a short training loop, given the same seed,
must produce identical per-iteration training metrics -- network init,
action sampling, env episode-start sampling, and minibatch shuffling must
all trace back to seed_everything(seed) plus the env's own seeded reset.

Wall-clock timing fields (rollout_time/update_time) are legitimately
non-deterministic and are excluded from the comparison.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Fields that are wall-clock and expected to differ between runs.
_NON_DETERMINISTIC_FIELDS = {"rollout_time", "update_time"}


def _build_tiny_config(seed: int):
    from config import (
        Config, DataConfig, FeatureConfig, RegimeConfig, ConstraintConfig,
        ShieldConfig, ActionSpaceConfig, PPOConfig, TrainingConfig,
        EnvironmentConfig, ActionSpaceType, RegimeModelType, ShieldType,
    )

    config = Config()
    config.data = DataConfig(tickers=["ASSET_0", "ASSET_1"])
    config.features = FeatureConfig()
    config.regime = RegimeConfig(model_type=RegimeModelType.HEURISTIC, n_regimes=3)
    config.constraints = ConstraintConfig()
    config.shield = ShieldConfig(shield_type=ShieldType.UNCERTAINTY_AWARE)
    config.action_space = ActionSpaceConfig(action_type=ActionSpaceType.CONTINUOUS)
    config.ppo = PPOConfig(hidden_sizes=[16, 16], batch_size=16, n_epochs=2)
    config.training = TrainingConfig(
        total_timesteps=64 * 3,
        rollout_length=64,
        episode_length=64,
        use_rolling_episodes=True,
        seed=seed,
    )
    config.environment = EnvironmentConfig()
    return config


def _run_short_training(seed: int):
    from seeding import seed_everything
    seed_everything(seed)

    import torch
    torch.set_num_threads(1)  # remove BLAS-thread-order as a source of noise

    from data_loader import SyntheticDataGenerator
    from features import FeatureBuilder
    from regime_model import create_regime_model
    from policy_shield import create_shield
    from trading_env import TradingCMDPEnv
    from trainer import PPOLagrangianTrainer

    config = _build_tiny_config(seed)

    # Data generation is independently seeded (unrelated to the seed under
    # test) so both runs train on identical data.
    gen = SyntheticDataGenerator(n_assets=2, n_days=300, seed=7)
    data, _ = gen.generate_regime_switching_data()

    builder = FeatureBuilder(config.features)
    features_df = builder.compute_features(data)

    prices = {t: data[(t, "Close")].values for t in config.data.tickers}

    regime_model = create_regime_model(config.regime, input_dim=features_df.shape[1])
    regime_model.fit(data)

    shield = create_shield(
        config.shield, config.constraints, n_assets=len(config.data.tickers)
    )

    env = TradingCMDPEnv(
        features=features_df.values,
        prices=prices,
        tickers=config.data.tickers,
        config=config,
        regime_model=regime_model,
        shield=shield,
        mode="train",
    )

    trainer = PPOLagrangianTrainer(env, config, device="cpu")
    metrics_history = trainer.train(total_timesteps=config.training.total_timesteps)

    return [m.to_dict() for m in metrics_history[:3]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    metrics = _run_short_training(args.seed)
    with open(args.out, "w") as f:
        json.dump(metrics, f)


def test_same_seed_reproduces_identical_training_metrics(tmp_path):
    out_a = tmp_path / "run_a.json"
    out_b = tmp_path / "run_b.json"

    for out in (out_a, out_b):
        result = subprocess.run(
            [sys.executable, str(Path(__file__)), "--seed", "123", "--out", str(out)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"subprocess failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    metrics_a = json.loads(out_a.read_text())
    metrics_b = json.loads(out_b.read_text())

    assert len(metrics_a) == 3
    assert len(metrics_b) == 3

    for iter_idx, (a, b) in enumerate(zip(metrics_a, metrics_b)):
        assert a.keys() == b.keys()
        for key in a:
            if key in _NON_DETERMINISTIC_FIELDS:
                continue
            assert a[key] == b[key], (
                f"iteration {iter_idx} field {key!r} differs: "
                f"{a[key]!r} vs {b[key]!r}"
            )


def test_different_seeds_are_not_forced_identical(tmp_path):
    """Sanity check that the comparison above isn't vacuous (e.g. because
    every metric happens to be a constant regardless of seed)."""
    out_a = tmp_path / "run_seed_1.json"
    out_b = tmp_path / "run_seed_2.json"

    for seed, out in ((1, out_a), (2, out_b)):
        result = subprocess.run(
            [sys.executable, str(Path(__file__)), "--seed", str(seed), "--out", str(out)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr

    metrics_a = json.loads(out_a.read_text())
    metrics_b = json.loads(out_b.read_text())

    numeric_keys = [
        k for k, v in metrics_a[0].items()
        if isinstance(v, (int, float)) and k not in _NON_DETERMINISTIC_FIELDS
    ]
    assert any(metrics_a[0][k] != metrics_b[0][k] for k in numeric_keys), (
        "expected at least one metric to differ between seed=1 and seed=2 "
        "on iteration 0 -- otherwise the identical-seed test above would be "
        "vacuously true"
    )


if __name__ == "__main__":
    main()
