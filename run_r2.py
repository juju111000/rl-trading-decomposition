#!/usr/bin/env python3
"""
Detached, resumable driver for the R2 long batch (E4, E5, E6). Launch with
launch_r2.py; poll with check_r2_progress.py. One process, one combo at a
time, per-combo resumability via results/r2/<exp>/<arm>/seed_<n>/eval_result.json
(a combo with a marker is skipped on relaunch; nothing already on disk is redone).

E4 -- matched-initialization cohort. Seeds 1-5 retrained through EXACTLY
      run_hardening._train_and_eval (the path seeds 6-20 used). Before any
      rollout or gradient step the trainer's freshly-created network is compared
      tensor-by-tensor (torch.equal) with the untrained arm's network rebuilt
      exactly as run_hardening.process_untrained builds it; both inits are saved.
      A mismatch aborts that seed BEFORE training and records it as UNMATCHED.
      Seeds 6-20 are re-verified by the same reconstruction (no training).
E5 -- oracle60_5d / oracle100_5d x seeds 1-20 (r2_oracle.build_oracle5_slices).
      NON-DEPLOYABLE / DIAGNOSTIC.
E6 -- seeded ladder a-h x seeds 1-5 (LADDER below). f == canonical config, so
      f is REUSED from E4 (identical config/seed/code path) after an exact
      config-equality assertion, not retrained.

Every training run is instrumented RNG-neutrally (wrapping env.step and the
rollout buffer's get(); no random draws) to record exact per-step cost-binding
counts and raw pre-normalization cost-advantage std per iteration (for E1).

Fold 4 is never loaded: r2_common installs a fail-closed guard at import.
"""
import argparse
import json
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import r2_common as rc  # installs the fold-4 guard first

import numpy as np
import pandas as pd
import torch

import main as main_module
from networks import create_actor_critic
from seeding import seed_everything
from run_hardening import _train_and_eval, _run_deterministic_episode
from run_validation_workstream import _base_config, _load_fold2_slices, build_env_with_features, _apply_overrides
from r2_oracle import build_oracle5_slices

R2 = rc.R2_DIR
BATCH_DIR = R2 / "batch"
HEARTBEAT_SECONDS = 30

OFF = {"environment.action_smoothing_beta": None, "environment.no_trade_band": None,
       "environment.decision_frequency_days": 1, "action_space.gross_leverage_cap": None,
       "ppo.entropy_decay": 1.0, "ppo.log_std_decay_floor": None}
SMOOTH = {"environment.action_smoothing_beta": 0.1}
CAP = {"action_space.gross_leverage_cap": 2.0}
ANNEAL = {"ppo.entropy_decay": 0.98435, "ppo.log_std_decay_floor": -1.897}   # 0.05 -> 0.005 over training
BAND = {"environment.no_trade_band": 0.10}
CADENCE = {"environment.decision_frequency_days": 5}
LADDER = {
    "a": ("daily, no band, no smoothing, no cap", [OFF]),
    "b": ("a + smoothing 0.1", [OFF, SMOOTH]),
    "c": ("a + 2.0x cap", [OFF, CAP]),
    "d": ("a + smoothing + cap", [OFF, SMOOTH, CAP]),
    "e": ("d + exploration annealing (entropy 0.05->0.005, log_std ceiling -> -1.897)", [OFF, SMOOTH, CAP, ANNEAL]),
    "f": ("e + band 10% + 5-day cadence (== canonical)", [OFF, SMOOTH, CAP, ANNEAL, BAND, CADENCE]),
    "g": ("e + band 10% only (daily decisions)", [OFF, SMOOTH, CAP, ANNEAL, BAND]),
    "h": ("e + 5-day cadence only (no band)", [OFF, SMOOTH, CAP, ANNEAL, CADENCE]),
}
E5_ARMS = {"oracle60_5d": 0.6, "oracle100_5d": 1.0}


class InitMismatch(RuntimeError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def _flat(d, p=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, p + k + "."))
        else:
            out[p + k] = v
    return out


def config_diff(a, b):
    """Field-by-field VALUE comparison (5 == 5.0); a JSON-string comparison is too strict."""
    fa, fb = _flat(a.to_dict()), _flat(b.to_dict())
    return {k: (fa.get(k), fb.get(k)) for k in set(fa) | set(fb) if fa.get(k) != fb.get(k)}


def ladder_config(seed, key):
    cfg = _base_config(seed)
    for ov in LADDER[key][1]:
        _apply_overrides(cfg, ov)
    return cfg


def combo_list():
    combos = [("e4_verify", "canonical", s) for s in range(6, 21)]
    combos += [("e4", "canonical", s) for s in range(1, 6)]
    combos += [("e6", "f", s) for s in range(1, 6)]                       # reuse of E4, no training
    combos += [("e6", k, s) for k in "abcde" for s in range(1, 6)]
    combos += [("e5", arm, s) for arm in E5_ARMS for s in range(1, 21)]
    combos += [("e6", k, s) for k in "gh" for s in range(1, 6)]
    return combos


def is_training(combo):
    exp, arm, _ = combo
    return exp in ("e4", "e5") or (exp == "e6" and arm != "f")


def marker(exp, arm, seed):
    return R2 / exp / arm / f"seed_{seed}" / "eval_result.json"


# =============================================================================
# RNG-neutral instrumentation + init capture
# =============================================================================

def _snapshot(module):
    return {k: v.detach().clone().cpu() for k, v in module.state_dict().items()}


def _instrument(trainer, store):
    env = trainer.env
    orig_step = env.step
    counts = {'steps': 0, 'nonzero': [0, 0, 0], 'excess_sum': [0.0, 0.0, 0.0]}

    def step(action):
        out = orig_step(action)
        cv = np.asarray(out[4]['cost_vec'], dtype=float)
        counts['steps'] += 1
        for k in range(3):
            if cv[k] > 0:
                counts['nonzero'][k] += 1
                counts['excess_sum'][k] += float(cv[k])
        return out

    env.step = step
    buf = trainer.buffer
    orig_get = buf.get
    per_iter = []

    def get(normalize_advantages=True):
        ptr = buf.ptr if buf.ptr else len(buf.costs)
        ca = np.asarray(buf.cost_advantages[:ptr], dtype=float)
        c = np.asarray(buf.costs[:ptr], dtype=float)
        per_iter.append({'raw_cost_adv_std': ca.std(axis=0).tolist(), 'frac_steps_nonzero': (c > 0).mean(axis=0).tolist(),
                         'reward_adv_std': float(np.asarray(buf.advantages[:ptr], dtype=float).std())})
        return orig_get(normalize_advantages=normalize_advantages)

    buf.get = get
    store['counts'], store['per_iteration'] = counts, per_iter


def _with_trainer_hook(on_trainer, fn):
    orig = main_module.create_trainer

    def hook(env, config, trainer_type="ppo_lagrangian", device="cpu"):
        tr = orig(env=env, config=config, trainer_type=trainer_type, device=device)
        on_trainer(tr)
        return tr

    main_module.create_trainer = hook
    try:
        return fn()
    finally:
        main_module.create_trainer = orig


def _save_instrumentation(path, store):
    if 'counts' not in store:
        return
    c = store['counts']
    out = {'steps': c['steps'],
           'frac_steps_nonzero': [n / max(c['steps'], 1) for n in c['nonzero']],
           'mean_excess_when_nonzero': [c['excess_sum'][k] / c['nonzero'][k] if c['nonzero'][k] else 0.0 for k in range(3)],
           'streams': ['drawdown', 'turnover', 'leverage'], 'per_iteration': store['per_iteration']}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out), encoding='utf-8')


def untrained_reference(seed, config, slices):
    """Exactly run_hardening.process_untrained's construction."""
    main_module.calibrate_regime(config, slices['train_aligned'])   # audit defect #9 (no RNG)
    seed_everything(seed)
    test_env = build_env_with_features(slices['test_features'], slices['test_prices'], config,
                                       slices['test_aligned'], mode="test")
    policy = create_actor_critic(
        obs_dim=test_env.observation_space.shape[0], action_dim=len(slices['tickers']), n_cost_constraints=3,
        config=config.ppo, action_type=config.action_space.action_type,
        n_discrete_levels=len(config.action_space.discrete_levels), shared=False)
    policy.eval()
    return policy, test_env


def compare_sd(a, b):
    if a.keys() != b.keys():
        return False, ["key sets differ"]
    bad = [k for k in a if a[k].shape != b[k].shape or not torch.equal(a[k], b[k])]
    return not bad, bad


# =============================================================================
# Combos
# =============================================================================

def run_e4_verify(seed):
    """Seeds 6-20: re-verify by reconstruction (no training)."""
    config = _base_config(seed)
    slices = _load_fold2_slices(config)
    ref, _ = untrained_reference(seed, config, slices)
    ref_sd = _snapshot(ref)
    seed_everything(seed)
    train_env = build_env_with_features(slices['train_features'], slices['train_prices'], config,
                                        slices['train_aligned'], mode="train")
    tr = main_module.create_trainer(env=train_env, config=config, trainer_type="ppo_lagrangian",
                                    device='cuda' if torch.cuda.is_available() else 'cpu')
    eq, bad = compare_sd(ref_sd, _snapshot(tr.actor_critic))
    return {'arm': 'canonical', 'seed': seed, 'init_equal': eq, 'mismatched_tensors': bad,
            'n_tensors': len(ref_sd), 'status': 'previously-verified (2026-10-09), re-verified by reconstruction',
            'trained_checkpoint': str(rc.fold2_canonical_checkpoint(seed) / "final_model.pt"),
            'trained_checkpoint_sha256': rc.file_sha256(rc.fold2_canonical_checkpoint(seed) / "final_model.pt")}


def run_e4(seed):
    out = R2 / "e4"
    init_dir = out / "init"
    init_dir.mkdir(parents=True, exist_ok=True)
    config = _base_config(seed)
    slices = _load_fold2_slices(config)

    ref, test_env = untrained_reference(seed, config, slices)
    ref_sd = _snapshot(ref)
    torch.save(ref_sd, init_dir / f"untrained_init_seed{seed}.pt")
    # functional check: the rebuilt untrained network reproduces the hardening untrained arm's result
    urow = _run_deterministic_episode(test_env, ref, slices['tickers'], slices['test_prices'])
    hard = pd.read_csv(rc.HARDENING_DIR / "hardening_results.csv")
    hu = float(hard[(hard.arm == "untrained") & (hard.seed == seed)]['return'].iloc[0])

    store, verdict = {}, {}

    def on_trainer(tr):
        sd = _snapshot(tr.actor_critic)
        torch.save(sd, init_dir / f"trainer_init_seed{seed}.pt")
        eq, bad = compare_sd(ref_sd, sd)
        verdict.update({'init_equal': eq, 'mismatched_tensors': bad, 'n_tensors': len(sd)})
        if not eq:
            raise InitMismatch(f"seed {seed}: {len(bad)} tensors differ before training")
        _instrument(tr, store)

    try:
        row = _with_trainer_hook(on_trainer, lambda: _train_and_eval(
            seed, config, slices, f"canonical_seed{seed}", out / "canonical_training"))
        status = "newly-reconstructed, verified equal before training"
    except InitMismatch as e:
        row, status = {}, f"UNMATCHED: {e} (not trained, excluded from paired analysis)"
    _save_instrumentation(out / "canonical_training" / f"canonical_seed{seed}" / "binding_instrumentation.json", store)
    ckpt = out / "canonical_training" / f"canonical_seed{seed}" / "final_model.pt"
    row.update({'arm': 'canonical', 'seed': seed, 'status': status, **verdict,
                'untrained_return_rebuilt': urow['return'], 'untrained_return_hardening': hu,
                'untrained_rebuild_matches_hardening': abs(urow['return'] - hu) < 1e-9,
                'untrained_init_checkpoint': str(init_dir / f"untrained_init_seed{seed}.pt"),
                'trainer_init_checkpoint': str(init_dir / f"trainer_init_seed{seed}.pt"),
                'trained_checkpoint': str(ckpt) if ckpt.exists() else None,
                'trained_checkpoint_sha256': rc.file_sha256(ckpt) if ckpt.exists() else None})
    return row


def run_e5(arm, seed):
    out = R2 / "e5"
    config = _base_config(seed)
    slices = build_oracle5_slices(config, E5_ARMS[arm])
    store = {}
    row = _with_trainer_hook(lambda tr: _instrument(tr, store), lambda: _train_and_eval(
        seed, config, slices, f"{arm}_seed{seed}", out / f"{arm}_training"))
    _save_instrumentation(out / f"{arm}_training" / f"{arm}_seed{seed}" / "binding_instrumentation.json", store)
    ckpt = out / f"{arm}_training" / f"{arm}_seed{seed}" / "final_model.pt"
    row.update({'arm': arm, 'seed': seed, 'accuracy': E5_ARMS[arm], 'non_deployable': True,
                'trained_checkpoint': str(ckpt), 'trained_checkpoint_sha256': rc.file_sha256(ckpt)})
    return row


def run_e6(key, seed):
    out = R2 / "e6"
    config = ladder_config(seed, key)
    if key == "f":
        canon = _base_config(seed)
        diff = config_diff(config, canon)
        assert not diff, f"ladder f != canonical config: {diff}"
        src = marker("e4", "canonical", seed)
        if not src.exists():
            raise RuntimeError("E4 seed not complete yet")
        row = json.loads(src.read_text())
        if not row.get('init_equal', False) or 'return' not in row:
            raise RuntimeError(f"E4 seed {seed} unusable for reuse: {row.get('status')}")
        keep = {k: row[k] for k in ('return', 'gross_price_pnl', 'txn_cost', 'turnover', 'max_dd', 'sharpe',
                                    'dd_violation_rate', 'trained_checkpoint', 'trained_checkpoint_sha256') if k in row}
        keep.update({'config': key, 'seed': seed, 'description': LADDER[key][0],
                     'reused_from': str(src), 'note': 'identical config, seed and code path as E4 -> reused, not retrained'})
        return keep
    slices = _load_fold2_slices(config)
    store = {}
    row = _with_trainer_hook(lambda tr: _instrument(tr, store), lambda: _train_and_eval(
        seed, config, slices, f"ladder_{key}_seed{seed}", out / f"ladder_{key}_training"))
    _save_instrumentation(out / f"ladder_{key}_training" / f"ladder_{key}_seed{seed}" / "binding_instrumentation.json", store)
    ckpt = out / f"ladder_{key}_training" / f"ladder_{key}_seed{seed}" / "final_model.pt"
    row.update({'config': key, 'seed': seed, 'description': LADDER[key][0],
                'overrides': {k: v for ov in LADDER[key][1] for k, v in ov.items()},
                'trained_checkpoint': str(ckpt), 'trained_checkpoint_sha256': rc.file_sha256(ckpt)})
    return row


def run_combo(combo):
    exp, arm, seed = combo
    if exp == "e4_verify":
        return run_e4_verify(seed)
    if exp == "e4":
        return run_e4(seed)
    if exp == "e5":
        return run_e5(arm, seed)
    return run_e6(arm, seed)


# =============================================================================
# Driver
# =============================================================================

def _rewrite_tables():
    for exp in ("e4_verify", "e4", "e5", "e6"):
        rows = [json.loads(p.read_text()) for p in sorted((R2 / exp).glob("*/seed_*/eval_result.json"))]
        if not rows:
            continue
        df = pd.DataFrame(rows)
        name = {"e4_verify": "e4_verification_seeds6_20", "e4": "e4_results", "e5": "e5_results",
                "e6": "ladder_seeded"}[exp]
        target = (R2 / f"{name}.csv") if exp == "e6" else (R2 / exp / f"{name}.csv")
        df.drop(columns=[c for c in df.columns if df[c].map(lambda v: isinstance(v, (list, dict))).any()]).to_csv(
            target, index=False)
        rc.write_manifest(exp, {'experiment': exp, 'script': 'run_r2.py', 'rows': rows, 'fold4_loaded': False})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="restrict to experiments, e.g. e4 e6")
    args = ap.parse_args()

    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    log = open(BATCH_DIR / "run.log", "a", buffering=1, encoding="utf-8")
    sys.stdout = sys.stderr = log

    combos = [c for c in combo_list() if not args.only or c[0].split('_')[0] in args.only or c[0] in args.only]
    state = {'status': 'running', 'pid': __import__('os').getpid(), 'started': now(), 'total_combos': len(combos),
             'total_training_runs': sum(is_training(c) for c in combos), 'current': None,
             'completed': 0, 'failed': [], 'last_update': now()}

    def heartbeat():
        while state['status'] == 'running':
            state['completed'] = sum(marker(*c).exists() for c in combos)
            state['last_update'] = now()
            (BATCH_DIR / "heartbeat.json").write_text(json.dumps(state, indent=2), encoding='utf-8')
            time.sleep(HEARTBEAT_SECONDS)

    threading.Thread(target=heartbeat, daemon=True).start()
    print(f"\n=== R2 batch start {now()} -- {len(combos)} combos, {state['total_training_runs']} training runs ===", flush=True)
    try:
        for combo in combos:
            m = marker(*combo)
            if m.exists():
                continue
            state['current'] = {'combo': list(combo), 'started': now()}
            print(f"\n--- {combo} start {now()}", flush=True)
            t0 = time.time()
            try:
                row = run_combo(combo)
            except Exception:
                tb = traceback.format_exc()
                print(tb, flush=True)
                state['failed'].append({'combo': list(combo), 'error': tb.splitlines()[-1], 'at': now()})
                continue
            row['runtime_seconds'] = round(time.time() - t0, 1)
            row['completed_utc'] = now()
            m.parent.mkdir(parents=True, exist_ok=True)
            m.write_text(json.dumps(row, indent=2, default=str), encoding='utf-8')
            print(f"--- {combo} done in {row['runtime_seconds']:.0f}s: return={row.get('return')}", flush=True)
            _rewrite_tables()
        state['status'] = 'finished' if not state['failed'] else 'finished_with_failures'
    except BaseException:
        state['status'] = 'crashed'
        print(traceback.format_exc(), flush=True)
        raise
    finally:
        state['completed'] = sum(marker(*c).exists() for c in combos)
        state['current'] = None
        state['last_update'] = now()
        (BATCH_DIR / "heartbeat.json").write_text(json.dumps(state, indent=2), encoding='utf-8')
        print(f"=== R2 batch end {now()} status={state['status']} ===", flush=True)


if __name__ == "__main__":
    main()
