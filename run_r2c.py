#!/usr/bin/env python3
"""
Corrected-accounting master batch (audit defects #8 slippage units and #9
regime look-ahead repaired). Detached via launch_r2c.py, polled with
check_r2c_progress.py. All outputs under results/r2/corrected/. Per-combo
resumability: every combo has an on-disk completion marker and is skipped on
relaunch; nothing already completed is redone.

Priority order:
  P1  hardening, fold 2: untrained (eval) + canonical + oracle60 + oracle75, seeds 1-20,
      all trained fresh (no checkpoint reuse). Canonical seeds go through
      run_hardening._train_and_eval with the initial network verified tensor-equal
      to the untrained arm's (torch.equal) BEFORE any gradient step.
  P2  E5 oracle60_5d / oracle100_5d x 20 seeds -- HELD unless --include-p2: the
      pre-launch gate (100% signal-following benchmark under canonical execution
      must net strongly positive) FAILED; see results/r2/corrected/e5/oracle_verification.md.
  P3  E6 ladder a-h x seeds 1-5. f == canonical config == P1 canonical seeds 1-5
      (identical config, seed and code path), so f is reused from P1 after a
      field-by-field config-equality assertion.
  P4a E2/E3 eval-only on fold 2 (+ E3 beliefs, folds 1-3) with corrected checkpoints.
  P5  main grid: 3 trainers x {none, uncertainty_aware} x folds 1-3 x seeds 1-5,
      300k timesteps, canonical overrides, no ablations.
  P4b E2/E3 re-run including folds 1/3 once the grid exists.
  P6  NOT RUN: fold-4 re-evaluation awaits explicit approval.
  P2' (--include-p2prime; replaces held P2) oracle training where the signal is
      expressible: NO smoothing, band 0.10 + 5-day cadence + 2.0 cap + corrected
      costs retained, everything else canonical. canonical_nosmooth x 20,
      oracle100_5d_nosmooth x 20, oracle60_5d_nosmooth x 5 (NON-DEPLOYABLE).
      Matched seeds across arms; the torch RNG state at network creation is
      fingerprinted per run, and canonical_nosmooth's init is checked
      tensor-equal to P1's saved untrained init. With --wait-for-pid the process
      first waits for the running batch to exit (queued, not interleaved).

Fold 4 never loaded: fail-closed guard (r2_common) at import.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

os.environ["R2_DIR"] = os.environ.get("R2C_ROOT", "./results/r2/corrected")   # R2C_ROOT: smoke tests only
sys.path.insert(0, str(Path(__file__).parent))

import r2_common as rc  # guard first; R2_DIR -> corrected tree

import pandas as pd
import torch

import main as main_module
import run_hardening as rh
import run_paper_experiments as rpe
import run_r2 as R   # helpers: instrumentation, init capture, ladder configs, E5 runner
import trainer as trainer_module
from r2_oracle import build_oracle5_slices

ROOT = rc.R2_DIR
HARD = ROOT / "hardening"
GRID = ROOT / "grid"
BATCH = ROOT / "batch"
SEEDS20, SEEDS5 = list(range(1, 21)), list(range(1, 6))
rh.CANONICAL_REUSE_UPTO = 0     # P1: everything trained fresh under corrected accounting
rh.ORACLE60_REUSE_UPTO = 0


P2P_ARMS = {"canonical_nosmooth": (None, SEEDS20), "oracle100_5d_nosmooth": (1.0, SEEDS20),
            "oracle60_5d_nosmooth": (0.6, SEEDS5)}


def combo_list(include_p2, include_p2prime=False):
    c = [("p1", "untrained", s) for s in SEEDS20]
    c += [("p1", "canonical", s) for s in SEEDS20]
    c += [("p1", "oracle60", s) for s in SEEDS20]
    c += [("p1", "oracle75", s) for s in SEEDS20]
    if include_p2:
        c += [("p2", arm, s) for arm in R.E5_ARMS for s in SEEDS20]
    c += [("p3", "f", s) for s in SEEDS5]
    c += [("p3", k, s) for k in "abcdegh" for s in SEEDS5]
    c += [("p4a", "eval", 0)]
    c += [("p5", "grid", s) for s in SEEDS5]
    c += [("p4b", "eval", 0)]
    if include_p2prime:
        c += [("p2p", arm, s) for arm, (_, seeds) in P2P_ARMS.items() for s in seeds]
    return c


TRAINING_RUNS = {"p1": {"canonical": 1, "oracle60": 1, "oracle75": 1}, "p2": None, "p3": None, "p5": 18}


def n_training(c):
    exp, arm, _ = c
    if exp == "p1":
        return TRAINING_RUNS["p1"].get(arm, 0)
    if exp in ("p2", "p2p"):
        return 1
    if exp == "p3":
        return 0 if arm == "f" else 1
    if exp == "p5":
        return 18   # 3 trainers x 2 shields x 3 folds per seed
    return 0


def marker(exp, arm, seed):
    if exp == "p1":
        return HARD / arm / f"seed_{seed}" / "eval_result.json"
    if exp == "p2":
        return ROOT / "e5" / arm / f"seed_{seed}" / "eval_result.json"
    if exp == "p2p":
        return ROOT / "p2prime" / arm / f"seed_{seed}" / "eval_result.json"
    if exp == "p3":
        return ROOT / "e6" / arm / f"seed_{seed}" / "eval_result.json"
    if exp == "p5":
        return BATCH / f"p5_seed{seed}.done"      # written only when all 18 arms have eval_result.json
    return BATCH / f"{exp}.done"


# ----------------------------------------------------------------------------- P1

def p1_canonical(seed):
    config = rh._base_config(seed)
    slices = rh._load_fold2_slices(config)
    ref, _ = R.untrained_reference(seed, config, slices)    # calibrates on train, then seed -> network
    ref_sd = R._snapshot(ref)
    init_dir = HARD / "init"
    init_dir.mkdir(parents=True, exist_ok=True)
    torch.save(ref_sd, init_dir / f"untrained_init_seed{seed}.pt")
    store, verdict = {}, {}

    def on_trainer(tr):
        sd = R._snapshot(tr.actor_critic)
        torch.save(sd, init_dir / f"trainer_init_seed{seed}.pt")
        eq, bad = R.compare_sd(ref_sd, sd)
        verdict.update({'init_equal': eq, 'mismatched_tensors': bad, 'n_tensors': len(sd)})
        if not eq:
            raise R.InitMismatch(f"seed {seed}: {len(bad)} tensors differ before training")
        R._instrument(tr, store)

    try:
        row = R._with_trainer_hook(on_trainer, lambda: rh._train_and_eval(
            seed, config, slices, f"canonical_seed{seed}", HARD / "canonical_training"))
        status = "verified equal to untrained init before training"
    except R.InitMismatch as e:
        row, status = {}, f"UNMATCHED: {e} (not trained; excluded from paired analysis)"
    R._save_instrumentation(HARD / "canonical_training" / f"canonical_seed{seed}" / "binding_instrumentation.json", store)
    ckpt = HARD / "canonical_training" / f"canonical_seed{seed}" / "final_model.pt"
    row.update({'arm': 'canonical', 'seed': seed, 'pairing_status': status, **verdict,
                'trained_checkpoint': str(ckpt) if ckpt.exists() else None,
                'trained_checkpoint_sha256': rc.file_sha256(ckpt) if ckpt.exists() else None})
    return row


def p1_oracle(arm, seed):
    acc = {"oracle60": 0.6, "oracle75": 0.75}[arm]
    store = {}
    row = R._with_trainer_hook(lambda tr: R._instrument(tr, store), lambda: rh.process_oracle(
        HARD, arm, acc, seed, 0, ""))
    R._save_instrumentation(HARD / f"{arm}_training" / f"{arm}_seed{seed}" / "binding_instrumentation.json", store)
    ckpt = HARD / f"{arm}_training" / f"{arm}_seed{seed}" / "final_model.pt"
    row.update({'trained_checkpoint': str(ckpt), 'trained_checkpoint_sha256': rc.file_sha256(ckpt),
                'non_deployable': True})
    return row


# ----------------------------------------------------------------------------- P2'

def nosmooth_config(seed):
    cfg = rh._base_config(seed)
    cfg.environment.action_smoothing_beta = None
    diff = R.config_diff(cfg, rh._base_config(seed))
    assert set(diff) == {"environment.action_smoothing_beta"}, diff   # only smoothing removed
    return cfg


def p2prime(arm, seed, out_root=None):
    out_root = out_root or (ROOT / "p2prime")
    acc, _ = P2P_ARMS[arm]
    config = nosmooth_config(seed)
    slices = build_oracle5_slices(config, acc) if acc is not None else rh._load_fold2_slices(config)
    store, info = {}, {}
    orig_cac = trainer_module.create_actor_critic

    def fingerprinted_cac(*a, **k):
        info['rng_sha_at_init'] = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
        return orig_cac(*a, **k)

    def on_trainer(tr):
        sd = R._snapshot(tr.actor_critic)
        info['obs_dim'] = int(tr.obs_dim)
        ref = HARD / "init" / f"untrained_init_seed{seed}.pt"
        if acc is None and ref.exists():
            eq, bad = R.compare_sd(torch.load(ref), sd)
            info.update({'init_equal_to_p1_untrained': eq, 'mismatched_tensors': bad})
        R._instrument(tr, store)

    trainer_module.create_actor_critic = fingerprinted_cac
    try:
        row = R._with_trainer_hook(on_trainer, lambda: rh._train_and_eval(
            seed, config, slices, f"{arm}_seed{seed}", out_root / f"{arm}_training"))
    finally:
        trainer_module.create_actor_critic = orig_cac
    R._save_instrumentation(out_root / f"{arm}_training" / f"{arm}_seed{seed}" / "binding_instrumentation.json", store)
    ckpt = out_root / f"{arm}_training" / f"{arm}_seed{seed}" / "final_model.pt"
    row.update({'arm': arm, 'seed': seed, 'accuracy': acc, 'smoothing': None, 'band': 0.10, 'cadence': 5,
                'non_deployable': acc is not None, **info,
                'trained_checkpoint': str(ckpt), 'trained_checkpoint_sha256': rc.file_sha256(ckpt)})
    return row


# ----------------------------------------------------------------------------- P3

def p3(key, seed):
    if key != "f":
        return R.run_e6(key, seed)     # writes under ROOT/e6 (R2_DIR is the corrected tree)
    config = R.ladder_config(seed, "f")
    diff = R.config_diff(config, rh._base_config(seed))
    assert not diff, f"ladder f != canonical config: {diff}"
    src = marker("p1", "canonical", seed)
    if not src.exists():
        raise RuntimeError(f"P1 canonical seed {seed} not complete yet")
    row = json.loads(src.read_text())
    if not row.get('init_equal', False) or 'return' not in row:
        raise RuntimeError(f"P1 canonical seed {seed} unusable for reuse: {row.get('pairing_status')}")
    keep = {k: row[k] for k in ('return', 'gross_price_pnl', 'txn_cost', 'turnover', 'max_dd', 'sharpe',
                                'dd_violation_rate', 'trained_checkpoint', 'trained_checkpoint_sha256') if k in row}
    keep.update({'config': 'f', 'seed': seed, 'description': R.LADDER['f'][0], 'reused_from': str(src),
                 'note': 'identical config, seed and code path as P1 canonical -> reused, not retrained'})
    return keep


# ----------------------------------------------------------------------------- P4 / P5

def p4():
    env = dict(os.environ, R2_DIR=str(ROOT))
    out = {}
    for script in ("r2_gated_same_policy.py", "r2_belief_validation.py"):
        t0 = time.time()
        r = subprocess.run([sys.executable, script], env=env, cwd=str(Path(__file__).parent),
                           capture_output=True, text=True)
        print(r.stdout[-3000:], r.stderr[-3000:], flush=True)
        if r.returncode != 0:
            raise RuntimeError(f"{script} failed (exit {r.returncode})")
        out[script] = round(time.time() - t0, 1)
    return {'scripts_runtime_seconds': out}


def p5(seed):
    results = main_module.run_calendar_walk_forward_experiment(
        tickers=None, windows=None, shield_arms=None, trainer_arms=None, total_timesteps=300000,
        use_synthetic=False, output_dir=str(GRID / f"seed_{seed}"), seed=seed, n_eval_episodes=5,
        config_overrides=rpe.CANONICAL_FOLD2_CONFIG_OVERRIDES, resume=True)
    rows = rpe._extract_rows(results, seed)
    missing = [str(p) for f in (1, 2, 3) for sh in ("none", "uncertainty_aware")
               for tr in ("vanilla_ppo", "reward_shaping", "ppo_lagrangian")
               if not (GRID / f"seed_{seed}" / f"fold_{f}" / sh / tr / "eval_result.json").exists()
               for p in [f"fold_{f}/{sh}/{tr}"]]
    if missing:
        raise RuntimeError(f"grid seed {seed}: arms without eval_result.json (relaunch resumes them): {missing}")
    return {'seed': seed, 'n_rows': len(rows),
            'note': 'single seed_everything per seed, arms sequential (as the original grid); a mid-seed '
                    'restart changes the RNG stream seen by later arms -- see run.log for restarts'}


def rewrite_tables():
    rows = [json.loads(p.read_text()) for arm in ("untrained", "canonical", "oracle60", "oracle75")
            for p in sorted((HARD / arm).glob("seed_*/eval_result.json"))]
    if rows:
        rh._rewrite_csv(HARD, rows)
        rc.write_manifest("corrected_hardening", {'experiment': 'P1 hardening (corrected accounting)',
                                                  'rows': rows, 'fold4_loaded': False})
    e6 = [json.loads(p.read_text()) for p in sorted((ROOT / "e6").glob("*/seed_*/eval_result.json"))]
    if e6:
        pd.DataFrame(e6).drop(columns=['overrides'], errors='ignore').to_csv(ROOT / "ladder_seeded.csv", index=False)
        rc.write_manifest("corrected_ladder", {'experiment': 'P3 ladder', 'rows': e6, 'fold4_loaded': False})
    e5 = [json.loads(p.read_text()) for p in sorted((ROOT / "e5").glob("*/seed_*/eval_result.json"))]
    if e5:
        pd.DataFrame(e5).to_csv(ROOT / "e5" / "e5_results.csv", index=False)
    pp = [json.loads(p.read_text()) for p in sorted((ROOT / "p2prime").glob("*/seed_*/eval_result.json"))]
    if pp:
        pd.DataFrame(pp).drop(columns=['mismatched_tensors'], errors='ignore').to_csv(
            ROOT / "p2prime" / "p2prime_results.csv", index=False)
        rc.write_manifest("corrected_p2prime", {'experiment': "P2' oracle training, no smoothing (oracle arms "
                                                "NON-DEPLOYABLE)", 'rows': pp, 'fold4_loaded': False})
        rc.write_manifest("corrected_e5", {'experiment': 'P2 E5 oracles (NON-DEPLOYABLE)', 'rows': e5,
                                           'fold4_loaded': False})
    grid_rows = []
    for s in SEEDS5:
        wf = GRID / f"seed_{s}" / "walk_forward_results.json"
        if wf.exists():
            grid_rows += rpe._extract_rows(json.loads(wf.read_text()), s)
    if grid_rows:
        pd.DataFrame(grid_rows).to_csv(GRID / "results.csv", index=False)


def run_combo(c):
    exp, arm, seed = c
    if exp == "p1":
        if arm == "untrained":
            return rh.process_untrained(HARD, seed)
        if arm == "canonical":
            return p1_canonical(seed)
        return p1_oracle(arm, seed)
    if exp == "p2":
        return R.run_e5(arm, seed)
    if exp == "p2p":
        return p2prime(arm, seed)
    if exp == "p3":
        return p3(arm, seed)
    if exp == "p5":
        return p5(seed)
    return p4()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-p2", action="store_true", help="also train the E5 oracle arms (held by default)")
    ap.add_argument("--include-p2prime", action="store_true", help="append P2' (no-smoothing oracle training)")
    ap.add_argument("--wait-for-pid", type=int, default=None, help="queue: start only after this PID exits")
    args = ap.parse_args()
    BATCH.mkdir(parents=True, exist_ok=True)
    if args.wait_for_pid:
        import check_r2c_progress as chk
        q = BATCH / "queued.json"
        while chk.pid_alive(args.wait_for_pid):
            q.write_text(json.dumps({'pid': os.getpid(), 'waiting_for_pid': args.wait_for_pid, 'since': R.now(),
                                     'include_p2prime': args.include_p2prime}, indent=2), encoding='utf-8')
            time.sleep(60)
        q.unlink(missing_ok=True)
        (BATCH / "r2c.pid").write_text(str(os.getpid()))
    log = open(BATCH / "run.log", "a", buffering=1, encoding="utf-8")
    sys.stdout = sys.stderr = log
    combos = combo_list(args.include_p2, args.include_p2prime)
    state = {'status': 'running', 'pid': os.getpid(), 'started': R.now(), 'include_p2': args.include_p2,
             'include_p2prime': args.include_p2prime,
             'total_combos': len(combos), 'total_training_runs': sum(n_training(c) for c in combos),
             'current': None, 'failed': [], 'last_update': R.now()}

    def heartbeat():
        while state['status'] == 'running':
            state['completed'] = sum(marker(*c).exists() for c in combos)
            state['last_update'] = R.now()
            (BATCH / "heartbeat.json").write_text(json.dumps(state, indent=2), encoding='utf-8')
            time.sleep(30)

    threading.Thread(target=heartbeat, daemon=True).start()
    print(f"\n=== corrected batch start {R.now()}: {len(combos)} combos, {state['total_training_runs']} "
          f"training runs, include_p2={args.include_p2} ===", flush=True)
    try:
        for c in combos:
            m = marker(*c)
            if m.exists():
                continue
            state['current'] = {'combo': list(c), 'started': R.now()}
            print(f"\n--- {c} start {R.now()}", flush=True)
            t0 = time.time()
            try:
                row = run_combo(c)
            except Exception:
                tb = traceback.format_exc()
                print(tb, flush=True)
                state['failed'].append({'combo': list(c), 'error': tb.strip().splitlines()[-1], 'at': R.now()})
                continue
            if c[0] in ("p2", "p2p", "p3") or (c[0] == "p1"):
                row['runtime_seconds'] = round(time.time() - t0, 1)
                row['completed_utc'] = R.now()
                if c[0] != "p1" or c[1] == "canonical":     # run_hardening writes its own markers for the others
                    m.parent.mkdir(parents=True, exist_ok=True)
                    m.write_text(json.dumps(row, indent=2, default=str), encoding='utf-8')
            elif c[0] in ("p4a", "p4b", "p5"):
                m.write_text(json.dumps({**row, 'completed_utc': R.now()}, indent=2), encoding='utf-8')
            (BATCH / "runtimes.jsonl").open("a").write(json.dumps(
                {'combo': list(c), 'seconds': round(time.time() - t0, 1), 'training_runs': n_training(c),
                 'at': R.now()}) + "\n")
            print(f"--- {c} done in {time.time() - t0:.0f}s", flush=True)
            rewrite_tables()
        state['status'] = 'finished' if not state['failed'] else 'finished_with_failures'
    except BaseException:
        state['status'] = 'crashed'
        print(traceback.format_exc(), flush=True)
        raise
    finally:
        state['completed'] = sum(marker(*c).exists() for c in combos)
        state['current'], state['last_update'] = None, R.now()
        (BATCH / "heartbeat.json").write_text(json.dumps(state, indent=2), encoding='utf-8')
        print(f"=== corrected batch end {R.now()} status={state['status']} ===", flush=True)


if __name__ == "__main__":
    main()
