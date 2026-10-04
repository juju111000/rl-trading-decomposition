#!/usr/bin/env python3
"""
FINAL EVIDENCE SESSION, Part A step 2 (training half). Canonical
ppo_lagrangian, shield in {none, uncertainty_aware}, 5 seeds each, the
frozen canonical execution-model config -- trained on 2021-01-01..2023-12-31,
tested on 2024-01-01..2025-12-31 (fold 4). No strategy changes: this is
run_paper_experiments.py's own grid machinery, unmodified, pointed at a new
window. resume=True so it can be restarted cheaply if interrupted.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from run_paper_experiments import run_paper_experiments, CANONICAL_FOLD2_CONFIG_OVERRIDES

FOLD_4_WINDOW = {'train': ('2021-01-01', '2023-12-31'), 'test': ('2024-01-01', '2025-12-31')}

if __name__ == "__main__":
    run_paper_experiments(
        windows=[FOLD_4_WINDOW],
        shield_arms=['none', 'uncertainty_aware'],
        trainer_arms=['ppo_lagrangian'],
        seeds=[1, 2, 3, 4, 5],
        total_timesteps=300000,
        output_dir="./results/fold4/grid",
        include_shield_ablation=False,
        include_inactivity_ablation=False,
        config_overrides=CANONICAL_FOLD2_CONFIG_OVERRIDES,
        resume=True,
    )
    print("\nFOLD4_GRID_DONE")
