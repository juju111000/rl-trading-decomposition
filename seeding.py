"""
Central seeding utility for reproducible experiments.

Call seed_everything(seed) once, as early as possible, at the top of any
entry-point script (main.py, debug_main.py, run_comparison.py,
evaluate_saved.py) -- before any data loading, model construction, or
environment creation. It seeds every global RNG this codebase draws from:

  - python's `random` module
  - numpy's global RNG (this is also what trainer.py's minibatch shuffle,
    `np.random.shuffle`, draws from -- no separate handling needed there)
  - torch's CPU and CUDA RNGs, plus deterministic cudnn flags

Note: setting PYTHONHASHSEED here does NOT retroactively fix hash
randomization for the *current* process -- Python reads that env var once,
at interpreter startup, before any user code runs. It's set here only so
that a subprocess this script spawns (e.g. run_improved_experiments.py
launching `python main.py ...`) inherits a fixed hash seed. Ticker/column
ordering in this codebase no longer depends on hash-dependent set()
iteration order (features.py, regime_model.py use insertion-order-preserving
dict.fromkeys() instead), so this line is defense-in-depth, not the
mechanism that makes ordering deterministic.
"""
import os
import random

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed every global RNG this codebase draws from."""
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
