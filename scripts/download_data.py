#!/usr/bin/env python3
"""
Re-fetch the raw daily OHLCV price data used by the paper from Yahoo Finance
(via yfinance) and cache it under ./data_cache/, exactly where the pipeline's
DataLoader looks for it. Raw price data is not redistributed in this repo.

Run from the repository root:

    python scripts/download_data.py            # fetch any ranges not yet cached
    python scripts/download_data.py --force    # re-download everything

The date ranges below are the buffered ranges the pipeline itself requests
(each fold's nominal train start minus main.FEATURE_WARMUP_BUFFER_DAYS = 40
calendar days), so the cache files written here are the ones the experiment
scripts will load. Any range not pre-fetched here is downloaded on demand by
the pipeline anyway; this script just does it up front, in one place.

Note: Yahoo Finance data can be revised after the fact, so a fresh download
may differ slightly from the data the committed results were computed on.
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import DataConfig
from data_loader import DataLoader

TICKERS = ["^GSPC", "^DJI", "000001.SS", "^GDAXI", "CL=F"]

# (start, end) ranges, both inclusive
RANGES = [
    ("2005-11-22", "2023-12-31"),  # folds 1-3 combined (2006-2023)
    ("2005-11-22", "2010-12-31"),  # fold 1 (train 2006-07, test 2008-10)
    ("2009-11-22", "2017-12-31"),  # fold 2 (train 2010-14, test 2015-17)
    ("2020-11-22", "2025-12-31"),  # fold 4 (train 2021-23, test 2024-25)
]


def main():
    parser = argparse.ArgumentParser(description="Re-download the paper's raw price data")
    parser.add_argument("--force", action="store_true", help="Bypass the cache and re-download")
    parser.add_argument("--cache-dir", type=str, default="./data_cache")
    args = parser.parse_args()

    for start, end in RANGES:
        print(f"=== {start} .. {end} ===")
        cfg = DataConfig(tickers=TICKERS, start_date=start, end_date=end, cache_dir=args.cache_dir)
        raw = DataLoader(cfg).download_data(force_download=args.force)
        for ticker, df in raw.items():
            print(f"  {ticker:10s} {len(df):5d} rows  {df.index.min().date()} .. {df.index.max().date()}")


if __name__ == "__main__":
    main()
