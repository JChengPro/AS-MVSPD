#!/usr/bin/env python3
"""Benchmark AS-MVSPD wall-clock time over representative small-n settings."""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from stablejoint import analyze_multiview, generate_signs


OUTPUT = (
    PROJECT
    / "results"
    / "publication_figures"
    / "source_data"
    / "runtime_scalability.csv"
)
SEED = 20260803
REPETITIONS = 5


def main() -> None:
    rows: list[dict[str, float | int | bool]] = []
    for n_subjects in (8, 12, 20):
        signs, complete_group = generate_signs(
            n_subjects, max_random=4095, seed=SEED
        )
        for features_per_view in (100, 500, 1000):
            rng = np.random.default_rng(
                SEED + 1000 * n_subjects + features_per_view
            )
            blocks = {
                f"View{view + 1}": pd.DataFrame(
                    rng.normal(size=(n_subjects, features_per_view))
                )
                for view in range(3)
            }
            analyze_multiview(blocks, signs=signs)
            timings = []
            for _ in range(REPETITIONS):
                start = perf_counter()
                analyze_multiview(blocks, signs=signs)
                timings.append(perf_counter() - start)
            rows.append(
                {
                    "n_subjects": n_subjects,
                    "views": 3,
                    "features_per_view": features_per_view,
                    "sign_patterns": len(signs),
                    "complete_sign_group": complete_group,
                    "repetitions": REPETITIONS,
                    "median_seconds": float(np.median(timings)),
                    "q1_seconds": float(np.quantile(timings, 0.25)),
                    "q3_seconds": float(np.quantile(timings, 0.75)),
                }
            )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
