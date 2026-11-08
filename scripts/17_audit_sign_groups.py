#!/usr/bin/env python3
"""Record seeds and SHA-256 hashes for prespecified sign-flip groups."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from stablejoint import generate_signs  # noqa: E402


def matrix_sha256(signs: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(signs).tobytes()).hexdigest()


def main() -> None:
    config = json.loads((PROJECT / "config" / "experiment.json").read_text(encoding="utf-8"))
    master_seed = int(config["seed"])
    validation_seed = master_seed + 9000
    contexts = [
        ("core_simulation_n8", 8, int(config["max_random_sign_flips"]), master_seed + 8),
        ("core_simulation_n12", 12, 255, master_seed + 12),
        ("core_simulation_n20", 20, 255, master_seed + 20),
        ("three_view_validation_n8", 8, 4095, master_seed),
        ("structured_correlation_n8", 8, 4095, master_seed + 11000 + 8),
        ("structured_correlation_n12", 12, 255, master_seed + 11000 + 12),
        ("asymmetry_joint_n8", 8, 4095, master_seed + 13100 + 8),
        ("asymmetry_joint_n12", 12, 255, master_seed + 13100 + 12),
        ("asymmetry_maxT_n8", 8, 4095, master_seed),
        ("published_baseline_n8", 8, 255, validation_seed + 400 + 8),
        ("published_baseline_n12", 12, 4095, validation_seed + 400 + 12),
        ("local_application_n8", 8, 4095, master_seed),
        ("public_haddad_n8", 8, int(config["max_random_sign_flips"]), master_seed + 8),
        ("public_mars_n70", 70, 255, master_seed + 70),
        ("public_he_n57", 57, 255, master_seed + 57),
        ("fixed_scale_n12", 12, 255, 2026091501),
        ("he_larger_subgroup_n57", 57, 4095, master_seed + 57),
        ("he_figure_loo_n56", 56, 255, master_seed + 7000 + 56),
    ]
    contexts.extend((f"he_audit_seed_{i}",57,255,master_seed+57+i) for i in range(1,21))
    contexts.extend((f"he_figure_seed_{i}",57,255,master_seed+7100+i) for i in range(20))
    rows = []
    for context, n_subjects, max_random, seed in contexts:
        signs, exact = generate_signs(n_subjects, max_random=max_random, seed=seed)
        rows.append(
            {
                "context": context,
                "n_subjects": n_subjects,
                "max_random": max_random,
                "seed": seed,
                "sign_patterns": int(len(signs)),
                "complete_full_group": bool(exact),
                "matrix_shape": list(signs.shape),
                "matrix_dtype": str(signs.dtype),
                "matrix_sha256": matrix_sha256(signs),
            }
        )

    output = PROJECT / "results" / "reproducibility"
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "master_seed": master_seed,
        "hash_definition": "SHA-256 of contiguous float64 sign-matrix bytes in row-major order",
        "sign_generation": "complete full group or complete GF(2) subgroup",
        "groups": rows,
    }
    (output / "sign_group_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
