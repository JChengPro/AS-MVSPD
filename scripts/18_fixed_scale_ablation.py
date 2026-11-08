"""Prespecified paired comparison of adaptive and fixed top-k tests."""
from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import norm

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from stablejoint import analyze_multiview, generate_signs
from stablejoint.core import self_normalized_feature_scores, _top_k_means, _tail_probabilities
from stablejoint.benchmark import simulate_paired_blocks

OUT = PROJECT / "results/method_validation/fixed_scale_ablation"
GRID = (1, 2, 5, 10, 20)
METHODS = ["AS-MVSPD"] + [f"Fixed top-{k}" for k in GRID]
GROUP = ["scenario", "n_subjects", "n_signal", "effect"]


def compare(blocks, signs):
    view_p, selected = [], []
    for block in blocks.values():
        scores = self_normalized_feature_scores(block.to_numpy(), signs, False)
        means, grid = _top_k_means(scores, GRID)
        scale_p = np.column_stack([_tail_probabilities(means[:, j]) for j in range(len(grid))])
        evidence = -np.log(np.maximum(scale_p, 1 / len(signs)))
        adaptive_p = _tail_probabilities(evidence.max(axis=1))[0]
        view_p.append(np.r_[adaptive_p, scale_p[0]])
        selected.append(grid[int(evidence[0].argmax())])
    return np.max(view_p, axis=0), np.asarray(view_p), selected


def wilson(s, n):
    z = norm.ppf(0.975)
    p = s / n
    den = 1 + z*z/n
    mid = (p + z*z/(2*n))/den
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return max(0., mid-half), min(1., mid+half)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    designs = [("null", 0, 0.0)] + [
        (scenario, signals, effect)
        for scenario in ("stable_sparse", "one_block")
        for signals in (1, 5, 20) for effect in (1.1, 1.4)
    ]
    signs, full = generate_signs(12, max_random=255, seed=2026091501)
    packed = signs.astype(np.int8)
    assert np.all(packed[0] == 1) and len(np.unique(packed, axis=0)) == 256
    members = {row.tobytes() for row in packed}
    assert all(row.tobytes() in members for s in packed for row in packed*s)
    np.save(OUT / "sign_matrix.npy", packed)
    config = dict(n_subjects=12, n_views=2, n_features_per_view=200,
                  loading_scale=0.35, noise="normal", k_grid=GRID,
                  replicates=500, alpha=0.05, signs_count=len(signs),
                  full_enumeration=bool(full), sign_seed=2026091501,
                  data_master_seed=2026091502, designs=designs,
                  sign_matrix_sha256=hashlib.sha256(packed.tobytes()).hexdigest(),
                  hash_encoding="C-order int8 matrix bytes; shape (256, 12)",
                  numpy_version=np.__version__,
                  source_sha256={str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [Path(__file__), PROJECT/'stablejoint/core.py', PROJECT/'stablejoint/benchmark.py']})
    (OUT / "design.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    # Check the shared-score implementation against the production inference API.
    checks = []
    for scenario in ("null", "stable_sparse", "one_block"):
        blocks, _ = simulate_paired_blocks(12, 200, 5, scenario, np.random.default_rng(718))
        joint, _, _ = compare(blocks, signs)
        for i, k in enumerate([None, *GRID]):
            expected = analyze_multiview(blocks, signs=signs, k_grid=GRID if k is None else (k,), influence_constraint=False).joint_pvalue
            assert joint[i] == expected, (scenario, k, joint[i], expected)
            checks.append(dict(scenario=scenario, method=METHODS[i], pvalue=expected, passed=True))
    pd.DataFrame(checks).to_csv(OUT / "implementation_checks.csv", index=False)
    rows = []
    start = time.monotonic()
    for design_id, (scenario, signals, effect) in enumerate(designs):
        for rep in range(500):
            seed = int(np.random.SeedSequence([2026091502, design_id, rep]).generate_state(1)[0])
            blocks, _ = simulate_paired_blocks(12, 200, signals, scenario, np.random.default_rng(seed), effect=effect, loading_scale=0.35)
            joint, views, selected = compare(blocks, signs)
            for i, method in enumerate(METHODS):
                rows.append(dict(scenario=scenario, n_subjects=12, n_signal=signals, effect=effect,
                                 replicate=rep, data_seed=seed, method=method, pvalue=joint[i],
                                 view1_p=views[0,i], view2_p=views[1,i],
                                 selected_k_view1=selected[0] if i == 0 else GRID[i-1],
                                 selected_k_view2=selected[1] if i == 0 else GRID[i-1],
                                 rejected=bool(joint[i] <= 0.05)))
        pd.DataFrame(rows).to_csv(OUT / "replicates.csv", index=False)
        print(f"{design_id+1}/{len(designs)} {scenario} signals={signals} effect={effect} elapsed={time.monotonic()-start:.1f}s", flush=True)
    data = pd.DataFrame(rows)
    summary = []
    for keys, part in data.groupby(GROUP + ["method"], sort=False):
        s, n = int(part.rejected.sum()), len(part)
        assert n == 500 and part.replicate.nunique() == 500
        low, high = wilson(s, n)
        summary.append(dict(zip(GROUP + ["method"], keys), successes=s, replicates=n, rate=s/n, ci_low=low, ci_high=high))
    pd.DataFrame(summary).to_csv(OUT / "summary.csv", index=False)
    # Paired outcomes quantify Monte Carlo uncertainty of each power difference.
    differences = []
    for keys, part in data.groupby(GROUP, sort=False):
        wide = part.pivot(index="replicate", columns="method", values="rejected").astype(int)
        for method in METHODS[1:]:
            diff = wide["AS-MVSPD"] - wide[method]
            differences.append(dict(zip(GROUP, keys), comparison=f"AS-MVSPD minus {method}",
                                    difference=diff.mean(), paired_mc_se=diff.std(ddof=1)/np.sqrt(len(diff)),
                                    adaptive_only=int((diff == 1).sum()), fixed_only=int((diff == -1).sum()), replicates=len(diff)))
    pd.DataFrame(differences).to_csv(OUT / "paired_comparisons.csv", index=False)
    assert np.allclose(data.pvalue, data[["view1_p", "view2_p"]].max(axis=1))
    assert np.allclose(data.pvalue*256, np.round(data.pvalue*256))
    config.update(completed=True, elapsed_seconds=time.monotonic()-start,
                  total_datasets=6500, total_method_results=len(data),
                  output_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.csv')})
    (OUT / "run_metadata.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
