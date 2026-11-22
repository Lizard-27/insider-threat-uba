"""Run the full Phase 1 pipeline from the command line (same steps as the notebook).

    python scripts/run_phase1.py                       # synthetic CERT-format data
    python scripts/run_phase1.py --data data/raw/r4.2 --answers data/raw/answers   # real CERT r4.2

Writes data/processed/user_day_features.csv and reports/phase1_metrics.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from uba.evaluation import (CostAssumptions, business_metrics, classification_metrics,  # noqa: E402
                            insider_level_detection, results_table)
from uba.features import build_from_dir  # noqa: E402
from uba.modeling import (SEED, TARGET, build_baseline_pipeline, feature_columns,  # noqa: E402
                          leakage_checks, prefix_invariance_check, rule_baseline, time_split)
from uba.synthetic_cert import generate_synthetic_cert  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="folder with CERT r4.2 csv files (default: synthetic)")
    ap.add_argument("--answers", default=None, help="CERT answers folder (default: <data>/answers)")
    ap.add_argument("--test-frac", type=float, default=0.30)
    args = ap.parse_args()

    if args.data is None:
        data_dir = generate_synthetic_cert(ROOT / "data/raw/synthetic_r4.2", seed=SEED)
    else:
        data_dir = Path(args.data)
    answers_dir = Path(args.answers) if args.answers else Path(data_dir) / "answers"

    build = build_from_dir(data_dir, answers_dir, save_to=ROOT / "data/processed/user_day_features.csv")
    table = build.table
    print(pd.DataFrame([r.as_dict() for r in build.reports]).to_string(index=False))
    print(f"\nUser-day table: {table.shape}, malicious rate = {table[TARGET].mean():.4%}")

    train, test, cutoff = time_split(table, args.test_frac)
    num, cat = feature_columns(table)
    pipe = build_baseline_pipeline(num, cat).fit(train[num + cat], train[TARGET])

    checks = leakage_checks(train, test, num, cat, fitted=pipe)
    checks = pd.concat([checks, pd.DataFrame([prefix_invariance_check(build, cutoff, num)])], ignore_index=True)
    print("\n" + checks.to_string(index=False))

    y = test[TARGET].values
    score = pipe.predict_proba(test[num + cat])[:, 1]
    pred = (score >= 0.5).astype(int)
    unweighted = build_baseline_pipeline(num, cat, class_weight=None).fit(train[num + cat], train[TARGET])
    uw_score = unweighted.predict_proba(test[num + cat])[:, 1]
    rule = rule_baseline(test)

    results = {
        "LogReg (balanced) @0.5": classification_metrics(y, pred, score),
        "LogReg (no weights) @0.5": classification_metrics(y, (uw_score >= 0.5).astype(int), uw_score),
        "SIEM rule": classification_metrics(y, rule),
    }
    print("\n" + results_table(results).round(3).to_string())
    n_days, n_users = test["day"].nunique(), test["user"].nunique()
    biz = {k: business_metrics(v, n_days, n_users) for k, v in results.items()}
    print("\n" + pd.DataFrame(biz).T.round(1).to_string())
    print("\n" + insider_level_detection(test, pred).to_string(index=False))

    out = ROOT / "reports/phase1_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "cutoff": str(cutoff.date()), "n_train": len(train), "n_test": len(test),
        "train_pos": int(train[TARGET].sum()), "test_pos": int(test[TARGET].sum()),
        "results": results, "business": biz, "cost_assumptions": CostAssumptions().__dict__,
        "leakage_checks": checks.to_dict(orient="records"),
    }, indent=2, default=float))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
