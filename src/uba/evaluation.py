"""Metrics, plots and the business-cost metric for insider-threat detection."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, fbeta_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)


@dataclass
class CostAssumptions:
    """Stated assumptions behind the business metric (edit to match the customer).

    * analyst_minutes_per_alert: time for a SOC/insider-risk analyst to triage one
      flagged user-day (open the user's logs, check context, close or escalate).
    * analyst_cost_per_hour: fully loaded analyst cost (USD).
    * cost_per_missed_malicious_day: cost of one malicious user-day going unnoticed.
      Derived as (average cost of a malicious-insider incident) / (average days to
      contain), using the Ponemon "Cost of Insider Threats" 2022 report figures
      (~USD 648k per malicious-insider incident, ~85 days to contain) -> ~USD 7.6k/day.
      Verify against the latest report before quoting it.
    """
    analyst_minutes_per_alert: float = 20.0
    analyst_cost_per_hour: float = 45.0
    cost_per_missed_malicious_day: float = 7_600.0


def classification_metrics(y_true, y_pred, y_score=None) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    m = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "alerts": int(tp + fp),
    }
    if y_score is not None:
        m["pr_auc"] = average_precision_score(y_true, y_score)
        m["roc_auc"] = roc_auc_score(y_true, y_score)
    return m


def business_metrics(m: dict, n_days: int, n_users: int, cost: CostAssumptions = CostAssumptions()) -> dict:
    hours = m["alerts"] * cost.analyst_minutes_per_alert / 60
    review_cost = hours * cost.analyst_cost_per_hour
    missed_cost = m["fn"] * cost.cost_per_missed_malicious_day
    return {
        "analyst_hours": hours,
        "alerts_per_day": m["alerts"] / max(n_days, 1),
        "alerts_per_1000_users_per_day": m["alerts"] / max(n_days, 1) / max(n_users, 1) * 1000,
        "review_cost_usd": review_cost,
        "missed_insider_cost_usd": missed_cost,
        "total_cost_usd": review_cost + missed_cost,
    }


def insider_level_detection(test: pd.DataFrame, y_pred) -> pd.DataFrame:
    """Per insider in the test period: was at least one malicious day flagged, and how fast?"""
    d = test[["user", "day", "label", "scenario"]].copy()
    d["pred"] = np.asarray(y_pred)
    rows = []
    for user, g in d[d["label"] == 1].groupby("user"):
        g = g.sort_values("day")
        hit = g[g["pred"] == 1]
        rows.append({
            "user": user, "scenario": int(g["scenario"].iloc[0]),
            "malicious_days_in_test": len(g), "flagged_days": len(hit),
            "detected": len(hit) > 0,
            "days_to_first_flag": (hit["day"].iloc[0] - g["day"].iloc[0]).days if len(hit) else None,
        })
    return pd.DataFrame(rows)


def results_table(results: dict[str, dict]) -> pd.DataFrame:
    cols = ["precision", "recall", "f1", "f2", "pr_auc", "roc_auc", "accuracy", "tp", "fp", "fn", "tn", "alerts"]
    df = pd.DataFrame(results).T
    return df[[c for c in cols if c in df.columns]]


def plot_confusion(y_true, y_pred, title: str, ax=None):
    ax = ax or plt.gca()
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=["benign", "malicious"],
                                            cmap="Blues", colorbar=False, ax=ax, values_format="d")
    ax.set_title(title)
    return ax


def plot_pr_curves(y_true, scores: dict[str, np.ndarray], ax=None, marker_points: dict | None = None):
    ax = ax or plt.gca()
    for name, s in scores.items():
        p, r, _ = precision_recall_curve(y_true, s)
        ax.plot(r, p, label=f"{name} (PR-AUC={average_precision_score(y_true, s):.3f})")
    for name, (rec, prec) in (marker_points or {}).items():
        ax.scatter([rec], [prec], s=60, zorder=5, label=name)
    ax.axhline(np.mean(y_true), ls="--", c="grey", lw=1, label=f"chance = {np.mean(y_true):.4f}")
    ax.set_xlabel("Recall (share of malicious user-days caught)")
    ax.set_ylabel("Precision (share of alerts that are real)")
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.01)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title("Precision-recall curve (test period)")
    return ax


def cost_assumptions_dict(cost: CostAssumptions = CostAssumptions()) -> dict:
    return asdict(cost)
