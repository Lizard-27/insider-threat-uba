"""Time-based split, leak-free preprocessing Pipeline, baselines and leakage checks."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .features import KEYS, build_feature_table, compute_aggregates

SEED = 42
TARGET = "label"
# identifiers / label-derived columns that must never be model inputs
NON_FEATURES = {"user", "day", "label", "scenario", "pc", "id", "insider", "date"}
CAT_FEATURES = ["role", "functional_unit"]


def feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    cat = [c for c in CAT_FEATURES if c in df.columns]
    num = [c for c in df.columns if c not in NON_FEATURES and c not in cat
           and pd.api.types.is_numeric_dtype(df[c])]
    return num, cat


def time_split(df: pd.DataFrame, test_frac: float = 0.30):
    """Chronological split: the last ``test_frac`` of calendar days form the test set.

    A time-based split mirrors deployment (train on the past, score the future)
    and prevents a model from learning an insider's behaviour on day t+1 and being
    tested on day t.
    """
    days = np.sort(df["day"].unique())
    cutoff = pd.Timestamp(days[int(len(days) * (1 - test_frac))])
    train = df[df["day"] < cutoff].reset_index(drop=True)
    test = df[df["day"] >= cutoff].reset_index(drop=True)
    return train, test, cutoff


def signed_log1p(x):
    """log1p that keeps the sign: tames heavy-tailed counts and handles negative deviations."""
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.log1p(np.abs(x))


def build_preprocessor(num_cols, cat_cols) -> ColumnTransformer:
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log1p", FunctionTransformer(signed_log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer([
        ("num", num_pipe, list(num_cols)),
        ("cat", OneHotEncoder(handle_unknown="ignore"), list(cat_cols)),
    ])


def build_baseline_pipeline(num_cols, cat_cols, class_weight="balanced", C: float = 1.0,
                            seed: int = SEED) -> Pipeline:
    """Baseline = Logistic Regression inside ONE Pipeline (impute -> log1p -> scale -> LR).

    ``class_weight='balanced'`` handles the extreme imbalance (<1% malicious
    user-days) by up-weighting the minority class in the loss, without creating
    synthetic insiders.
    """
    return Pipeline([
        ("prep", build_preprocessor(num_cols, cat_cols)),
        ("clf", LogisticRegression(class_weight=class_weight, C=C, max_iter=5000, random_state=seed)),
    ])


def rule_baseline(df: pd.DataFrame) -> np.ndarray:
    """A typical hand-written SIEM rule, used as a non-ML reference point.

    Alert if the user used removable media after hours, OR visited a leak or
    hacking-tool site.
    """
    return ((df["n_usb_after_hours"] > 0) | (df["n_http_leak"] > 0) | (df["n_http_hack"] > 0)).astype(int).values


# ------------------------------------------------------------------ leakage
def leakage_checks(train: pd.DataFrame, test: pd.DataFrame, num_cols, cat_cols,
                   fitted: Pipeline | None = None) -> pd.DataFrame:
    """Documented leakage checks. Every row must say PASS before results are trusted."""
    feats = list(num_cols) + list(cat_cols)
    res = []

    def add(name, ok, detail):
        res.append({"check": name, "result": "PASS" if ok else "FAIL", "detail": detail})

    add("Train strictly before test in time", train["day"].max() < test["day"].min(),
        f"train ends {train['day'].max().date()}, test starts {test['day'].min().date()}")
    overlap = len(pd.merge(train[KEYS], test[KEYS], on=KEYS))
    add("No user-day in both train and test", overlap == 0, f"{overlap} overlapping user-days")
    bad = sorted(set(feats) & NON_FEATURES)
    add("No identifier / label columns among features", not bad, f"forbidden present: {bad or 'none'}")
    corr = train[num_cols].corrwith(train[TARGET]).abs().max()
    add("No single feature is a near-copy of the label", corr < 0.95, f"max |corr(feature, label)| = {corr:.3f}")
    add("Both splits contain malicious user-days", train[TARGET].sum() > 0 and test[TARGET].sum() > 0,
        f"train {int(train[TARGET].sum())}, test {int(test[TARGET].sum())}")
    if fitted is not None:
        num_pipe = fitted.named_steps["prep"].named_transformers_["num"]
        Xtr = num_pipe.named_steps["log1p"].transform(num_pipe.named_steps["impute"].transform(train[num_cols]))
        ok = np.allclose(num_pipe.named_steps["scale"].mean_, Xtr.mean(axis=0))
        add("Scaler/imputer statistics fitted on TRAIN only", ok,
            "StandardScaler.mean_ equals the training-set mean (test never seen by fit)")
    return pd.DataFrame(res)


def prefix_invariance_check(build, cutoff: pd.Timestamp, num_cols) -> dict:
    """Recompute the whole feature table using ONLY logs before ``cutoff``.

    If any training-period feature changes, a feature is peeking at future data
    (e.g. a user-level mean computed over the full timeline). ``build`` is the
    :class:`uba.features.FeatureBuild` returned by ``build_from_dir``.
    """
    trunc = {k: v[v["date"] < cutoff] for k, v in build.logs.items()}
    aggs = compute_aggregates(trunc, build.ldap)
    http = build.aggs["http"]
    aggs["http"] = http[http.index.get_level_values("day") < cutoff]
    t2 = build_feature_table(aggs, build.ldap, build.labels)
    a = build.table[build.table["day"] < cutoff].sort_values(KEYS).reset_index(drop=True)
    b = t2.sort_values(KEYS).reset_index(drop=True)
    same_rows = len(a) == len(b) and (a[KEYS].values == b[KEYS].values).all()
    max_diff = float(np.nanmax(np.abs(a[num_cols].values - b[num_cols].values))) if same_rows else np.inf
    return {"check": "Prefix invariance (features use past data only)",
            "result": "PASS" if same_rows and max_diff < 1e-9 else "FAIL",
            "detail": f"{len(a)} train rows recomputed from truncated logs; max abs difference = {max_diff:.2e}"}
