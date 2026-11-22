"""Build (and optionally execute) notebooks/Phase1_Insider_Threat_UBA.ipynb.

    python scripts/build_notebook.py            # write the notebook
    python scripts/build_notebook.py --execute  # write + run it, saving outputs
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "Phase1_Insider_Threat_UBA.ipynb"
cells = []


def md(s: str):
    cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s: str):
    cells.append(nbf.v4.new_code_cell(s.strip("\n")))


# =========================================================================== 0
md(r"""
# Insider Threat & User Behaviour Analytics from Logs — Phase 1

**CET433 Selected Topics in Cyber · Fall 2026 · Project 8**
Instructors: Dr. Mariam Labib, Eng. Maryam Adel

| Member | Phase 1 responsibility |
|---|---|
| **Mohamed Moamen** (team leader) | Problem & market case, privacy/ethics note, integration, talk |
| Ahmed Mohamed | Data acquisition, log schema, cleaning & EDA |
| Marwan Hassan | Per-user daily feature engineering & labelling |
| Ahmed Osama | Pipeline, leakage checks, baseline model & metrics |

**Phase 1 deliverables in this notebook:** market case · privacy & ethics note · dataset & licence · log schema ·
EDA · cleaning · per-user daily feature table · split-before-fit pipeline · leakage checks · class-imbalance handling ·
baseline with confusion matrix, precision, recall, F1 · business metric · Phase 2 plan & risks.

> Run top-to-bottom with **Runtime → Run all**. Defaults use a *synthetic* dataset in the exact CERT r4.2 format;
> set `USE_SYNTHETIC = False` in §3 to run on the real CERT r4.2 files.
""")

code(r"""
# 0. Setup: works in Google Colab and locally
import os, sys, subprocess

REPO_URL = "https://github.com/Lizard-27/insider-threat-uba.git"   # team GitHub repo

if not os.path.isdir("src/uba"):
    if os.path.isdir("../src/uba"):
        os.chdir("..")                                # opened from notebooks/
    elif "google.colab" in sys.modules:
        if not os.path.isdir("insider-threat-uba"):
            subprocess.run(["git", "clone", "-q", REPO_URL, "insider-threat-uba"], check=True)
        os.chdir("insider-threat-uba")
    else:
        raise RuntimeError("Run from the repository root or the notebooks/ folder.")
sys.path.insert(0, os.path.abspath("src"))
print("Working directory:", os.getcwd())
""")

code(r"""
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import set_config

from uba.synthetic_cert import generate_synthetic_cert
from uba.data_loading import load_insider_answers
from uba.features import build_from_dir, FEATURE_DESCRIPTIONS
from uba.modeling import (SEED, TARGET, time_split, feature_columns, build_baseline_pipeline,
                          rule_baseline, leakage_checks, prefix_invariance_check, signed_log1p)
from uba.evaluation import (CostAssumptions, classification_metrics, business_metrics, results_table,
                            insider_level_detection, plot_confusion, plot_pr_curves, cost_assumptions_dict)

warnings.filterwarnings("ignore", category=FutureWarning)
np.random.seed(SEED)                       # fixed seed everywhere (generator, models)
set_config(display="diagram")
sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.max_columns", 50, "display.width", 160)
FIG_DIR = Path("reports/figures"); FIG_DIR.mkdir(parents=True, exist_ok=True)
def savefig(name): plt.savefig(FIG_DIR / f"{name}.png", dpi=130, bbox_inches="tight")
""")

# =========================================================================== 1
md(r"""
## 1. Problem and market case

**Customer.** The insider-risk / security team of a **mid-sized bank (~2,000 employees)**, and the
data-loss-prevention (DLP) or HR-compliance vendors that sell to such teams. Banks are a good first
customer: they hold valuable customer and financial data, regulators expect an insider-risk programme,
and they already collect the needed logs (Active Directory logons, endpoint USB control, mail gateway,
web proxy).

**The threat and its cost.** Insiders (employees, contractors) steal data or sabotage systems **using
access they legitimately have**, so perimeter controls and signature-based tools do not see them. The
Ponemon Institute *Cost of Insider Threats Global Report (2022)* puts the average annual cost of insider
incidents at about **USD 15.4 M per organisation**, about **USD 650 k per malicious-insider
incident**, and an average **85 days to contain** an incident. *(Verify against the latest edition
before the talk.)*

**Why AI helps.** Static rules ("alert on any USB use after 19:00") either drown analysts in alerts or
miss slow, low-volume theft. Machine learning can (i) learn a **behaviour baseline per user and per role**,
(ii) combine weak signals across logon, device, file, e-mail and web logs into one risk score, and
(iii) rank user-days so analysts review the riskiest first.

**Unit of analysis.** One **user-day** (a user's activity on one calendar day). The model scores each
user-day; analysts receive the top-ranked user-days as a watch-list.

**Success metric (defined up front).**
* **Primary:** *recall* of malicious user-days, at a **workable alert budget of ≤ 5 alerts per 1,000 users
  per day**. Missing an insider is much more costly than one extra review, so recall is weighted above precision.
* **Reported alongside:** precision, F1, F2 (recall-weighted), **PR-AUC** (threshold-free, robust to the
  <1 % positive rate), the confusion matrix, and **insider-level detection** (was each insider flagged at
  least once, and how many days after their first malicious act).
* **Business metric:** analyst hours spent on alerts + estimated cost of missed malicious user-days (§12).
* **Accuracy is *not* used to select models:** with ~0.4 % malicious user-days, a model that never
  raises an alert is 99.6 % accurate and useless.
* **Phase 2 target:** recall ≥ 0.80 with precision ≥ 0.30 on the test period, and every test-period
  insider flagged within 3 days of starting.
""")

# =========================================================================== 2
md(r"""
## 2. Privacy and ethics note

User behaviour analytics watches **people**, not machines, so it can cause real harm if it is misused.

1. **Data.** We use only the public, fully **synthetic** CERT Insider Threat Test Dataset (no real
   employees) and, until it is downloaded, a synthetic generator in the same format. No real
   organisation's logs are collected. E-mail and web *content* columns are never loaded; we use metadata
   counts only (data minimisation).
2. **Pseudonymisation.** Models see behaviour counts and job role, never names, e-mail addresses or
   psychometric scores. User IDs are kept only to join tables and to route an alert; they are not model inputs.
3. **Decision support, not verdicts.** A high score means "worth a closer look", **not** "guilty".
   Every alert must be reviewed by a trained analyst, follow a documented HR/legal process, and the
   employee must have a right to respond. The system must never trigger automatic sanctions.
4. **False accusations are the main harm.** False positives cost analyst time *and* can damage an
   innocent employee's reputation. So we report false positives explicitly, keep alert volumes low,
   and in Phase 2 we will measure **false-positive impact per role** (e.g. are IT admins or on-call
   staff flagged unfairly often?).
5. **Proportionality and transparency.** In a real deployment, monitoring must be disclosed in the
   acceptable-use policy, be lawful under local law (e.g. GDPR-style data-protection rules and labour law),
   keep data only for a limited period, and restrict who can see scores.
6. **Bias.** Role and working-pattern features may correlate with protected traits (e.g. carers working
   unusual hours). Features are behaviour-based, psychometric data is excluded, and Phase 2 will audit
   error rates across roles.
""")

# =========================================================================== 3
md(r"""
## 3. Dataset: source and licence

| | |
|---|---|
| **Primary dataset** | **CERT Insider Threat Test Dataset, release r4.2**, Carnegie Mellon University Software Engineering Institute (CERT Division) with ExactData LLC |
| **Where** | CMU KiltHub: <https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247> |
| **Licence** | Distributed for research use; the licence/copyright terms are in `license.txt` inside the archive (ExactData terms). **Cite the KiltHub record; read `license.txt` after downloading and quote it in the report.** |
| **Content** | ~1,000 synthetic employees over ~17 months; `logon`, `device`, `file`, `email`, `http` logs, monthly `LDAP` snapshots, `psychometric.csv`; separate `answers` archive with labelled insiders (scenarios 1–3 in r4.2) |
| **Size** | ~1.3 GB zipped, ~16 GB unzipped (most of it `http.csv`) |
| **Second source (Phase 2)** | LANL *Comprehensive, Multi-Source Cyber-Security Events* authentication logs (<https://csr.lanl.gov/data/cyber1/>), for checking generalisation |

**What runs below.** The team has not downloaded r4.2 yet, so the notebook defaults to a **synthetic dataset
written in exactly the r4.2 format**: same files, columns, timestamp format, LDAP layout and answers layout.
The generator plays out the three r4.2 insider scenarios and adds realistic benign look-alikes (on-call
staff, heavy-but-legitimate USB users, casual job-site visits, ordinary leavers). **Every downstream cell is
unchanged when `USE_SYNTHETIC = False`.** All numbers in this notebook are therefore *pipeline-validation*
results and must be re-run on r4.2 before submission.
""")

code(r"""
USE_SYNTHETIC = True                      # False -> use the real CERT r4.2 files below

if USE_SYNTHETIC:
    DATA_DIR = generate_synthetic_cert("data/raw/synthetic_r4.2", n_users=200, n_days=270, seed=SEED)
    ANSWERS_DIR = DATA_DIR / "answers"
else:
    # Download r4.2.tar.bz2 + answers.tar.bz2 from KiltHub and extract to data/raw/
    # (in Colab: mount Google Drive and point these paths there)
    DATA_DIR = Path("data/raw/r4.2")
    ANSWERS_DIR = Path("data/raw/answers")

print("DATA_DIR   :", DATA_DIR.resolve())
print("ANSWERS_DIR:", ANSWERS_DIR.resolve())
print(sorted(p.name for p in DATA_DIR.iterdir()))
""")

# =========================================================================== 4
md(r"""
## 4. Log schema

Five event logs share the key `(id, date, user, pc)`. `date` is `MM/DD/YYYY HH:MM:SS`. The LDAP folder holds
one HR snapshot per month (leavers disappear). Ground truth is in `answers/insiders.csv` plus one file per
insider listing their malicious events.
""")

code(r"""
schema = {
    "logon.csv":  ("Windows logon/logoff",                    "activity ∈ {Logon, Logoff}"),
    "device.csv": ("Removable media (USB) connect/disconnect", "activity ∈ {Connect, Disconnect}"),
    "file.csv":   ("Files copied to removable media",         "filename, content (content not used)"),
    "email.csv":  ("Sent e-mail metadata",                    "to/cc/bcc (';'-separated), from, size (bytes), attachments (count)"),
    "http.csv":   ("Web proxy requests",                      "url, content (content not used)"),
}
rows = []
for f, (desc, notes) in schema.items():
    head = pd.read_csv(DATA_DIR / f, nrows=3)
    with open(DATA_DIR / f, "rb") as fh:
        n = sum(1 for _ in fh) - 1
    rows.append({"file": f, "what it records": desc, "columns": ", ".join(head.columns), "rows": f"{n:,}", "notes": notes})
ldap_example = pd.read_csv(sorted((DATA_DIR / "LDAP").glob("*.csv"))[0], nrows=3)
rows.append({"file": "LDAP/YYYY-MM.csv", "what it records": "Monthly HR directory snapshot",
             "columns": ", ".join(ldap_example.columns), "rows": f"{len(list((DATA_DIR/'LDAP').glob('*.csv')))} files", "notes": "role, unit, supervisor"})
rows.append({"file": "answers/insiders.csv", "what it records": "Ground-truth insiders",
             "columns": "dataset, scenario, details, user, start, end", "rows": "", "notes": "details -> per-insider event file"})
display(pd.DataFrame(rows).style.hide(axis="index"))
for f in ["logon.csv", "device.csv", "file.csv", "email.csv", "http.csv"]:
    print(f"\n--- {f}"); display(pd.read_csv(DATA_DIR / f, nrows=3))
""")

# =========================================================================== 5
md(r"""
## 5. Loading and cleaning

`build_from_dir` loads each log **without the large free-text `content` columns**, cleans it, aggregates
it into user-day features, and reads the labels. Cleaning steps per log:

* drop duplicate event `id`s; drop rows missing `date`, `user` or `pc`;
* parse timestamps with the fixed format (unparseable dates are counted and dropped);
* normalise case (user, pc, e-mail addresses, URLs, file names), and convert `size`/`attachments` to numbers;
* sort chronologically. `http.csv` is processed in **1 M-row chunks** so the 14 GB real file fits in Colab RAM.
""")

code(r"""
%%time
build = build_from_dir(DATA_DIR, ANSWERS_DIR, save_to="data/processed/user_day_features.csv")
table = build.table
display(pd.DataFrame([r.as_dict() for r in build.reports]).style.hide(axis="index"))
print("User-day feature table:", table.shape)
""")

# =========================================================================== 6
md(r"""
## 6. Exploratory data analysis (unlabelled, whole timeline)

This section only describes *behaviour* and does not look at labels, so using the whole timeline is safe.
Label-aware EDA (benign vs malicious) is done **on the training period only** in §9.
""")

code(r"""
logon = build.logs["logon"]
counts = {k: len(v) for k, v in build.logs.items()} | {"http": int(build.aggs["http"]["n_http"].sum())}
fig, ax = plt.subplots(1, 3, figsize=(17, 4))
pd.Series(counts).sort_values().plot.barh(ax=ax[0], color="steelblue"); ax[0].set_title("Events per log"); ax[0].set_xscale("log")
dau = logon[logon.activity == "Logon"].groupby(logon.date.dt.normalize())["user"].nunique()
dau.plot(ax=ax[1], lw=1); ax[1].set_title("Daily active users (weekends dip, leavers drift down)"); ax[1].set_xlabel("")
lh = logon[logon.activity == "Logon"].date
(lh.dt.hour + lh.dt.minute / 60).plot.hist(bins=48, ax=ax[2], color="slategray")
ax[2].axvspan(0, 7, color="orange", alpha=.15); ax[2].axvspan(19, 24, color="orange", alpha=.15, label="after hours")
ax[2].set_title("Logon time of day"); ax[2].set_xlabel("hour"); ax[2].legend()
plt.tight_layout(); savefig("eda_overview"); plt.show()
""")

code(r"""
fig, ax = plt.subplots(1, 3, figsize=(17, 4))
per_user = table.groupby("user")[["n_usb", "n_email", "n_http"]].mean()
sns.histplot(per_user["n_usb"], bins=40, ax=ax[0]); ax[0].set_title("Mean USB connections per active day, per user")
ax[0].set_xlabel("~70% of users never use USB")
table["role"].value_counts().plot.bar(ax=ax[1], color="teal"); ax[1].set_title("User-days by role"); ax[1].tick_params(axis="x", rotation=45)
sns.boxplot(data=table, x="role", y="n_email", ax=ax[2], showfliers=False); ax[2].set_title("E-mails per day by role")
ax[2].tick_params(axis="x", rotation=45)
plt.tight_layout(); savefig("eda_users_roles"); plt.show()
display(table.describe().T.round(2))
""")

# =========================================================================== 7
md(r"""
## 7. Per-user daily feature table

Each row is **one user on one day**. Features are counts from that day's events, plus **personal-baseline
features** that compare today with the same user's *previous* days (`*_dev`, `first_usb_ever`). These
history features use only past rows (`cumsum` minus today, then divided by the number of prior days), so no
future information enters; §10 tests this directly.
""")

code(r"""
num_cols, cat_cols = feature_columns(table)
fd = pd.DataFrame([(k, *v) for k, v in FEATURE_DESCRIPTIONS.items()], columns=["feature", "source", "description"])
fd["used"] = fd.feature.isin(num_cols + cat_cols)
display(fd.style.hide(axis="index"))
print(f"{len(num_cols)} numeric + {len(cat_cols)} categorical features")
display(table[["user", "day"] + num_cols[:12] + ["role", TARGET]].head())
""")

# =========================================================================== 8
md(r"""
## 8. Split before fitting: time-based

We split **by calendar time**: the first 70 % of days are training, the last 30 % are the test. This matches
deployment (train on the past, score the future) and stops the model from learning an insider's later
behaviour and being tested on earlier days. **Nothing is fitted before this split**: imputation, scaling and
encoding are all inside the model `Pipeline` and are fitted on the training rows only. The random seed is fixed (`SEED = 42`).
""")

code(r"""
train, test, cutoff = time_split(table, test_frac=0.30)
summary = pd.DataFrame({
    "period": [f"{train.day.min().date()} → {train.day.max().date()}", f"{test.day.min().date()} → {test.day.max().date()}"],
    "user-days": [len(train), len(test)],
    "malicious user-days": [int(train[TARGET].sum()), int(test[TARGET].sum())],
    "malicious rate": [f"{train[TARGET].mean():.3%}", f"{test[TARGET].mean():.3%}"],
    "insiders": [train.loc[train[TARGET]==1, "user"].nunique(), test.loc[test[TARGET]==1, "user"].nunique()],
}, index=["train", "test"])
display(summary)

fig, ax = plt.subplots(1, 2, figsize=(14, 3.6))
daily = table.groupby("day")[TARGET].sum()
ax[0].bar(daily.index, daily.values, width=1, color="crimson"); ax[0].axvline(cutoff, c="k", ls="--", label="train/test cutoff")
ax[0].set_title("Malicious user-days over time"); ax[0].legend()
table[TARGET].value_counts().rename({0: "benign", 1: "malicious"}).plot.bar(ax=ax[1], color=["steelblue", "crimson"], logy=True)
ax[1].set_title(f"Class imbalance: {table[TARGET].mean():.2%} positive (log scale)"); ax[1].tick_params(axis="x", rotation=0)
plt.tight_layout(); savefig("split_and_imbalance"); plt.show()
""")

# =========================================================================== 9
md(r"""
## 9. Label-aware EDA (training period only)

How do malicious user-days differ from benign ones? We use the **training split only**, so feature and model
choices are not shaped by the test set.
""")

code(r"""
key = ["n_after_hours_logon", "n_usb", "n_usb_after_hours", "n_file", "n_http_job", "n_http_leak",
       "n_email_other_sender", "first_usb_ever"]
# most counts are zero on most days, so compare HOW OFTEN each behaviour occurs, and its mean
present = (train[key] > 0).groupby(train[TARGET]).mean().T * 100
present.columns = ["benign", "malicious"]
means = train.groupby(TARGET)[key].mean().T
means.columns = ["benign", "malicious"]
fig, ax = plt.subplots(1, 2, figsize=(17, 4.5))
present.plot.bar(ax=ax[0], color=["steelblue", "crimson"])
ax[0].set_ylabel("% of user-days with value > 0"); ax[0].set_title("Training period: how often each behaviour occurs")
means.plot.bar(ax=ax[1], color=["steelblue", "crimson"], logy=True)
ax[1].set_ylabel("mean per user-day (log scale)"); ax[1].set_title("Training period: mean value per user-day")
for a in ax: a.tick_params(axis="x", rotation=25)
plt.tight_layout(); savefig("eda_benign_vs_malicious"); plt.show()
display(pd.concat({"% days > 0": present.round(1), "mean": means.round(2)}, axis=1))

corr = train[num_cols].corrwith(train[TARGET]).sort_values()
plt.figure(figsize=(7, 8)); corr.plot.barh(color=np.where(corr > 0, "crimson", "steelblue"))
plt.title("Correlation of each feature with the label (train)"); savefig("feature_label_corr"); plt.show()

ins = load_insider_answers(ANSWERS_DIR)
display(ins.groupby("scenario").agg(insiders=("user", "count")).rename(index={
    1: "1 · after-hours USB + leak upload", 2: "2 · job hunting + data theft", 3: "3 · admin keylogger + mass mail"}))
""")

# =========================================================================== 10
md(r"""
## 10. Preprocessing pipeline and leakage check

**One `Pipeline` per model.** The `ColumnTransformer` does:
* numeric → `SimpleImputer(median)` → `signed log1p` (counts are heavy-tailed) → `StandardScaler`;
* categorical (`role`, `functional_unit`) → `OneHotEncoder(handle_unknown="ignore")`.

**Class imbalance.** At ~0.4 % positives we use `class_weight="balanced"` in the loss: positives are
up-weighted by ~250×, with no synthetic insiders created. We compare against an unweighted model below. Resampling
(SMOTE / undersampling) inside an `imblearn` pipeline is planned for Phase 2.

**Leakage checks.** We check the time ordering, that no user-day appears in both sets, that identifiers and
label-derived columns are absent, that no feature is a near-copy of the label, that the scaler is fitted on
training data only, and **prefix invariance**: the feature table is rebuilt from logs *truncated at the cutoff*
and every training row must match exactly.
""")

code(r"""
X_train, y_train = train[num_cols + cat_cols], train[TARGET]
X_test,  y_test  = test[num_cols + cat_cols],  test[TARGET]

baseline = build_baseline_pipeline(num_cols, cat_cols, class_weight="balanced")
baseline.fit(X_train, y_train)          # <- the ONLY place the baseline sees data, and it is train only
baseline
""")

code(r"""
checks = leakage_checks(train, test, num_cols, cat_cols, fitted=baseline)
checks = pd.concat([checks, pd.DataFrame([prefix_invariance_check(build, cutoff, num_cols)])], ignore_index=True)
display(checks.style.hide(axis="index").map(lambda v: "color:green;font-weight:bold" if v == "PASS"
                                            else ("color:red;font-weight:bold" if v == "FAIL" else ""), subset=["result"]))
assert (checks.result == "PASS").all(), "Leakage check failed: do not trust the results below"
""")

# =========================================================================== 11
md(r"""
## 11. Baseline model and metrics

**Baseline = Logistic Regression** (class-weighted, default threshold 0.5). We chose it because it is the
simplest sensible probabilistic model: fast, stable on tens of thousands of rows, and its coefficients are
directly interpretable, which matters when a score can affect an employee. Two reference points:

* **Unweighted logistic regression**: shows what class-imbalance handling changes;
* **SIEM rule**: "USB after hours OR leak/hacking site", the kind of static rule a SOC would write without ML.

All models are scored on the **unseen test period**.
""")

code(r"""
unweighted = build_baseline_pipeline(num_cols, cat_cols, class_weight=None).fit(X_train, y_train)
s_bal = baseline.predict_proba(X_test)[:, 1]
s_unw = unweighted.predict_proba(X_test)[:, 1]
preds = {
    "LogReg balanced (baseline)": (s_bal >= 0.5).astype(int),
    "LogReg unweighted":          (s_unw >= 0.5).astype(int),
    "SIEM rule":                  rule_baseline(test),
    "Always benign":              np.zeros(len(test), dtype=int),
}
scores = {"LogReg balanced (baseline)": s_bal, "LogReg unweighted": s_unw}
results = {k: classification_metrics(y_test, p, scores.get(k)) for k, p in preds.items()}
res = results_table(results)
display(res.style.format({c: "{:.3f}" for c in ["precision", "recall", "f1", "f2", "pr_auc", "roc_auc", "accuracy"]}, na_rep="–")
        .format({c: "{:.0f}" for c in ["tp", "fp", "fn", "tn", "alerts"]})
        .background_gradient(subset=["recall", "f1", "pr_auc"], cmap="Greens"))
print("Note: 'Always benign' reaches %.1f%% accuracy while catching nothing, which is why accuracy is not our metric."
      % (100 * results["Always benign"]["accuracy"]))
""")

code(r"""
fig, ax = plt.subplots(1, 4, figsize=(20, 4.2))
for a, name in zip(ax[:3], ["LogReg balanced (baseline)", "LogReg unweighted", "SIEM rule"]):
    plot_confusion(y_test, preds[name], name, ax=a)
rule_m = results["SIEM rule"]
plot_pr_curves(y_test, scores, ax=ax[3], marker_points={"SIEM rule": (rule_m["recall"], rule_m["precision"])})
plt.tight_layout(); savefig("baseline_confusion_pr"); plt.show()
""")

code(r"""
print("Insider-level detection by the baseline (test period):")
display(insider_level_detection(test, preds["LogReg balanced (baseline)"]).style.hide(axis="index"))

names = baseline.named_steps["prep"].get_feature_names_out()
coef = pd.Series(baseline.named_steps["clf"].coef_[0], index=[n.split("__", 1)[1] for n in names])
top = coef.reindex(coef.abs().sort_values(ascending=False).index)[:15].sort_values()
plt.figure(figsize=(7, 5)); top.plot.barh(color=np.where(top > 0, "crimson", "steelblue"))
plt.title("Baseline: top-15 standardised coefficients\n(red = raises insider risk)"); savefig("baseline_coefficients"); plt.show()
""")

# =========================================================================== 12
md(r"""
## 12. Business metric: analyst hours and cost of missed insiders

Stated assumptions (edit them in `CostAssumptions` for the customer):

* an analyst spends **20 min** triaging one alerted user-day, at a fully loaded **USD 45 / hour**;
* one missed malicious user-day costs **≈ USD 7,600**: about USD 648 k per malicious-insider incident ÷ about 85 days to
  contain (Ponemon 2022). This is a rough planning figure, not a precise loss model.

*Total cost = review cost of all alerts + cost of missed malicious user-days.* The "alert budget" column
checks the success criterion of ≤ 5 alerts per 1,000 users per day.
""")

code(r"""
cost = CostAssumptions()
n_days, n_users = test["day"].nunique(), test["user"].nunique()
biz = pd.DataFrame({k: business_metrics(m, n_days, n_users, cost) for k, m in results.items()}).T
biz["within alert budget (≤5/1000/day)"] = biz["alerts_per_1000_users_per_day"] <= 5
display(biz.style.format({"analyst_hours": "{:.1f}", "alerts_per_day": "{:.2f}", "alerts_per_1000_users_per_day": "{:.1f}",
                          "review_cost_usd": "${:,.0f}", "missed_insider_cost_usd": "${:,.0f}", "total_cost_usd": "${:,.0f}"}))
print("Assumptions:", cost_assumptions_dict(cost))
""")

code(r"""
# Save the artefacts the report and Phase 2 build on
Path("reports").mkdir(exist_ok=True)
with open("reports/phase1_metrics.json", "w") as fh:
    json.dump({"data": "synthetic CERT-r4.2 format" if USE_SYNTHETIC else "CERT r4.2",
               "cutoff": str(cutoff.date()), "split": summary.astype(str).to_dict(),
               "results": results, "business": biz.astype(object).to_dict(orient="index"),
               "cost_assumptions": cost_assumptions_dict(cost),
               "leakage_checks": checks.to_dict(orient="records")}, fh, indent=2, default=float)
res.round(4).to_csv("reports/phase1_results_table.csv")
print("Saved: data/processed/user_day_features.csv, reports/phase1_metrics.json, reports/phase1_results_table.csv, reports/figures/*.png")
""")

# =========================================================================== 13
md(r"""
## 13. Phase 1 conclusions

* A **leak-free pipeline** turns five raw log types into a labelled per-user daily table; all leakage checks pass.
* The class-weighted **logistic-regression baseline** catches most malicious user-days and flags every
  test-period insider, but at threshold 0.5 its **precision is low** (too many alerts for the budget). Without
  class weights, the model is far more precise but misses a few more days. **Choosing the threshold is a
  business decision**, which Phase 2 makes explicit.
* Interesting: the *unweighted* model has the higher **PR-AUC** (it ranks better), while class weighting mostly
  shifts scores upward. With very few positives, heavy re-weighting can overfit them. Phase 2 will therefore
  compare class weights against resampling and threshold tuning, rather than assume weighting is best.
* Both ML models beat the static SIEM rule on recall and PR-AUC, so combining weak signals across log sources helps.
* Accuracy is uninformative here (an "always benign" model scores 99.6 %).

## 14. Plan and risks for Phase 2

| Phase 2 item (12 marks) | Plan |
|---|---|
| ≥ 3 models, CV tuning | **Isolation Forest** (unsupervised user-day scores), **Random Forest** and **gradient boosting** (supervised), **LOF** as a comparison; tuned with forward-chaining time-series CV on the training period only |
| Threshold for business cost | choose the threshold that minimises `total_cost_usd` under the alert budget; report the watch-list (top-k) precision |
| Explainability | **SHAP** for one false negative and one false positive, and SHAP reasons per user on the ranked watch-list |
| Robustness | **drift test** (train on early months, test on later ones) + an **evasion test** (insider spreads USB copies over normal hours) |
| Demo | Streamlit dashboard: upload logs → ranked watch-list with SHAP reasons |
| Stretch | **peer-group baselines** (each user vs their role); LSTM on daily sequences |

**Risks and mitigations**
1. *Real r4.2 size (16 GB)*: chunked http aggregation is already in place; cache the feature table to Google Drive.
2. *Very few positives (r4.2 has about 70 insiders)*: use PR-AUC and insider-level metrics, time-series CV, and class weights.
3. *Synthetic-to-real gap*: all numbers here validate the pipeline only and must be re-run on r4.2.
4. *Label granularity*: answers mark events; we label a user-day malicious if it holds ≥ 1 malicious event (documented choice).
5. *False-positive harm to staff*: alert budget, human review, and an audit of error rates per role.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}, "colab": {"provenance": []}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, NB_PATH)
    print("Wrote", NB_PATH)
    if args.execute:
        from nbclient import NotebookClient
        client = NotebookClient(nb, timeout=1200, kernel_name="python3",
                                resources={"metadata": {"path": str(NB_PATH.parent)}})
        client.execute()
        nbf.write(nb, NB_PATH)
        print("Executed and saved with outputs")


if __name__ == "__main__":
    main()
