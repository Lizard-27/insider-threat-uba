# Insider Threat & User Behaviour Analytics from Logs

**CET433 Selected Topics in Cyber, Fall 2026, Project 8.** The project scores every *user-day* from enterprise
logs (logon, USB, file copy, e-mail, web) for insider-threat risk and gives analysts a ranked watch-list.

| Member | Role |
|---|---|
| **Mohamed Moamen** (team leader) | Problem & market case, privacy/ethics, integration, presentation |
| Ahmed Mohamed | Data acquisition, log schema, cleaning, EDA |
| Marwan Hassan | Feature engineering (per-user daily table), labelling |
| Ahmed Osama | Pipeline, leakage checks, baseline model, metrics |

## Phase 1 status

| Phase 1 item (8 marks) | Where |
|---|---|
| Problem & market case, success metric | [docs/proposal.md](docs/proposal.md), notebook §1 |
| Privacy & ethics note | notebook §2, [docs/phase1_report.md](docs/phase1_report.md) §3 |
| Dataset source & licence | notebook §3, [data/README.md](data/README.md) |
| Log schema | notebook §4 |
| Cleaning + EDA | notebook §5–6, §9 |
| Per-user daily feature table | notebook §7, `src/uba/features.py` |
| Split before fit (time-based), leakage check | notebook §8, §10, `src/uba/modeling.py` |
| Class-imbalance handling | `class_weight="balanced"` vs unweighted, notebook §10–11 |
| Baseline + confusion matrix, P/R/F1, business metric | notebook §11–12 |

> **Data note.** Results committed now come from a **synthetic dataset in the exact CERT r4.2 format**
> (`src/uba/synthetic_cert.py`). The pipeline needs no code changes to run on the real r4.2 files; set
> `USE_SYNTHETIC = False` in the notebook. Re-run on r4.2 before final submission.

## Quick start

**Google Colab (recommended)**
1. Push this repo to GitHub and set `REPO_URL` in the first cell of `notebooks/Phase1_Insider_Threat_UBA.ipynb`.
2. Open the notebook in Colab (File → Open notebook → GitHub) and choose **Runtime → Run all**.
   The notebook clones the repo, generates the synthetic data (~1 min) and runs everything.

**Local**
```bash
pip install -r requirements.txt
python scripts/run_phase1.py                 # full pipeline, prints metrics, writes reports/phase1_metrics.json
jupyter notebook notebooks/Phase1_Insider_Threat_UBA.ipynb
```

**Real CERT r4.2**
```bash
# extract r4.2.tar.bz2 -> data/raw/r4.2/ and answers.tar.bz2 -> data/raw/answers/
python scripts/run_phase1.py --data data/raw/r4.2 --answers data/raw/answers
```
`http.csv` (~14 GB) is aggregated in 1 M-row chunks, and text `content` columns are never loaded.

## Repository layout

```
src/uba/
  synthetic_cert.py   CERT-r4.2-format synthetic log generator (3 insider scenarios + benign confounders)
  data_loading.py     load + clean logs, LDAP, ground-truth answers -> malicious user-days
  features.py         per-user daily feature table (counts + leak-free personal-baseline features)
  modeling.py         time split, sklearn Pipeline, baseline, SIEM-rule reference, leakage checks
  evaluation.py       precision/recall/F1/F2/PR-AUC, confusion matrix, insider-level detection, business cost
scripts/
  run_phase1.py       command-line run of the whole Phase 1 pipeline
  build_notebook.py   regenerates (and with --execute runs) the Phase 1 notebook
notebooks/Phase1_Insider_Threat_UBA.ipynb   main deliverable (executed, with outputs)
docs/                 proposal, Phase 1 report draft, contribution log
reports/              metrics JSON, results table, figures
data/                 (git-ignored) raw and processed data
```

## Method in one paragraph

Raw events are cleaned (dedupe, missing keys, timestamp parsing, case normalisation) and aggregated into one
row per **user per day**. Each row has 35 numeric behaviour features (after-hours logons, USB connections, file
copies, external e-mail, recipients, sender-identity mismatch, web categories such as job, leak, cloud and hacking
sites) plus role, and **personal-baseline deviations** computed from the user's *previous* days only. A user-day
is labelled malicious if the CERT answers list at least one malicious event on it. Data is split **by time**
(first 70 % of days train, last 30 % test). One scikit-learn `Pipeline` (median impute → signed log1p →
standard scale / one-hot → class-weighted logistic regression) is fitted on training data only. Seven
documented leakage checks, including a **prefix-invariance** test that rebuilds features from truncated logs,
must all pass before metrics are reported.

## Licence and ethics

Code: for coursework use. Data: CERT Insider Threat Test Dataset © Carnegie Mellon University / ExactData.
See its `license.txt`. The synthetic data contains no real people. See the privacy & ethics note
in the notebook (§2).
