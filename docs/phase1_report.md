# Insider Threat & User Behaviour Analytics from Logs: Phase 1 Report (draft)

**CET433 Selected Topics in Cyber · Fall 2026 · Project 8**
**Team:** Mohamed Moamen (team leader), Ahmed Mohamed, Marwan Hassan, Ahmed Osama
**Instructors:** Dr. Mariam Labib, Eng. Maryam Adel

> Draft status: all numbers below come from the **synthetic CERT-r4.2-format dataset** and validate the pipeline.
> Replace them with results on the real CERT r4.2 data before submission (re-run the notebook with
> `USE_SYNTHETIC = False`). Places to update are marked **[UPDATE]**.

---

## 1. Introduction and market case

### 1.1 Problem
An *insider threat* is harm to an organisation by someone with authorised access: an employee, contractor or
partner who steals data, commits fraud or sabotages systems. Insiders act through legitimate accounts and
during normal work, so firewalls, antivirus and signature-based intrusion detection do not see the attack. The
evidence is in **behaviour**: logging on at unusual times, first-time use of USB drives, copying many files,
visiting job or leak sites, or using a colleague's machine or identity.

### 1.2 Customer
Our named customer is the **insider-risk / security team of a mid-sized bank (~2,000 employees)**. Secondary
customers are **DLP and HR-compliance vendors** that could embed the model. Banks hold high-value data, face
regulatory expectations for insider-risk controls, and already collect the logs needed (Active Directory,
endpoint device control, e-mail gateway, web proxy).

### 1.3 Cost of the threat
The Ponemon Institute's *Cost of Insider Threats Global Report (2022)* reports an average annual cost of
insider incidents of about **USD 15.4 M per organisation**, about **USD 650 k per malicious-insider incident**,
and an average of **85 days to contain** an incident. **[UPDATE: verify against the latest edition and cite precisely.]**

### 1.4 Why AI helps
Rule-based SIEM alerts are either too broad (alert fatigue) or too narrow (slow, low-volume theft is missed).
ML can (i) model normal behaviour **per user and per role**, (ii) combine weak signals from several log sources,
and (iii) output a **risk score** that ranks user-days, so analysts review the most suspicious first.

### 1.5 Success metric (defined before modelling)
* **Primary:** recall of malicious user-days at an alert budget of **≤ 5 alerts per 1,000 users per day**.
* **Secondary:** precision, F1, F2, PR-AUC, confusion matrix, and insider-level detection (share of insiders
  flagged at least once, and days to first flag).
* **Business:** analyst hours and total cost (review cost + missed-insider cost), §8.
* **Why not accuracy:** about 0.4 % of user-days are malicious, so a model that never alerts scores 99.6 %.
* **Phase 2 target:** recall ≥ 0.80 at precision ≥ 0.30 and every test insider flagged within 3 days.

## 2. Related work (to expand)
The CERT Insider Threat Test Dataset (Glasser & Lindauer, 2013) is the standard benchmark for insider-threat
detection. Common approaches aggregate events per user-day or user-week and apply unsupervised anomaly
detection (Isolation Forest, autoencoders) or supervised classifiers on the labelled insiders. Recurrent
models (LSTM) over daily activity sequences have also been explored. **[UPDATE: add 3–5 citations.]**

## 3. Privacy and ethics statement
1. **Data:** only public synthetic data (CERT r4.2 contains no real people) and our own synthetic generator.
   No real organisation's logs are used.
2. **Minimisation:** e-mail bodies, web-page content and file contents are never loaded; only metadata counts.
3. **No identity features:** names, e-mail addresses and psychometric scores are excluded from the model. User
   IDs are used only to join tables and route alerts.
4. **Human in the loop:** a score means "worth a review", not "guilty". Alerts go to trained analysts under a
   documented HR/legal process with a right of reply. There are no automatic sanctions.
5. **False-positive harm:** false accusations damage careers and trust. We report false positives, keep a strict
   alert budget, and in Phase 2 we audit error rates by role.
6. **Deployment conditions:** monitoring disclosed in the acceptable-use policy, lawful basis under applicable
   data-protection and labour law, limited retention, and role-based access to scores.

## 4. Data

### 4.1 Source and licence
* **CERT Insider Threat Test Dataset r4.2**, CMU SEI CERT Division & ExactData LLC, via CMU KiltHub:
  <https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247>.
  It is released for research, with terms in `license.txt` inside the archive. **[UPDATE: quote the licence terms.]**
* Second source (Phase 2): LANL *Comprehensive, Multi-Source Cyber-Security Events*, <https://csr.lanl.gov/data/cyber1/>.
* **Synthetic stand-in:** `src/uba/synthetic_cert.py` writes 200 users × 270 days in the exact r4.2 format and
  plays out the three r4.2 insider scenarios (4 + 4 + 3 insiders). It adds benign look-alikes: on-call staff
  working late, heavy legitimate USB users, casual job-site browsing, and ordinary leavers.

### 4.2 Log schema
| File | Columns | Meaning |
|---|---|---|
| `logon.csv` | id, date, user, pc, activity | Logon / Logoff |
| `device.csv` | id, date, user, pc, activity | USB Connect / Disconnect |
| `file.csv` | id, date, user, pc, filename, content | File copied to removable media |
| `email.csv` | id, date, user, pc, to, cc, bcc, from, size, attachments, content | Sent e-mail |
| `http.csv` | id, date, user, pc, url, content | Web request |
| `LDAP/YYYY-MM.csv` | employee_name, user_id, email, role, business_unit, functional_unit, department, team, supervisor | Monthly HR snapshot |
| `answers/insiders.csv` | dataset, scenario, details, user, start, end | Labelled insiders; `details` names a file listing each malicious event |

### 4.3 Exploratory analysis (key observations)
* Logons cluster at 06:30–09:30. After-hours logons (before 07:00, from 19:00, or at weekends) are rare and mostly come from on-call staff.
* About 70 % of users never use removable media, so **first-time USB use** is informative.
* Malicious user-days are **0.39 %** of the user-days (141 of 36,248) **[UPDATE]**, which is severe imbalance.
* On training data, malicious days show much higher rates of after-hours logons, USB use, file copies and
  job-site visits than benign days (figure `reports/figures/eda_benign_vs_malicious.png`).

## 5. Preprocessing and features

### 5.1 Cleaning
Each log is deduplicated on `id`; rows missing `date`, `user` or `pc` are dropped; timestamps are parsed with
the fixed format (failures counted and dropped); user, PC, address, URL and file-name values are case-normalised;
`size` and `attachments` are converted to numbers. The 14 GB `http.csv` is processed in 1 M-row chunks.

### 5.2 Per-user daily feature table
One row per (user, day) with any activity, 37 features in total:
* **Logon:** logons, after-hours logons, first logon hour, last logoff hour, distinct PCs.
* **Device / file:** USB connections (total, after hours), files copied (total, after hours, documents, executables, archives).
* **E-mail:** sent, after hours, with an external recipient, total and max recipients, attachments, volume, and
  e-mails whose *From* address is not the user's own (identity misuse).
* **Web:** requests (total, after hours) and by category: job search, leak sites, cloud storage, hacking tools.
* **Context:** day of week, weekend flag, role, functional unit (from the user's *first* LDAP snapshot).
* **Personal baseline (leak-free):** today's value minus the user's mean over *previous* days for USB, files,
  after-hours logons, external e-mail and after-hours web; first-ever USB use.

### 5.3 Labels
A user-day is **malicious** if the answer files list at least one malicious event on that day for that user.

### 5.4 Encoding, scaling and imbalance
All inside one scikit-learn `Pipeline`: median imputation → signed `log1p` (heavy-tailed counts) →
`StandardScaler` for numeric features; `OneHotEncoder(handle_unknown="ignore")` for role and functional unit.
Imbalance is handled with `class_weight="balanced"` and compared with an unweighted model.

## 6. Split and leakage control
* **Time-based split:** first 70 % of calendar days → train (2010-01-04 to 2010-07-10, 25,716 user-days, 95
  malicious, 7 insiders); last 30 % → test (2010-07-11 to 2010-09-30, 10,532 user-days, 46 malicious, 4 insiders) **[UPDATE]**.
* **Split before fit:** every fitted statistic (imputer medians, scaler means, model weights) comes from
  training rows only. Label-aware EDA also uses training rows only.
* **Fixed seed:** `SEED = 42` for data generation and models.
* **Documented leakage checks (all PASS):**

| Check | Result |
|---|---|
| Train strictly before test in time | PASS |
| No user-day in both splits | PASS |
| No identifier/label columns among features | PASS |
| No feature is a near-copy of the label (max abs corr 0.57) | PASS |
| Both splits contain positives | PASS |
| Scaler fitted on train only (`mean_` = train mean) | PASS |
| **Prefix invariance:** features rebuilt from logs truncated at the cutoff match the training rows exactly | PASS |

## 7. Baseline model and results

**Baseline:** class-weighted Logistic Regression, threshold 0.5. It is simple, fast, stable and has
interpretable coefficients, which matters when a score affects a person. References: an unweighted
logistic regression and a hand-written **SIEM rule** (after-hours USB OR leak/hacking site).

Test-period results **[UPDATE with r4.2]**:

| Model | Precision | Recall | F1 | F2 | PR-AUC | Accuracy | TP | FP | FN | Alerts |
|---|---|---|---|---|---|---|---|---|---|---|
| LogReg balanced (baseline) | 0.149 | **0.696** | 0.245 | 0.401 | 0.481 | 0.981 | 32 | 183 | 14 | 215 |
| LogReg unweighted | **0.811** | 0.652 | **0.723** | **0.679** | **0.768** | 0.998 | 30 | 7 | 16 | 37 |
| SIEM rule | 0.323 | 0.435 | 0.370 | 0.407 | n/a | 0.994 | 20 | 42 | 26 | 62 |
| Always benign | 0.000 | 0.000 | 0.000 | 0.000 | n/a | **0.996** | 0 | 0 | 46 | 0 |

**Insider-level:** the baseline flagged **all 4** test-period insiders. Three were flagged on their first
malicious day; the scenario-2 job-hunter was flagged 10 days after their first malicious day.

**Interpretation.**
* Both ML models clearly beat the static rule on recall, showing the value of combining signals.
* Class weighting raises recall but, at the default threshold, gives too many alerts (14 per 1,000 users per
  day vs the 5 budget). The unweighted model ranks better (higher PR-AUC). With few positives, heavy
  re-weighting can overfit, so Phase 2 will tune weighting, resampling and the threshold together.
* "Always benign" has the highest accuracy and catches nothing, which confirms our choice of metric.

## 8. Business metric
Assumptions: 20 analyst minutes per alert at USD 45/h; one missed malicious user-day ≈ USD 7,600 (≈ USD 648 k
per malicious-insider incident ÷ 85 days to contain, Ponemon 2022).

| Model | Analyst hours | Alerts / 1,000 users / day | Review cost | Missed-insider cost | Total |
|---|---|---|---|---|---|
| LogReg balanced | 71.7 | 14.3 | $3,225 | $106,400 | **$109,625** |
| LogReg unweighted | 12.3 | 2.5 | $555 | $121,600 | $122,155 |
| SIEM rule | 20.7 | 4.1 | $930 | $197,600 | $198,530 |
| Always benign | 0 | 0 | $0 | $349,600 | $349,600 |

Missed insiders dominate the cost, so recall matters most. But the balanced model exceeds the alert budget, and
a cost-optimal threshold is a Phase 2 deliverable.

## 9. Plan and risks for Phase 2
| Item | Plan |
|---|---|
| Models (≥ 3, CV-tuned) | Isolation Forest (unsupervised), Random Forest, gradient boosting; LOF comparison; forward-chaining time-series CV on train only |
| Threshold | minimise total cost subject to the alert budget; top-k watch-list precision |
| Explainability | SHAP for ≥ 1 false negative and ≥ 1 false positive; SHAP reasons per user on the watch-list |
| Robustness | drift (early vs late months) and evasion (insider spreads copies over work hours) |
| Demo | Streamlit: upload logs → ranked watch-list with reasons |
| Stretch | peer-group (role) baselines; LSTM on daily sequences |

**Risks:** (1) r4.2 size → chunked processing, cache features to Drive; (2) few insiders → PR-AUC,
insider-level metrics, careful CV; (3) synthetic-to-real gap → re-run everything on r4.2; (4) day-level
labels are coarse → documented; (5) false-positive harm → alert budget, human review, per-role audit.

## References
1. Lindauer, B. (2020). *Insider Threat Test Dataset*. Carnegie Mellon University. Dataset. https://doi.org/10.1184/R1/12841247 **[UPDATE: confirm DOI/citation from KiltHub]**
2. Glasser, J., & Lindauer, B. (2013). Bridging the gap: A pragmatic approach to generating insider threat data. *IEEE Security and Privacy Workshops*.
3. Ponemon Institute (2022). *Cost of Insider Threats Global Report*. **[UPDATE: confirm edition and figures]**
4. Kent, A. D. (2015). *Comprehensive, Multi-Source Cyber-Security Events*. Los Alamos National Laboratory.
5. Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *JMLR* 12.
