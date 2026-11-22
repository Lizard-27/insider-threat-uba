# Project Proposal: Insider Threat & User Behaviour Analytics from Logs

**CET433 Selected Topics in Cyber · Fall 2026 · Project 8 · Phase 1**
**Team:** Mohamed Moamen (team leader) · Ahmed Mohamed · Marwan Hassan · Ahmed Osama

### Customer
The **insider-risk / security team of a mid-sized bank (~2,000 employees)**. Secondary buyers are
data-loss-prevention (DLP) and HR-compliance vendors who would embed the model in their product. Banks hold
high-value customer and financial data, regulators expect an insider-risk programme, and the needed logs
(Active Directory logons, endpoint USB control, mail gateway, web proxy) already exist.

### Problem and cost of the threat
Employees and contractors who steal data or sabotage systems use **legitimate access**, so perimeter and
signature-based tools miss them. The Ponemon Institute *Cost of Insider Threats Global Report 2022* estimates
the average annual cost of insider incidents at **~USD 15.4 M per organisation**, about **USD 650 k per
malicious-insider incident**, and **~85 days to contain** an incident *(figures to be re-checked against the
latest edition)*. Static rules ("alert on any after-hours USB use") either flood analysts with alerts or miss
slow, low-volume theft.

### Why AI helps
Machine learning can learn a **behaviour baseline per user and per role**, combine weak signals from
five log sources into one risk score, and **rank user-days** so a small analyst team reviews the riskiest
first, instead of reviewing every rule hit.

### Approach
* **Data:** CERT Insider Threat Test Dataset r4.2 (CMU SEI / ExactData; logon, device, file, e-mail, web,
  LDAP + labelled insiders). LANL authentication logs are a second source in Phase 2. A synthetic generator in
  the exact r4.2 format validates the pipeline before the 16 GB download.
* **Unit:** one row per user per day, with 35 behaviour features plus role and personal-baseline deviations.
* **Phase 1 baseline:** class-weighted Logistic Regression in a leak-free scikit-learn Pipeline with a
  time-based split, compared with a hand-written SIEM rule.
* **Phase 2:** Isolation Forest (unsupervised), Random Forest and gradient boosting (supervised), and LOF as a
  comparison. Then a cost-based threshold, SHAP reasons per user, a drift/evasion test and a Streamlit watch-list dashboard.

### Success metric (fixed up front)
* **Primary:** recall of malicious user-days at **≤ 5 alerts per 1,000 users per day**.
* **Also reported:** precision, F1, F2, PR-AUC, confusion matrix, and insider-level detection (each
  insider flagged, and days to first flag). Accuracy is not used: an "always benign" model is 99.6 % accurate.
* **Business metric:** analyst hours (20 min per alert at USD 45/h) plus estimated cost of missed malicious
  days (≈ USD 7.6 k/day = incident cost ÷ days to contain).
* **Phase 2 target:** recall ≥ 0.80 at precision ≥ 0.30; every test insider flagged within 3 days.

### Privacy and ethics (summary)
Synthetic data only; text content never loaded; no names or psychometrics used as features; scores are
**decision support for a human analyst under an HR/legal process, never automatic sanctions**; false positives
and per-role error rates are reported; real deployment requires disclosure, a lawful basis and limited retention.

### Team roles
| Member | Phase 1 | Phase 2 |
|---|---|---|
| Mohamed Moamen (lead) | Market case, ethics, integration, talk | Threshold & business cost, report, demo integration |
| Ahmed Mohamed | Dataset, schema, cleaning, EDA | Real r4.2 run, drift test, LANL source |
| Marwan Hassan | Feature table, labelling | Peer-group features, SHAP explanations |
| Ahmed Osama | Pipeline, leakage checks, baseline | Model comparison & CV tuning, Streamlit dashboard |
