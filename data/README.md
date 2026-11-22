# Data folder (not committed)

| Path | Contents | How to get it |
|---|---|---|
| `raw/synthetic_r4.2/` | Synthetic logs in the exact CERT r4.2 format | Created automatically by the notebook or `python scripts/run_phase1.py` |
| `raw/r4.2/` | Real CERT Insider Threat r4.2: `logon.csv`, `device.csv`, `file.csv`, `email.csv`, `http.csv`, `LDAP/`, `psychometric.csv` | Download `r4.2.tar.bz2` from <https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247> and extract here |
| `raw/answers/` | Ground truth: `insiders.csv` and `r4.2-1/`, `r4.2-2/`, `r4.2-3/` | Download `answers.tar.bz2` from the same KiltHub record and extract here |
| `processed/user_day_features.csv` | The per-user daily feature table | Written by the pipeline |

Read `license.txt` inside the r4.2 archive and quote its terms in the report.
