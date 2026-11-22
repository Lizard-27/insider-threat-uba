"""Per-user daily (user-day) behavioural feature table.

One row = one user on one calendar day with at least one logged event.
Every feature is computed from events *on that day* or from that user's
*previous* days only (``shift(1)``), so a row never sees the future. The
"prefix-invariance" leakage check in :mod:`uba.modeling` verifies this.

All per-day aggregates are plain counts/sums, so the huge real ``http.csv`` can
be aggregated chunk by chunk and summed (see :func:`aggregate_http_chunked`).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data_loading import iter_log_chunks, load_ldap, load_log, malicious_user_days

COMPANY_DOMAIN = "dtaa.com"
WORK_START_HOUR = 7   # events before 07:00 ...
WORK_END_HOUR = 19    # ... or from 19:00 onward, or on Sat/Sun, are "after hours"

# Web categories, as a commercial web proxy / DLP product would supply them.
URL_CATEGORIES = {
    "job": r"indeed|monster|careerbuilder|simplyhired|glassdoor|jobhunt|linkedin\.com/jobs|job",
    "leak": r"wikileaks",
    "cloud": r"dropbox|box\.com|drive\.google|mega\.nz|onedrive",
    "hack": r"keylog|spectorsoft|refog|hack",
}

KEYS = ["user", "day"]

FEATURE_DESCRIPTIONS = {
    "n_logon": ("logon", "Logon events that day"),
    "n_after_hours_logon": ("logon", "Logons before 07:00, from 19:00, or on a weekend"),
    "first_logon_hour": ("logon", "Hour (fractional) of the first logon"),
    "last_logoff_hour": ("logon", "Hour (fractional) of the last logoff (NaN if none -> imputed)"),
    "n_pcs": ("logon", "Distinct PCs the user touched (e.g. a colleague's machine)"),
    "n_usb": ("device", "Removable-media (USB) connections"),
    "n_usb_after_hours": ("device", "USB connections after hours"),
    "n_file": ("file", "Files copied to removable media"),
    "n_file_after_hours": ("file", "File copies after hours"),
    "n_file_doc": ("file", "Copied documents (.doc/.pdf/.txt/office)"),
    "n_file_exe": ("file", "Copied executables (.exe)"),
    "n_file_zip": ("file", "Copied archives (.zip/.rar/.7z)"),
    "n_email": ("email", "E-mails sent"),
    "n_email_after_hours": ("email", "E-mails sent after hours"),
    "n_email_ext": ("email", "E-mails with at least one recipient outside the company"),
    "n_email_rcpt": ("email", "Total recipients (to+cc+bcc)"),
    "max_email_rcpt": ("email", "Largest recipient list in one e-mail (mass mail)"),
    "n_email_attach": ("email", "Total attachments"),
    "email_kb": ("email", "Total e-mail volume in KB"),
    "n_email_other_sender": ("email", "E-mails whose From address is not the user's own (identity misuse)"),
    "n_http": ("http", "Web requests"),
    "n_http_after_hours": ("http", "Web requests after hours"),
    "n_http_job": ("http", "Requests to job-search sites"),
    "n_http_leak": ("http", "Requests to leak sites (e.g. wikileaks)"),
    "n_http_cloud": ("http", "Requests to cloud-storage sites"),
    "n_http_hack": ("http", "Requests to hacking/keylogger sites"),
    "day_of_week": ("calendar", "0 = Monday ... 6 = Sunday"),
    "is_weekend": ("calendar", "1 if Saturday/Sunday"),
    "role": ("LDAP", "Job role (first LDAP snapshot), one-hot encoded"),
    "functional_unit": ("LDAP", "Functional unit (first LDAP snapshot), one-hot encoded"),
    "user_days_seen": ("history", "Number of the user's previous active days"),
    "n_usb_dev": ("history", "Today's USB count minus the user's own past daily mean"),
    "n_file_dev": ("history", "Today's file copies minus the user's own past daily mean"),
    "n_after_hours_logon_dev": ("history", "After-hours logons minus the user's past mean"),
    "n_email_ext_dev": ("history", "External e-mails minus the user's past mean"),
    "n_http_after_hours_dev": ("history", "After-hours web requests minus the user's past mean"),
    "first_usb_ever": ("history", "1 if the user uses USB today for the first time after >= 10 active days"),
}
# behaviours we also express relative to the user's own past ("personal baseline")
HISTORY_COLS = ["n_usb", "n_file", "n_after_hours_logon", "n_email_ext", "n_http_after_hours"]


def _base(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["day"] = d["date"].dt.normalize()
    hour = d["date"].dt.hour
    d["hour"] = hour + d["date"].dt.minute / 60
    d["after_hours"] = (hour < WORK_START_HOUR) | (hour >= WORK_END_HOUR) | (d["date"].dt.dayofweek >= 5)
    return d


def aggregate_logon(logon: pd.DataFrame) -> pd.DataFrame:
    d = _base(logon)
    on = d[d["activity"] == "Logon"]
    off = d[d["activity"] == "Logoff"]
    g = on.groupby(KEYS)
    out = pd.DataFrame({
        "n_logon": g.size(),
        "n_after_hours_logon": g["after_hours"].sum(),
        "first_logon_hour": g["hour"].min(),
        "n_pcs": d.groupby(KEYS)["pc"].nunique(),
    })
    out["last_logoff_hour"] = off.groupby(KEYS)["hour"].max()
    return out


def aggregate_device(device: pd.DataFrame) -> pd.DataFrame:
    d = _base(device)
    c = d[d["activity"] == "Connect"]
    g = c.groupby(KEYS)
    return pd.DataFrame({"n_usb": g.size(), "n_usb_after_hours": g["after_hours"].sum()})


def aggregate_file(file: pd.DataFrame) -> pd.DataFrame:
    d = _base(file)
    fn = d["filename"]
    d["is_doc"] = fn.str.contains(r"\.(?:docx?|pdf|txt|xlsx?|pptx?)$", regex=True)
    d["is_exe"] = fn.str.endswith(".exe")
    d["is_zip"] = fn.str.contains(r"\.(?:zip|rar|7z)$", regex=True)
    g = d.groupby(KEYS)
    return pd.DataFrame({
        "n_file": g.size(), "n_file_after_hours": g["after_hours"].sum(),
        "n_file_doc": g["is_doc"].sum(), "n_file_exe": g["is_exe"].sum(), "n_file_zip": g["is_zip"].sum(),
    })


def aggregate_email(email: pd.DataFrame, user_email: pd.Series | None = None,
                    company_domain: str = COMPANY_DOMAIN) -> pd.DataFrame:
    d = _base(email)
    rcpt = (d["to"] + ";" + d["cc"] + ";" + d["bcc"]).str.replace(r";+", ";", regex=True).str.strip(";")
    d["n_rcpt"] = np.where(rcpt.str.len() > 0, rcpt.str.count(";") + 1, 0)
    ext_pat = r"@(?!" + re.escape(company_domain) + r"\b)"  # any recipient outside the company
    d["is_ext"] = rcpt.str.contains(ext_pat, regex=True)
    if user_email is not None:
        own = d["user"].map(user_email)
        d["other_sender"] = own.notna() & (d["from"] != own)
    else:
        d["other_sender"] = False
    g = d.groupby(KEYS)
    return pd.DataFrame({
        "n_email": g.size(), "n_email_after_hours": g["after_hours"].sum(),
        "n_email_ext": g["is_ext"].sum(), "n_email_rcpt": g["n_rcpt"].sum(),
        "max_email_rcpt": g["n_rcpt"].max(), "n_email_attach": g["attachments"].sum(),
        "email_kb": g["size"].sum() / 1024, "n_email_other_sender": g["other_sender"].sum(),
    })


def aggregate_http(http: pd.DataFrame) -> pd.DataFrame:
    d = _base(http)
    agg = {"n_http": ("url", "size"), "n_http_after_hours": ("after_hours", "sum")}
    for cat, pat in URL_CATEGORIES.items():
        d[f"is_{cat}"] = d["url"].str.contains(pat, regex=True)
        agg[f"n_http_{cat}"] = (f"is_{cat}", "sum")
    return d.groupby(KEYS).agg(**agg)


def aggregate_http_chunked(data_dir, chunksize: int = 1_000_000, nrows: int | None = None):
    """Aggregate a huge http.csv chunk by chunk (all features are additive)."""
    parts, reports = [], []
    for chunk, rep in iter_log_chunks(data_dir, "http", chunksize=chunksize, nrows=nrows):
        parts.append(aggregate_http(chunk))
        reports.append(rep)
    agg = pd.concat(parts).groupby(level=KEYS).sum()
    total = reports[0]
    for r in reports[1:]:
        for f in ("rows_in", "duplicate_ids", "missing_key_fields", "bad_dates", "rows_out"):
            setattr(total, f, getattr(total, f) + getattr(r, f))
    return agg, total


def add_history_features(df: pd.DataFrame, cols=HISTORY_COLS) -> pd.DataFrame:
    """Deviation of today's value from the same user's mean over *previous* days only."""
    df = df.sort_values(KEYS).copy()
    g = df.groupby("user", sort=False)
    df["user_days_seen"] = g.cumcount()  # number of prior active days
    for c in cols:
        past_sum = g[c].cumsum() - df[c]
        past_mean = past_sum / df["user_days_seen"].replace(0, np.nan)
        df[f"{c}_dev"] = (df[c] - past_mean).fillna(0.0)
    prior_usb = g["n_usb"].cumsum() - df["n_usb"]
    df["first_usb_ever"] = ((df["n_usb"] > 0) & (prior_usb == 0) & (df["user_days_seen"] >= 10)).astype(int)
    return df


def build_feature_table(aggs: dict[str, pd.DataFrame], ldap: pd.DataFrame,
                        labels: pd.DataFrame | None = None) -> pd.DataFrame:
    """Merge per-log aggregates into the user-day table, add context, history and label."""
    table = pd.concat(list(aggs.values()), axis=1)
    count_cols = [c for c in table.columns if c.startswith(("n_", "max_", "email_"))]
    table[count_cols] = table[count_cols].fillna(0)
    for c in count_cols:
        if c != "email_kb":
            table[c] = table[c].astype(int)
    table = table.reset_index()
    table["day_of_week"] = table["day"].dt.dayofweek
    table["is_weekend"] = (table["day_of_week"] >= 5).astype(int)
    table = table.merge(ldap[["user", "role", "functional_unit"]], on="user", how="left")
    table[["role", "functional_unit"]] = table[["role", "functional_unit"]].fillna("Unknown")
    table = add_history_features(table)
    if labels is not None:
        lab = labels[KEYS + ["scenario"]].copy()
        lab["label"] = 1
        table = table.merge(lab, on=KEYS, how="left")
        table["label"] = table["label"].fillna(0).astype(int)
        table["scenario"] = table["scenario"].fillna(0).astype(int)
    return table.sort_values(["day", "user"]).reset_index(drop=True)


def compute_aggregates(logs: dict[str, pd.DataFrame], ldap: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Aggregate in-memory cleaned logs (any subset of the five log types)."""
    email_map = ldap.set_index("user")["email"]
    fns = {"logon": aggregate_logon, "device": aggregate_device, "file": aggregate_file,
           "email": lambda d: aggregate_email(d, email_map), "http": aggregate_http}
    return {k: fns[k](v) for k, v in logs.items()}


@dataclass
class FeatureBuild:
    table: pd.DataFrame            # the user-day feature table (with label)
    logs: dict                     # cleaned logon/device/file/email frames
    aggs: dict                     # per-log user-day aggregates (incl. http)
    reports: list                  # CleaningReport per log
    ldap: pd.DataFrame
    labels: pd.DataFrame           # malicious (user, day, scenario)


def build_from_dir(data_dir, answers_dir, http_chunksize: int = 1_000_000,
                   save_to: str | Path | None = None) -> FeatureBuild:
    """End-to-end: CSV folder -> cleaned logs -> labelled user-day feature table.

    http.csv is aggregated in chunks and its raw rows are not kept in memory.
    """
    ldap = load_ldap(data_dir)
    logs, reports = {}, []
    for log in ["logon", "device", "file", "email"]:
        logs[log], rep = load_log(data_dir, log)
        reports.append(rep)
    aggs = compute_aggregates(logs, ldap)
    aggs["http"], rep = aggregate_http_chunked(data_dir, chunksize=http_chunksize)
    reports.append(rep)
    labels = malicious_user_days(answers_dir)
    table = build_feature_table(aggs, ldap, labels)
    if save_to is not None:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(save_to, index=False)
    return FeatureBuild(table, logs, aggs, reports, ldap, labels)
