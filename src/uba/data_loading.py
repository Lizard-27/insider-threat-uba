"""Load, clean and label CERT r4.2-format logs (real or synthetic).

Everything here works on the real CERT r4.2 folder layout::

    DATA_DIR/{logon,device,file,email,http}.csv, DATA_DIR/LDAP/*.csv
    ANSWERS_DIR/insiders.csv, ANSWERS_DIR/r4.2-<scenario>/<details>.csv

Large text columns (``content``) are never loaded: they are not needed for the
Phase 1 behavioural features and they are what makes the real files huge.
``http.csv`` (~14 GB in the real release) is read in chunks by
:func:`iter_log_chunks` so it fits in Colab memory.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import pandas as pd

DATE_FMT = "%m/%d/%Y %H:%M:%S"

# Columns we actually use from each log (content columns deliberately excluded)
USECOLS = {
    "logon": ["id", "date", "user", "pc", "activity"],
    "device": ["id", "date", "user", "pc", "activity"],
    "file": ["id", "date", "user", "pc", "filename"],
    "email": ["id", "date", "user", "pc", "to", "cc", "bcc", "from", "size", "attachments"],
    "http": ["id", "date", "user", "pc", "url"],
}
LOG_TYPES = list(USECOLS)


@dataclass
class CleaningReport:
    log: str
    rows_in: int = 0
    duplicate_ids: int = 0
    missing_key_fields: int = 0
    bad_dates: int = 0
    rows_out: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"log": self.log, "rows_in": self.rows_in, "duplicate_ids": self.duplicate_ids,
                "missing_user/date/pc": self.missing_key_fields, "unparseable_dates": self.bad_dates,
                "rows_out": self.rows_out}


def _read(path: Path, usecols, **kw) -> pd.DataFrame | Iterator[pd.DataFrame]:
    header = pd.read_csv(path, nrows=0).columns
    cols = [c for c in usecols if c in header]
    return pd.read_csv(path, usecols=cols, dtype=str, keep_default_na=False, na_values=[""], **kw)


def clean_log(df: pd.DataFrame, log: str) -> tuple[pd.DataFrame, CleaningReport]:
    """Deduplicate, drop rows missing key fields, parse dates, fix dtypes."""
    rep = CleaningReport(log=log, rows_in=len(df))
    n = len(df)
    df = df.drop_duplicates(subset="id")
    rep.duplicate_ids = n - len(df)

    n = len(df)
    df = df.dropna(subset=["date", "user", "pc"])
    rep.missing_key_fields = n - len(df)

    df = df.copy()
    df["user"] = df["user"].str.strip().str.upper()
    df["pc"] = df["pc"].str.strip().str.upper()
    df["date"] = pd.to_datetime(df["date"], format=DATE_FMT, errors="coerce")
    rep.bad_dates = int(df["date"].isna().sum())
    df = df.dropna(subset=["date"])

    if "activity" in df:
        df["activity"] = df["activity"].str.strip().str.title()
    if log == "email":
        for c in ("size", "attachments"):
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        for c in ("to", "cc", "bcc", "from"):
            df[c] = df[c].fillna("").str.lower()
    if log == "http":
        df["url"] = df["url"].fillna("").str.lower()
    if log == "file":
        df["filename"] = df["filename"].fillna("").str.lower()

    df = df.sort_values("date", kind="stable").reset_index(drop=True)
    rep.rows_out = len(df)
    return df, rep


def load_log(data_dir: str | Path, log: str, nrows: int | None = None) -> tuple[pd.DataFrame, CleaningReport]:
    """Load one log fully into memory and clean it. Use for logon/device/file/email."""
    df = _read(Path(data_dir) / f"{log}.csv", USECOLS[log], nrows=nrows)
    return clean_log(df, log)


def iter_log_chunks(data_dir: str | Path, log: str, chunksize: int = 1_000_000,
                    nrows: int | None = None) -> Iterator[tuple[pd.DataFrame, CleaningReport]]:
    """Yield cleaned chunks of a (possibly huge) log, e.g. the real 14 GB http.csv.

    Note: duplicate ``id`` removal is per chunk here; CERT ids are unique so this is
    a sanity guard rather than a requirement.
    """
    for chunk in _read(Path(data_dir) / f"{log}.csv", USECOLS[log], chunksize=chunksize, nrows=nrows):
        yield clean_log(chunk, log)


def load_ldap(data_dir: str | Path) -> pd.DataFrame:
    """Return one row per user with the attributes from their *first* LDAP snapshot.

    Using the first snapshot (what HR knew when the user appeared) avoids leaking
    the future fact "this person later left the company" into the features.
    ``last_month`` is kept for EDA only and is never used as a model feature.
    """
    files = sorted((Path(data_dir) / "LDAP").glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No LDAP/*.csv files under {data_dir}")
    frames = []
    for f in files:
        d = pd.read_csv(f, dtype=str)
        d["month"] = f.stem
        frames.append(d)
    ldap = pd.concat(frames, ignore_index=True)
    ldap["user_id"] = ldap["user_id"].str.strip().str.upper()
    ldap["email"] = ldap["email"].str.strip().str.lower()
    first = ldap.sort_values("month").groupby("user_id", as_index=False).first()
    last = ldap.groupby("user_id")["month"].max().rename("last_month")
    return first.merge(last, on="user_id").rename(columns={"user_id": "user", "month": "first_month"})


def load_insider_answers(answers_dir: str | Path, dataset: str = "4.2") -> pd.DataFrame:
    """Return the insiders table (dataset, scenario, details, user, start, end) for one release."""
    ins = pd.read_csv(Path(answers_dir) / "insiders.csv", dtype=str)
    ins = ins[ins["dataset"].astype(float).round(1) == float(dataset)].copy()
    ins["user"] = ins["user"].str.upper()
    ins["start"] = pd.to_datetime(ins["start"], format=DATE_FMT, errors="coerce")
    ins["end"] = pd.to_datetime(ins["end"], format=DATE_FMT, errors="coerce")
    ins["scenario"] = ins["scenario"].astype(int)
    return ins.reset_index(drop=True)


def malicious_user_days(answers_dir: str | Path, dataset: str = "4.2") -> pd.DataFrame:
    """Return unique (user, day, scenario) triples that contain >= 1 malicious event.

    Labels come from the per-insider answer files, whose lines are the original log
    rows prefixed with the log type, e.g. ``logon,{ID},01/02/2010 23:10:00,ABC1234,PC-1,Logon``.
    If an answer file is missing we fall back to every day in [start, end].
    """
    ins = load_insider_answers(answers_dir, dataset)
    rows = []
    for r in ins.itertuples(index=False):
        f = Path(answers_dir) / f"r{dataset}-{r.scenario}" / r.details
        if f.exists():
            with open(f, newline="", encoding="utf-8", errors="replace") as fh:
                for line in csv.reader(fh):
                    if len(line) < 4:
                        continue
                    ts = pd.to_datetime(line[2], format=DATE_FMT, errors="coerce")
                    if pd.notna(ts):
                        rows.append((line[3].strip().upper(), ts.normalize(), r.scenario))
        else:
            for day in pd.date_range(r.start.normalize(), r.end.normalize(), freq="D"):
                rows.append((r.user, day, r.scenario))
    out = pd.DataFrame(rows, columns=["user", "day", "scenario"]).drop_duplicates(["user", "day"])
    return out.reset_index(drop=True)
