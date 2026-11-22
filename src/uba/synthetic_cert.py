"""Synthetic log generator that mimics the CERT Insider Threat Test Dataset r4.2.

Why this exists
---------------
The real CERT r4.2 release (CMU SEI / ExactData, ~16 GB unzipped) is the target
dataset for this project. Until it is downloaded, this module writes a *small*
dataset with the **same files, columns, date format and answer-file layout** so
that every downstream step (loading, cleaning, features, labels, models) runs
unchanged when ``DATA_DIR`` is pointed at the real data.

Files written (identical names/columns to r4.2)::

    <out>/logon.csv        id,date,user,pc,activity                 (Logon/Logoff)
    <out>/device.csv       id,date,user,pc,activity                 (Connect/Disconnect)
    <out>/file.csv         id,date,user,pc,filename,content         (copies to removable media)
    <out>/email.csv        id,date,user,pc,to,cc,bcc,from,size,attachments,content
    <out>/http.csv         id,date,user,pc,url,content
    <out>/psychometric.csv employee_name,user_id,O,C,E,A,N
    <out>/LDAP/YYYY-MM.csv employee_name,user_id,email,role,business_unit,
                           functional_unit,department,team,supervisor
    <out>/answers/insiders.csv               dataset,scenario,details,user,start,end
    <out>/answers/r4.2-<s>/r4.2-<s>-<USER>.csv   malicious events, one per line,
                                                  prefixed with the log type

Insider scenarios follow the three r4.2 scenario descriptions:

1. A user who never used removable drives starts logging in after hours, uses a
   thumb drive and uploads data to wikileaks.org, then leaves the organisation.
2. A user browses job sites, contacts a competitor, and in the final weeks copies
   data to a thumb drive at a much higher rate than before, then leaves.
3. A disgruntled system administrator downloads a keylogger, copies it to the
   supervisor's machine via USB, then uses the supervisor's identity to send an
   alarming mass e-mail, and leaves.

Benign "confounders" (on-call staff working late, heavy-but-legitimate USB users,
people casually looking at job sites, normal leavers) are injected on purpose so
the task is not trivially separable.
"""
from __future__ import annotations

import csv
import string
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

DATE_FMT = "%m/%d/%Y %H:%M:%S"
COMPANY_DOMAIN = "dtaa.com"  # the fictional company used by CERT

# role, functional unit, department, sampling weight
ROLES = [
    ("Salesman", "5 - SalesAndMarketing", "2 - Sales", 0.24),
    ("ComputerProgrammer", "2 - ResearchAndEngineering", "3 - SoftwareManagement", 0.22),
    ("ProductionLineWorker", "3 - Manufacturing", "1 - Assembly", 0.18),
    ("Technician", "3 - Manufacturing", "4 - Quality", 0.12),
    ("Manager", "1 - Administration", "5 - Executive", 0.10),
    ("ITAdmin", "4 - InformationTechnology", "6 - ITOperations", 0.08),
    ("Secretary", "1 - Administration", "7 - Office", 0.06),
]

FIRST = ["Aaron", "Bella", "Carlos", "Dina", "Ethan", "Farah", "George", "Hana", "Ivan", "Jana",
         "Karim", "Lina", "Mark", "Nour", "Omar", "Paula", "Quinn", "Rana", "Samir", "Tara",
         "Umar", "Vera", "Walid", "Xena", "Yara", "Zane", "Adam", "Basma", "Cyrus", "Dalia"]
LAST = ["Abbott", "Barker", "Cohen", "Dawson", "Ellis", "Fischer", "Grant", "Hughes", "Irwin",
        "Jensen", "Keller", "Lawson", "Mercer", "Nolan", "Ortiz", "Porter", "Quincy", "Reyes",
        "Stone", "Turner", "Underwood", "Vaughn", "Walsh", "Young", "Zimmer", "Hassan", "Saleh"]

BENIGN_SITES = ["google.com", "bing.com", "yahoo.com", "cnn.com", "bbc.co.uk", "espn.com",
                "weather.com", "amazon.com", "ebay.com", "wikipedia.org", "nytimes.com",
                "reddit.com", "youtube.com", "facebook.com", "twitter.com", "intranet.dtaa.com",
                "msn.com", "stackoverflow.com", "github.com", "office.com", "dtaa.com"]
JOB_SITES = ["indeed.com", "monster.com", "careerbuilder.com", "simplyhired.com",
             "glassdoor.com", "jobhuntersbible.com", "linkedin.com/jobs"]
LEAK_SITES = ["wikileaks.org"]
CLOUD_SITES = ["dropbox.com", "drive.google.com", "box.com", "mega.nz"]
HACK_SITES = ["keylogger-download.net", "spectorsoft.com", "refog.com", "actualkeylogger.com"]
EXTERNAL_EMAIL_DOMAINS = ["gmail.com", "yahoo.com", "hotmail.com", "aol.com", "comcast.net"]
COMPETITOR_DOMAINS = ["rivaldefense.example", "apexsystems.example"]  # fictional

WORDS = ["meeting", "report", "budget", "project", "schedule", "design", "test", "review",
         "invoice", "customer", "order", "update", "plan", "contract", "quality", "release",
         "server", "network", "backup", "training", "policy", "lunch", "travel", "status"]
FILE_EXT = [".doc", ".pdf", ".txt", ".jpg", ".zip", ".exe"]
FILE_EXT_P = [0.36, 0.22, 0.16, 0.14, 0.09, 0.03]


@dataclass
class UserProfile:
    user_id: str
    name: str
    email: str
    role: str
    functional_unit: str
    department: str
    team: str
    supervisor: str
    pc: str
    start_h: float
    end_h: float
    usb_rate: float
    file_rate: float
    email_rate: float
    ext_frac: float
    http_rate: float
    weekend_p: float
    after_hours_p: float
    job_p: float
    leave_day: int | None = None
    scenario: int | None = None
    window: tuple[int, int] | None = None  # inclusive day indices of malicious activity
    step: int = 0  # progress through a multi-step scenario (scenario 3)


class SyntheticCERT:
    """Generate a CERT-r4.2-shaped dataset. Deterministic for a given seed."""

    def __init__(self, n_users: int = 200, n_days: int = 270, start: str = "2010-01-04",
                 n_insiders: tuple[int, int, int] = (4, 4, 3), seed: int = 42):
        self.n_users = n_users
        self.n_days = n_days
        self.start = datetime.fromisoformat(start)
        self.n_insiders = n_insiders
        self.rng = np.random.default_rng(seed)
        self.rows: dict[str, list] = {k: [] for k in ["logon", "device", "file", "email", "http"]}
        self.mal: dict[str, list] = {k: [] for k in self.rows}
        self.users: list[UserProfile] = []

    # ------------------------------------------------------------------ users
    def _make_users(self) -> None:
        rng = self.rng
        used_ids: set[str] = set()
        role_p = np.array([r[3] for r in ROLES])
        role_p = role_p / role_p.sum()
        pcs = rng.choice(np.arange(1000, 9999), size=self.n_users + 40, replace=False)
        for i in range(self.n_users):
            first, mid, last = rng.choice(FIRST), rng.choice(FIRST), rng.choice(LAST)
            while True:
                uid = f"{first[0]}{mid[0]}{last[0]}{rng.integers(1000, 9999)}".upper()
                if uid not in used_ids:
                    used_ids.add(uid)
                    break
            role, fu, dept, _ = ROLES[rng.choice(len(ROLES), p=role_p)]
            is_usb_user = rng.random() < 0.30
            heavy_usb = is_usb_user and rng.random() < 0.20
            on_call = role in ("ITAdmin", "Manager") and rng.random() < 0.35 or rng.random() < 0.04
            self.users.append(UserProfile(
                user_id=uid,
                name=f"{first} {mid} {last}",
                email=f"{first}.{last}.{uid[-4:]}@{COMPANY_DOMAIN}",
                role=role, functional_unit=fu, department=dept,
                team=f"{rng.integers(1, 9)} - Team{rng.integers(1, 6)}",
                supervisor="",
                pc=f"PC-{pcs[i]}",
                start_h=float(rng.normal(8.2, 0.6)),
                end_h=float(rng.normal(17.1, 0.6)),
                usb_rate=float(rng.uniform(1.5, 3.0) if heavy_usb else (rng.uniform(0.2, 1.0) if is_usb_user else 0.0)),
                file_rate=float(rng.uniform(1.0, 3.0)),
                email_rate=float(rng.uniform(4, 14) * (1.5 if role in ("Salesman", "Manager") else 1.0)),
                ext_frac=float(rng.uniform(0.25, 0.5) if role == "Salesman" else rng.uniform(0.03, 0.2)),
                http_rate=float(rng.uniform(6, 18)),
                weekend_p=float(0.20 if rng.random() < 0.08 else 0.02),
                after_hours_p=float(rng.uniform(0.12, 0.30) if on_call else rng.uniform(0.0, 0.03)),
                job_p=float(rng.uniform(0.04, 0.12) if rng.random() < 0.10 else 0.003),
            ))
        self.shared_pcs = [f"PC-{p}" for p in pcs[self.n_users:]]
        # supervisors: managers supervise everybody else; managers report to a random manager
        managers = [u for u in self.users if u.role == "Manager"] or self.users[:1]
        for u in self.users:
            boss = managers[rng.integers(len(managers))]
            u.supervisor = boss.name if boss is not u else managers[0].name

    def _assign_insiders(self) -> None:
        rng = self.rng
        n1, n2, n3 = self.n_insiders
        admins = [u for u in self.users if u.role == "ITAdmin"]
        others = [u for u in self.users if u.role not in ("ITAdmin", "Manager")]
        if len(admins) < n3:
            raise ValueError("Not enough ITAdmin users for scenario 3; raise n_users.")
        s3 = list(rng.choice(admins, size=n3, replace=False))
        pool = [u for u in others if u not in s3]
        picks = list(rng.choice(pool, size=n1 + n2, replace=False))
        s1, s2 = picks[:n1], picks[n1:]
        # spread insider "end" days evenly over the timeline, interleaving scenarios,
        # so every scenario appears both before and after a 70/30 time split.
        order = []
        for k in range(max(n1, n2, n3)):
            for group, sc in ((s2, 2), (s1, 1), (s3, 3)):
                if k < len(group):
                    order.append((group[k], sc))
        ends = np.linspace(70, self.n_days - 8, len(order)).astype(int)
        for (u, sc), end in zip(order, ends):
            length = {1: rng.integers(7, 11), 2: rng.integers(45, 60), 3: 6}[sc]
            u.scenario = sc
            u.window = (int(end - length + 1), int(end))
            u.leave_day = int(end + 1 + rng.integers(0, 3))
            if sc == 1:
                u.usb_rate = 0.0       # scenario 1: never used removable media before
                u.after_hours_p = 0.0  # and never worked after hours before
        # a few ordinary leavers so "leaving" alone is not a giveaway
        normal = [u for u in self.users if u.scenario is None]
        for u in rng.choice(normal, size=max(1, self.n_users // 25), replace=False):
            u.leave_day = int(rng.integers(60, self.n_days))

    # ---------------------------------------------------------------- helpers
    def _add(self, kind: str, row: tuple, mal: bool = False) -> None:
        self.rows[kind].append(row)
        self.mal[kind].append(mal)

    def _rand_times(self, sessions, n):
        if n <= 0:
            return []
        durs = np.array([(e - s).total_seconds() for s, e, _ in sessions])
        idx = self.rng.choice(len(sessions), size=n, p=durs / durs.sum())
        out = []
        for i in idx:
            s, e, pc = sessions[i]
            out.append((s + timedelta(seconds=float(self.rng.uniform(60, max(61, durs[i] - 60)))), pc))
        return out

    def _words(self, k=4):
        return " ".join(self.rng.choice(WORDS, size=k))

    def _colleague_emails(self, k):
        idx = self.rng.choice(len(self.users), size=k, replace=False)
        return [self.users[i].email for i in idx]

    def _email(self, u, t, pc, to, frm=None, attachments=0, size=None, mal=False, cc="", bcc=""):
        size = int(size if size is not None else self.rng.lognormal(10, 0.6))
        self._add("email", (t, u.user_id, pc, ";".join(to), cc, bcc, frm or u.email, size,
                            int(attachments), self._words(6)), mal)

    def _http(self, u, t, pc, site, mal=False):
        path = "".join(self.rng.choice(list(string.ascii_lowercase), size=6))
        self._add("http", (t, u.user_id, pc, f"http://{site}/{path}", self._words(5)), mal)

    def _usb_session(self, u, t, pc, n_files, mal=False, exts=None, names=None):
        dur = timedelta(minutes=float(self.rng.uniform(4, 40)))
        self._add("device", (t, u.user_id, pc, "Connect"), mal)
        for j in range(n_files):
            ft = t + dur * float(self.rng.uniform(0.05, 0.95))
            if names is not None:
                fname = names[j % len(names)]
            else:
                ext = self.rng.choice(exts or FILE_EXT, p=None if exts else FILE_EXT_P)
                fname = "".join(self.rng.choice(list(string.ascii_uppercase + string.digits), size=8)) + ext
            self._add("file", (ft, u.user_id, pc, fname, self._words(5)), mal)
        self._add("device", (t + dur, u.user_id, pc, "Disconnect"), mal)

    # ------------------------------------------------------------ normal day
    def _normal_day(self, u: UserProfile, date0: datetime, weekend: bool):
        rng = self.rng
        if weekend:
            on = date0 + timedelta(hours=float(rng.uniform(9, 14)))
            off = on + timedelta(hours=float(rng.uniform(1, 4)))
            scale = 0.4
        else:
            on = date0 + timedelta(hours=float(np.clip(rng.normal(u.start_h, 0.35), 6.0, 11)))
            off = date0 + timedelta(hours=float(np.clip(rng.normal(u.end_h, 0.45), 13, 19.5)))
            scale = 1.0
        self._add("logon", (on, u.user_id, u.pc, "Logon"))
        self._add("logon", (off, u.user_id, u.pc, "Logoff"))
        sessions = [(on, off, u.pc)]
        if rng.random() < u.after_hours_p:  # legitimate late work / on-call
            s = date0 + timedelta(hours=float(rng.uniform(19.5, 23.0)))
            e = s + timedelta(minutes=float(rng.uniform(15, 90)))
            self._add("logon", (s, u.user_id, u.pc, "Logon"))
            self._add("logon", (e, u.user_id, u.pc, "Logoff"))
            sessions.append((s, e, u.pc))
        if rng.random() < 0.03:  # shared / meeting-room PC
            pc = self.shared_pcs[rng.integers(len(self.shared_pcs))]
            s = on + (off - on) * float(rng.uniform(0.2, 0.7))
            e = s + timedelta(minutes=float(rng.uniform(10, 60)))
            self._add("logon", (s, u.user_id, pc, "Logon"))
            self._add("logon", (e, u.user_id, pc, "Logoff"))
            sessions.append((s, e, pc))

        for t, pc in self._rand_times(sessions, rng.poisson(u.email_rate * scale)):
            n_to = int(rng.integers(1, 4))
            if rng.random() < u.ext_frac:
                to = [f"{self._words(1)}{rng.integers(10, 99)}@{rng.choice(EXTERNAL_EMAIL_DOMAINS)}"]
                to += self._colleague_emails(n_to - 1)
            else:
                to = self._colleague_emails(n_to)
            self._email(u, t, pc, to, attachments=rng.poisson(0.3))
        for t, pc in self._rand_times(sessions, rng.poisson(u.http_rate * scale)):
            r = rng.random()
            site = rng.choice(CLOUD_SITES) if r < 0.02 else rng.choice(BENIGN_SITES)
            self._http(u, t, pc, site)
        if rng.random() < u.job_p:  # benign curiosity about job sites
            for t, pc in self._rand_times(sessions, int(rng.integers(1, 3))):
                self._http(u, t, pc, rng.choice(JOB_SITES))
        if u.usb_rate > 0:
            for t, pc in self._rand_times(sessions, rng.poisson(u.usb_rate * scale)):
                self._usb_session(u, t, pc, int(rng.poisson(u.file_rate)))
        return sessions

    # ------------------------------------------------------- insider overlays
    def _insider_day(self, u: UserProfile, d: int, date0: datetime, sessions):
        rng = self.rng
        w0, w1 = u.window
        if u.scenario == 1:
            if rng.random() < 0.85 or d == w1:
                s = date0 + timedelta(hours=float(rng.uniform(21.5, 23.5)))
                e = s + timedelta(minutes=float(rng.uniform(30, 120)))
                self._add("logon", (s, u.user_id, u.pc, "Logon"), True)
                self._add("logon", (e, u.user_id, u.pc, "Logoff"), True)
                for _ in range(int(rng.integers(1, 3))):
                    t = s + (e - s) * float(rng.uniform(0.05, 0.6))
                    self._usb_session(u, t, u.pc, int(rng.poisson(4) + 2), mal=True,
                                      exts=[".doc", ".pdf", ".zip"])
                for _ in range(int(rng.integers(1, 4))):
                    self._http(u, s + (e - s) * float(rng.uniform(0.3, 0.95)), u.pc, "wikileaks.org", mal=True)
        elif u.scenario == 2:
            if rng.random() < 0.6:
                for t, pc in self._rand_times(sessions, int(rng.poisson(3) + 1)):
                    self._http(u, t, pc, rng.choice(JOB_SITES), mal=True)
            if rng.random() < 0.12:
                t, pc = self._rand_times(sessions, 1)[0]
                self._email(u, t, pc, [f"hr@{rng.choice(COMPETITOR_DOMAINS)}"], attachments=1,
                            size=rng.lognormal(12, 0.5), mal=True)
            if d >= w1 - 11 and rng.random() < 0.75:  # data theft in the final ~2 weeks
                for t, pc in self._rand_times(sessions, int(rng.poisson(2) + 1)):
                    self._usb_session(u, t, pc, int(rng.poisson(9) + 4), mal=True,
                                      exts=[".doc", ".pdf", ".zip", ".txt"])
        elif u.scenario == 3:
            boss = next((v for v in self.users if v.name == u.supervisor and v is not u), None)
            boss_pc = boss.pc if boss else self.shared_pcs[0]
            boss_mail = boss.email if boss else f"supervisor@{COMPANY_DOMAIN}"
            step = u.step  # steps run on consecutive *working* days of the window
            u.step += 1
            if step == 0:  # download keylogger, put it on a thumb drive
                t, pc = self._rand_times(sessions, 1)[0]
                for k in range(int(rng.integers(2, 4))):
                    self._http(u, t + timedelta(minutes=2 * k), pc, rng.choice(HACK_SITES), mal=True)
                self._usb_session(u, t + timedelta(minutes=15), pc, 1, mal=True, names=["keylogger.exe"])
            elif step == 1:  # install it on the supervisor's machine after hours
                s = date0 + timedelta(hours=float(rng.uniform(19.5, 21.5)))
                e = s + timedelta(minutes=float(rng.uniform(10, 25)))
                self._add("logon", (s, u.user_id, boss_pc, "Logon"), True)
                self._usb_session(u, s + timedelta(minutes=2), boss_pc, 1, mal=True, names=["keylogger.exe"])
                self._add("logon", (e, u.user_id, boss_pc, "Logoff"), True)
            elif step == 2:  # use stolen credentials: alarming mass e-mail as the supervisor
                s = date0 + timedelta(hours=float(rng.uniform(19.5, 22.0)))
                e = s + timedelta(minutes=float(rng.uniform(10, 30)))
                self._add("logon", (s, u.user_id, boss_pc, "Logon"), True)
                to = self._colleague_emails(min(60, len(self.users) - 1))
                self._email(u, s + timedelta(minutes=5), boss_pc, to, frm=boss_mail, mal=True)
                self._add("logon", (e, u.user_id, boss_pc, "Logoff"), True)

    # ---------------------------------------------------------------- driver
    def generate(self) -> "SyntheticCERT":
        self._make_users()
        self._assign_insiders()
        rng = self.rng
        for d in range(self.n_days):
            date0 = self.start + timedelta(days=d)
            weekend = date0.weekday() >= 5
            for u in self.users:
                if u.leave_day is not None and d >= u.leave_day:
                    continue
                in_window = u.window is not None and u.window[0] <= d <= u.window[1]
                works = rng.random() < (u.weekend_p if weekend else 0.96)
                if in_window and not weekend:
                    works = True
                if not works:
                    continue
                sessions = self._normal_day(u, date0, weekend)
                if in_window:
                    self._insider_day(u, d, date0, sessions)
        return self

    # ------------------------------------------------------------------ write
    @staticmethod
    def _ids(rng, n):
        alphabet = np.array(list(string.ascii_uppercase + string.digits))
        raw = rng.choice(alphabet, size=(n, 20))
        return ["{" + "".join(r[:4]) + "-" + "".join(r[4:12]) + "-" + "".join(r[12:]) + "}" for r in raw]

    def write(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        (out / "LDAP").mkdir(parents=True, exist_ok=True)
        (out / "answers").mkdir(parents=True, exist_ok=True)
        cols = {
            "logon": ["date", "user", "pc", "activity"],
            "device": ["date", "user", "pc", "activity"],
            "file": ["date", "user", "pc", "filename", "content"],
            "email": ["date", "user", "pc", "to", "cc", "bcc", "from", "size", "attachments", "content"],
            "http": ["date", "user", "pc", "url", "content"],
        }
        malicious_rows: list[tuple[str, list]] = []
        for kind, c in cols.items():
            df = pd.DataFrame(self.rows[kind], columns=c)
            df["_mal"] = self.mal[kind]
            df = df.sort_values("date", kind="stable").reset_index(drop=True)
            df.insert(0, "id", self._ids(self.rng, len(df)))
            df["date"] = df["date"].dt.strftime(DATE_FMT)
            for rec in df[df["_mal"]].drop(columns="_mal").itertuples(index=False):
                malicious_rows.append((kind, list(rec)))
            df.drop(columns="_mal").to_csv(out / f"{kind}.csv", index=False)

        # LDAP monthly snapshots (leavers disappear from later months, as in r4.2)
        months = pd.period_range(self.start, self.start + timedelta(days=self.n_days - 1), freq="M")
        for m in months:
            first_day = (m.start_time.to_pydatetime() - self.start).days
            snap = [u for u in self.users if u.leave_day is None or u.leave_day > max(first_day, 0)]
            pd.DataFrame([{
                "employee_name": u.name, "user_id": u.user_id, "email": u.email, "role": u.role,
                "business_unit": 1, "functional_unit": u.functional_unit, "department": u.department,
                "team": u.team, "supervisor": u.supervisor,
            } for u in snap]).to_csv(out / "LDAP" / f"{m.strftime('%Y-%m')}.csv", index=False)

        pd.DataFrame([{"employee_name": u.name, "user_id": u.user_id,
                       **{k: int(self.rng.integers(10, 50)) for k in "OCEAN"}}
                      for u in self.users]).to_csv(out / "psychometric.csv", index=False)

        # answers: insiders.csv + one event file per insider (r4.2 layout)
        by_user: dict[str, list] = {}
        for kind, rec in malicious_rows:
            by_user.setdefault(rec[2], []).append((kind, rec))
        ins_rows = []
        for u in [u for u in self.users if u.scenario is not None]:
            events = by_user.get(u.user_id, [])
            if not events:
                continue
            name = f"r4.2-{u.scenario}-{u.user_id}.csv"
            folder = out / "answers" / f"r4.2-{u.scenario}"
            folder.mkdir(parents=True, exist_ok=True)
            events.sort(key=lambda kr: datetime.strptime(kr[1][1], DATE_FMT))
            with open(folder / name, "w", newline="") as fh:
                w = csv.writer(fh)
                for kind, rec in events:
                    w.writerow([kind, *rec])
            ins_rows.append({"dataset": 4.2, "scenario": u.scenario, "details": name, "user": u.user_id,
                             "start": events[0][1][1], "end": events[-1][1][1]})
        pd.DataFrame(ins_rows).to_csv(out / "answers" / "insiders.csv", index=False)
        return out


def generate_synthetic_cert(out_dir: str | Path, n_users: int = 200, n_days: int = 270,
                            seed: int = 42, overwrite: bool = False) -> Path:
    """Generate (or reuse) a synthetic CERT-r4.2-shaped dataset in ``out_dir``."""
    out = Path(out_dir)
    if (out / "logon.csv").exists() and (out / "answers" / "insiders.csv").exists() and not overwrite:
        return out
    return SyntheticCERT(n_users=n_users, n_days=n_days, seed=seed).generate().write(out)


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="data/raw/synthetic_r4.2")
    ap.add_argument("--users", type=int, default=200)
    ap.add_argument("--days", type=int, default=270)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    p = generate_synthetic_cert(args.out, args.users, args.days, args.seed, overwrite=True)
    print(f"Synthetic CERT-style dataset written to {p.resolve()}")
