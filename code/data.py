"""Data loading and federated partitioning for HiPerC-FL experiments.

Tasks
-----
T1  Coronary artery disease  - UCI Heart Disease, 4 real hospitals (natural federation)
T2  Diabetes                 - CDC BRFSS 2015 (UCI 891), semi-synthetic federation by income band
T3  Hypertension (HighBP)    - same file, same federation

A "Fed" is a list of client dicts. Every client belongs to one hospital/region ("hosp").
Standardisation statistics are computed from the pooled *training* data of all clients using
sums / sums of squares / counts only (what secure aggregation can deliver); no raw rows are pooled.
"""
import os
import numpy as np
import pandas as pd
import torch

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

HEART_COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
              "exang", "oldpeak", "slope", "ca", "thal", "num"]
HEART_SITES = ["cleveland", "hungarian", "switzerland", "va"]
SITE_NAMES = {"cleveland": "Cleveland (US)", "hungarian": "Budapest (HU)",
              "switzerland": "Zurich/Basel (CH)", "va": "Long Beach VA (US)"}


# ---------------------------------------------------------------- helpers
def _stratified_split(y, fr, rng):
    """Split indices into train/val/test with fractions fr=(a,b,c), stratified, safe for tiny classes."""
    tr, va, te = [], [], []
    for c in (0, 1):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        nte = max(1, int(round(fr[2] * n))) if n >= 3 else 0
        nva = max(1, int(round(fr[1] * n))) if n >= 3 else 0
        te += list(idx[:nte]); va += list(idx[nte:nte + nva]); tr += list(idx[nte + nva:])
    return np.array(tr), np.array(va), np.array(te)


def _round_robin(idx, y, k, rng):
    """Assign indices to k clients, class by class, so every client sees the same class mix."""
    parts = [[] for _ in range(k)]
    for c in (0, 1):
        ii = idx[y[idx] == c].copy()
        rng.shuffle(ii)
        for j, v in enumerate(ii):
            parts[j % k].append(v)
    return [np.array(p, dtype=int) for p in parts]


def _to_t(a):
    return torch.tensor(np.asarray(a), dtype=torch.float32)


# ---------------------------------------------------------------- T1: heart
def load_heart_raw():
    out = []
    for h, s in enumerate(HEART_SITES):
        df = pd.read_csv(os.path.join(DATA, f"processed.{s}.data"), header=None, names=HEART_COLS,
                         na_values="?")
        df["hosp"] = h
        out.append(df)
    df = pd.concat(out, ignore_index=True).astype(float)
    # physiologically impossible zeros are missing-value codes
    df.loc[df["chol"] == 0, "chol"] = np.nan
    df.loc[df["trestbps"] == 0, "trestbps"] = np.nan
    df["y"] = (df["num"] > 0).astype(int)
    return df


HEART_NUM = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
HEART_BIN = ["sex", "fbs", "exang"]
HEART_CAT = {"cp": [1, 2, 3, 4], "restecg": [0, 1, 2], "slope": [1, 2, 3], "thal": [3, 6, 7]}
HEART_MISS_IND = ["trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]


def _heart_design(df):
    """Un-standardised design matrix (NaN kept for numeric) + names. One-hot of categoricals."""
    cols, names = [], []
    for c in HEART_NUM + HEART_BIN:
        cols.append(df[c].values); names.append(c)
    for c, levels in HEART_CAT.items():
        for l in levels:
            cols.append(np.where(df[c].isna(), np.nan, (df[c] == l).astype(float)).astype(float))
            names.append(f"{c}={int(l)}")
    for c in HEART_MISS_IND:
        cols.append(df[c].isna().astype(float).values); names.append(f"miss_{c}")
    return np.column_stack(cols).astype(float), names


def build_heart(seed, n_sub=3, holdout=None, drop=()):
    """Federation for T1. Each real hospital is split into n_sub clients (e.g. clinics/wards).
    holdout=h  -> hospital h contributes NO training data; all its rows form the external test set."""
    rng = np.random.RandomState(seed)
    df = load_heart_raw()
    X, names = _heart_design(df)
    if drop:                                   # sensitivity analyses: remove attribute groups by name prefix
        keep = [j for j, n in enumerate(names) if not any(n == d or n.startswith(d + "=") or n == "miss_" + d for d in drop)]
        X = X[:, keep]; names = [names[j] for j in keep]
    y = df["y"].values
    hosp = df["hosp"].values
    sex = df["sex"].values
    age = df["age"].values
    n_ind = len(HEART_MISS_IND)
    cont = [names.index(c) for c in HEART_NUM if c in names]          # standardised columns
    clients, held = [], None
    for h in range(len(HEART_SITES)):
        idx = np.where(hosp == h)[0]
        if holdout is not None and h == holdout:
            held = idx
            continue
        a, b, c = _stratified_split(y[idx], (0.6, 0.2, 0.2), rng)
        trp = _round_robin(idx[a], y, n_sub, rng)
        vap = _round_robin(idx[b], y, n_sub, rng)
        tep = _round_robin(idx[c], y, n_sub, rng)
        for j in range(n_sub):
            clients.append(dict(hosp=h, tr=trp[j], va=vap[j], te=tep[j]))
    # federated standardisation from training rows only (sum, sumsq, count of observed values)
    tr_all = np.concatenate([c["tr"] for c in clients])
    mu = np.zeros(X.shape[1]); sd = np.ones(X.shape[1])
    for j in cont + [names.index(c) for c in HEART_BIN if c in names]:
        v = X[tr_all, j]; v = v[~np.isnan(v)]
        mu[j], sd[j] = v.mean(), max(v.std(), 1e-6)
    Xs = (X - mu) / sd
    Xs = np.nan_to_num(Xs, nan=0.0)                       # mean-impute after standardising + indicators
    fed = []
    for c in clients:
        d = dict(hosp=c["hosp"])
        for s in ("tr", "va", "te"):
            ii = c[s]
            d["X" + s] = _to_t(Xs[ii]); d["y" + s] = _to_t(y[ii])
            d["idx_" + s] = ii
        d["meta_te"] = dict(sex=sex[c["te"]], age=age[c["te"]], hosp=hosp[c["te"]])
        d["meta_va"] = dict(sex=sex[c["va"]], age=age[c["va"]], hosp=hosp[c["va"]])
        fed.append(d)
    ext = None
    if held is not None:
        ext = dict(X=_to_t(Xs[held]), y=_to_t(y[held]), hosp=holdout,
                   meta=dict(sex=sex[held], age=age[held], hosp=hosp[held]))
    info = dict(names=names, n_hosp=len(HEART_SITES) - (holdout is not None), task="T1")
    return fed, ext, info


# ---------------------------------------------------------------- T2/T3: BRFSS
CDC_CONT = ["BMI", "MentHlth", "PhysHlth", "GenHlth", "Age", "Education", "Income"]
INCOME_BAND = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2, 7: 3, 8: 4}
REGION_NAMES = ["Income 1-2", "Income 3-4", "Income 5-6", "Income 7", "Income 8"]


def build_cdc(task, seed, n_per_region=4, alpha=None, frac_data=1.0, regions="band5", holdout=None):
    """T2: target Diabetes_binary. T3: target HighBP (HighBP removed from features).
    5 regions (income bands, natural covariate/prior shift) x n_per_region clients.
    alpha=None : clients inside a region are random (stratified) splits -> only the natural between-region shift.
    alpha=float: additional synthetic label skew inside each region, Dirichlet(alpha) (sensitivity analysis)."""
    rng = np.random.RandomState(seed)
    df = pd.read_csv(os.path.join(DATA, "cdc_diabetes.csv"))
    if frac_data < 1:
        df = df.sample(frac=frac_data, random_state=seed).reset_index(drop=True)
    target = "Diabetes_binary" if task == "T2" else "HighBP"
    y = df[target].values.astype(int)
    feat = [c for c in df.columns if c not in ("ID", target)]
    X = df[feat].values.astype(float)
    region = df["Income"].map(INCOME_BAND).values if regions == "band5" else (df["Income"].values - 1)
    n_reg = 5 if regions == "band5" else 8
    age = df["Age"].values; sex = df["Sex"].values; inc = df["Income"].values
    bmi = df["BMI"].values
    clients = []
    ext = None; remap = {}
    for r in range(n_reg):
        idx = np.where(region == r)[0]
        if holdout is not None and r == holdout:
            ext_idx = idx
            continue
        remap[r] = len(remap)
        while alpha is None:
            parts = _round_robin(idx, y, n_per_region, rng)
            break
        while alpha is not None:
            parts = [[] for _ in range(n_per_region)]
            for c in (0, 1):
                ii = idx[y[idx] == c].copy(); rng.shuffle(ii)
                p = rng.dirichlet(alpha * np.ones(n_per_region))
                cuts = (np.cumsum(p)[:-1] * len(ii)).astype(int)
                for j, s in enumerate(np.split(ii, cuts)):
                    parts[j].append(s)
            parts = [np.concatenate(p) for p in parts]
            if all(len(p) >= 300 and y[p].sum() >= 30 and (1 - y[p]).sum() >= 30 for p in parts):
                break
        for p in parts:
            a, b, c = _stratified_split(y[p], (0.7, 0.15, 0.15), rng)
            clients.append(dict(hosp=remap[r], tr=p[a], va=p[b], te=p[c]))
    tr_all = np.concatenate([c["tr"] for c in clients])
    cont = [feat.index(c) for c in CDC_CONT]
    mu = np.zeros(X.shape[1]); sd = np.ones(X.shape[1])
    for j in cont:
        mu[j], sd[j] = X[tr_all, j].mean(), max(X[tr_all, j].std(), 1e-6)
    Xs = (X - mu) / sd
    fed = []
    for c in clients:
        d = dict(hosp=c["hosp"])
        for s in ("tr", "va", "te"):
            ii = c[s]
            d["X" + s] = _to_t(Xs[ii]); d["y" + s] = _to_t(y[ii]); d["idx_" + s] = ii
        d["meta_te"] = dict(sex=sex[c["te"]], age=age[c["te"]], hosp=c["hosp"] * np.ones(len(c["te"]), dtype=int),
                            income=inc[c["te"]], bmi=bmi[c["te"]])
        d["meta_va"] = dict(sex=sex[c["va"]], age=age[c["va"]], hosp=c["hosp"] * np.ones(len(c["va"]), dtype=int),
                            income=inc[c["va"]], bmi=bmi[c["va"]])
        fed.append(d)
    if holdout is not None:
        ext = dict(X=_to_t(Xs[ext_idx]), y=_to_t(y[ext_idx]), hosp=holdout)
    info = dict(names=feat, n_hosp=len(remap), task=task, raw=dict(X=X, y=y, age=age, sex=sex, region=region,
                                                          feat=feat))
    return fed, ext, info


# ---------------------------------------------------------------- T4: NHANES 2005-2018 (measured data, real temporal shift)
def build_nhanes(seed, holdout=None, n_sub=3):
    """Adult diabetes cohort from 7 NHANES cycles (see nhanes_prep.py). Each survey cycle is an environment ("hospital"),
    split into n_sub clients by class-stratified round-robin. holdout = list of cycle indices that contribute no training data;
    all their rows form the external set (with ext['cycle'])."""
    rng = np.random.RandomState(seed)
    df = pd.read_csv(os.path.join(DATA, "nhanes_processed.csv"))
    race = pd.get_dummies(df["race"].astype(int), prefix="race").astype(float)
    cols = {"age": df.age.values, "sex": df.sex.values.astype(float), "pir": df.pir.values, "bmi": df.bmi.values, "waist": df.waist.values,
            "sbp": df.sbp.values, "dbp": df.dbp.values, "tc": df.tc.values, "hdl": df.hdl.values}
    names = list(cols) + list(race.columns) + ["miss_pir", "miss_bmi_waist", "miss_bp", "miss_lipid"]
    X = np.column_stack([cols[c] for c in cols] + [race[c].values for c in race.columns] +
                        [df.pir.isna().values.astype(float), (df.bmi.isna() | df.waist.isna()).values.astype(float),
                         (df.sbp.isna() | df.dbp.isna()).values.astype(float), (df.tc.isna() | df.hdl.isna()).values.astype(float)])
    y = df.y.values.astype(int); cyc = df.cycle.values
    hold = set(holdout or []); train_cycles = [c for c in range(7) if c not in hold]; remap = {c: i for i, c in enumerate(train_cycles)}
    clients = []
    for c in train_cycles:
        idx = np.where(cyc == c)[0]
        a, b, t = _stratified_split(y[idx], (0.6, 0.2, 0.2), rng)
        trp = _round_robin(idx[a], y, n_sub, rng); vap = _round_robin(idx[b], y, n_sub, rng); tep = _round_robin(idx[t], y, n_sub, rng)
        for j in range(n_sub):
            clients.append(dict(hosp=remap[c], cyc=c, tr=trp[j], va=vap[j], te=tep[j]))
    tr_all = np.concatenate([c["tr"] for c in clients])
    cont = [names.index(c) for c in ("age", "pir", "bmi", "waist", "sbp", "dbp", "tc", "hdl")]
    mu = np.zeros(X.shape[1]); sd = np.ones(X.shape[1])
    for j in cont:
        v = X[tr_all, j]; v = v[~np.isnan(v)]; mu[j], sd[j] = v.mean(), max(v.std(), 1e-6)
    Xs = np.nan_to_num((X - mu) / sd, nan=0.0)
    fed = []
    for c in clients:
        d = dict(hosp=c["hosp"], cycle=c["cyc"])
        for s_ in ("tr", "va", "te"):
            ii = c[s_]; d["X" + s_] = _to_t(Xs[ii]); d["y" + s_] = _to_t(y[ii]); d["idx_" + s_] = ii
        for s_ in ("te", "va"):
            ii = c[s_]; d["meta_" + s_] = dict(sex=df.sex.values[ii], age=df.age.values[ii], hosp=np.full(len(ii), c["cyc"]))
        fed.append(d)
    ext = None
    if hold:
        ii = np.where(np.isin(cyc, list(hold)))[0]
        ext = dict(X=_to_t(Xs[ii]), y=_to_t(y[ii]), cycle=cyc[ii], idx=ii)
    info = dict(names=names, n_hosp=len(train_cycles), task="T4", raw=dict(Xs=Xs, y=y, cyc=cyc, age=df.age.values, sex=df.sex.values))
    return fed, ext, info


def build_sepsis(seed, holdout=None, horizon=None):
    """T5: sepsis during the ICU stay after a 24-hour landmark (PhysioNet/CinC Challenge 2019, two hospital systems; see sepsis_prep.py).
    Hospital = hospital system (0, 1); clients = ICU-unit groups inside each system (unit 1, unit 2, other), so there are six clients.
    Each client is split 60/20/20 stratified by outcome. holdout = list of systems that contribute no training data; all their patients form
    the external set (ext['system'])."""
    rng = np.random.RandomState(seed)
    df = pd.read_csv(os.path.join(DATA, "sepsis_processed.csv"))
    if horizon == 24:                                        # sensitivity: sepsis within 24 h after the landmark, patients with complete follow-up or an event
        df = df[df.elig24 == 1].reset_index(drop=True); df["y"] = df["y24"]
    VARS = [c[:-5] for c in df.columns if c.endswith("_last")]; LABS = VARS[8:]
    cols = {"age": df.age.values, "sex": df.sex.values.astype(float), "hospadm": df.hospadm.values,
            "unit1": np.nan_to_num(df.unit1.values.astype(float)), "unit2": np.nan_to_num(df.unit2.values.astype(float))}
    for v in VARS:
        for k in ("last", "min", "max"):
            cols[f"{v}_{k}"] = df[f"{v}_{k}"].values
    names = list(cols) + [f"obs_{v}" for v in LABS]
    X = np.column_stack([cols[c] for c in cols] + [(df[f"{v}_n"].values > 0).astype(float) for v in LABS])
    y = df.y.values.astype(int); sysid = df.system.values.astype(int)
    unit = np.where(df.unit1.values == 1, 0, np.where(df.unit2.values == 1, 1, 2))
    hold = set(holdout or []); train_sys = [s_ for s_ in (0, 1) if s_ not in hold]; remap = {s_: i for i, s_ in enumerate(train_sys)}
    clients = []
    for s_ in train_sys:
        for u in (0, 1, 2):
            idx = np.where((sysid == s_) & (unit == u))[0]
            if len(idx) == 0:
                continue
            a, b, t = _stratified_split(y[idx], (0.6, 0.2, 0.2), rng)
            clients.append(dict(hosp=remap[s_], sys=s_, tr=idx[a], va=idx[b], te=idx[t]))
    tr_all = np.concatenate([c["tr"] for c in clients])
    nb = len(cols); mu = np.zeros(X.shape[1]); sd = np.ones(X.shape[1])
    cont = [0, 2] + list(range(5, nb))                       # age, hours to ICU admission and all measurements (not sex, unit flags, indicators)
    for j in cont:
        v = X[tr_all, j]; v = v[~np.isnan(v)]
        if len(v):
            mu[j], sd[j] = v.mean(), max(v.std(), 1e-6)
    Xs = np.nan_to_num((X - mu) / sd, nan=0.0)               # missing values are set to the training mean
    Xs[:, cont] = np.clip(Xs[:, cont], -5, 5)                # limit extreme laboratory values
    fed = []
    for c in clients:
        d = dict(hosp=c["hosp"], system=c["sys"])
        for s2 in ("tr", "va", "te"):
            ii = c[s2]; d["X" + s2] = _to_t(Xs[ii]); d["y" + s2] = _to_t(y[ii]); d["idx_" + s2] = ii
        for s2 in ("te", "va"):
            ii = c[s2]; d["meta_" + s2] = dict(sex=df.sex.values[ii], age=df.age.values[ii], hosp=np.full(len(ii), c["sys"]))
        fed.append(d)
    ext = None
    if hold:
        ii = np.where(np.isin(sysid, list(hold)))[0]
        ext = dict(X=_to_t(Xs[ii]), y=_to_t(y[ii]), system=sysid[ii], idx=ii)
    info = dict(names=names, n_hosp=len(train_sys), task="T5", raw=dict(Xs=Xs, y=y, sys=sysid, age=df.age.values, sex=df.sex.values))
    return fed, ext, info


def build_mx(seed, holdout=None, origin=None):
    """T6: 30-day mortality in confirmed adult COVID-19 (Mexican Ministry of Health open registry), forward in time. Hospital = health-system sector (SSA, IMSS, ISSSTE,
    private, other public); clients = (sector, state) cells. Train = admissions up to 2021-03-31, validation = 2021-04-01 to 2021-06-30 (later than training, before the test
    period), test = admissions from 2021-07-01 (Delta and Omicron waves, vaccination). The sample is fixed by mx_prep.py, so seeds change only the learning procedure.
    holdout = list of sectors that contribute no training data; their test-period patients form the external set (ext['sector']).
    origin = (train_end, val_end, test_end) as ISO dates: rolling-origin variant (rolling_mx.py) that re-cuts the same sample by admission date, i.e. train <= train_end, validation <= val_end,
    test <= test_end; rows after test_end are not used. None keeps the primary design above."""
    df = pd.read_csv(os.path.join(DATA, "mx_processed.csv.gz"))
    # hospital admission at registration is NOT used: the registry updates it after the course of the illness (AUROC 0.96 alone, 42.6% vs 1.2% mortality), i.e. it leaks the outcome
    feats = ["age", "sex", "preg", "diabetes", "epoc", "asma", "inmusupr", "hipertension", "otra_com", "cardiovascular", "obesidad", "renal_cronica", "tabaquismo"]
    X = df[feats].values.astype(float); y = df.y.values.astype(int); sec = df.sector.values; per = df.period.values; stt = df.state.values
    if origin is not None:
        adm = pd.to_datetime(df.adm); per = np.where(adm <= origin[0], "train", np.where(adm <= origin[1], "val", np.where(adm <= origin[2], "test", "none")))
    hold = set(holdout or []); train_sec = [s_ for s_ in range(5) if s_ not in hold]; remap = {s_: i for i, s_ in enumerate(train_sec)}
    clients = []
    for s_ in train_sec:
        for st in sorted(set(stt[sec == s_])):
            m = (sec == s_) & (stt == st)
            tr = np.where(m & (per == "train"))[0]; va = np.where(m & (per == "val"))[0]; te = np.where(m & (per == "test"))[0]
            if len(tr) >= 200 and len(va) >= 50 and len(te) >= 50 and y[tr].sum() >= 5:
                clients.append(dict(hosp=remap[s_], sec=s_, tr=tr, va=va, te=te))
    tr_all = np.concatenate([c["tr"] for c in clients]); cont = [feats.index("age")]
    mu = np.zeros(X.shape[1]); sd = np.ones(X.shape[1])
    for j in cont:
        mu[j], sd[j] = X[tr_all, j].mean(), max(X[tr_all, j].std(), 1e-6)
    Xs = (X - mu) / sd
    fed = []
    for c in clients:
        dd = dict(hosp=c["hosp"], sector=c["sec"])
        for s2 in ("tr", "va", "te"):
            ii = c[s2]; dd["X" + s2] = _to_t(Xs[ii]); dd["y" + s2] = _to_t(y[ii]); dd["idx_" + s2] = ii
        for s2 in ("te", "va"):
            ii = c[s2]; dd["meta_" + s2] = dict(sex=X[ii, 1], age=X[ii, 0], hosp=np.full(len(ii), c["sec"]))
        fed.append(dd)
    ext = None
    if hold:
        ii = np.where(np.isin(sec, list(hold)) & (per == "test"))[0]
        ext = dict(X=_to_t(Xs[ii]), y=_to_t(y[ii]), sector=sec[ii], idx=ii)
    info = dict(names=feats, n_hosp=len(train_sec), task="T6", raw=dict(Xs=Xs, y=y, sec=sec, age=X[:, 0], sex=X[:, 1]))
    return fed, ext, info


def build(task, seed, **kw):
    if task == "T1":
        return build_heart(seed, **kw)
    if task == "T4":
        return build_nhanes(seed, **kw)
    if task == "T5":
        return build_sepsis(seed, **kw)
    if task == "T6":
        return build_mx(seed, **kw)
    if task == "T5H24":
        return build_sepsis(seed, horizon=24, **kw)
    return build_cdc(task, seed, **kw)


def describe(fed):
    ntr = sum(len(c["ytr"]) for c in fed); nva = sum(len(c["yva"]) for c in fed)
    nte = sum(len(c["yte"]) for c in fed)
    return dict(clients=len(fed), train=ntr, val=nva, test=nte)
