"""Independent re-implementation of the pooled logistic-regression headline numbers (no torch, no code from fl.py/data.py/metrics.py).
(a) For every task: arrays dumped by leakage_audit.py -> scikit-learn logistic regression (C chosen on validation) -> AUROC by the Mann-Whitney rank formula
    (scipy), compared with the pooled logistic regression of the main runs (seed 0) and with sklearn's roc_auc_score.
(b) For T6, the whole path from the processed registry table: own period rule from admission dates, own client filter, own standardisation, own fit.
Writes results/audit_independent.json.   Usage: python -I audit_independent.py   (needs numpy, pandas, scipy, scikit-learn)"""
import os, json
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "..", "results"); TMP = os.path.join(HERE, "..", "data", "audit_tmp")


def auc_rank(y, p):
    r = rankdata(p); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fit_select(Xtr, ytr, Xva, yva):
    best = None
    for C in (0.01, 0.1, 1.0, 10.0):
        m = LogisticRegression(C=C, max_iter=2000).fit(Xtr, ytr); a = roc_auc_score(yva, m.predict_proba(Xva)[:, 1])
        if best is None or a > best[0]:
            best = (a, m, C)
    return best


out = {}
for task in ("T1", "T2", "T3", "T4", "T5", "T6"):
    f = os.path.join(TMP, f"arrays_{task}.npz")
    if not os.path.exists(f):
        continue
    z = np.load(f); a, m, C = fit_select(z["Xtr"], z["ytr"], z["Xva"], z["yva"]); p = m.predict_proba(z["Xte"])[:, 1]
    ref = pd.read_csv(os.path.join(RES, f"runs_{task}_main.csv")); ref = ref[(ref.spec == "logreg") & (ref.seed == 0)].auroc.iloc[0]
    out[task] = dict(auroc_rank=round(auc_rank(z["yte"], p), 4), auroc_sklearn=round(float(roc_auc_score(z["yte"], p)), 4), main_run_logreg_seed0=round(float(ref), 4),
                     abs_diff=round(abs(auc_rank(z["yte"], p) - float(ref)), 4), C=C)
    print(task, out[task], flush=True)

# (a2) saved HiPerC-FL and FedAvg predictions: AUROC by the rank formula against the value in the run table
for task in ("T1", "T2", "T3", "T4", "T5", "T6"):
    ref = pd.read_csv(os.path.join(RES, f"runs_{task}_main.csv")); diffs = []
    for spec in ("hiperc", "fedavg"):
        f = os.path.join(RES, "preds", f"{task}_main", f"{spec}_s0.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f); v = auc_rank(z["y"].astype(int), z["p"].astype(float)); r = ref[(ref.spec == spec) & (ref.seed == 0)].auroc.iloc[0]; diffs.append(abs(v - float(r)))
    out[task]["saved_predictions_max_abs_diff_rank_vs_table"] = round(float(max(diffs)), 5); print(task, "saved predictions", out[task]["saved_predictions_max_abs_diff_rank_vs_table"])

# (b) T6 from the processed registry table
df = pd.read_csv(os.path.join(HERE, "..", "data", "mx_processed.csv.gz")); adm = pd.to_datetime(df["adm"])
per = np.where(adm <= "2021-03-31", "train", np.where(adm <= "2021-06-30", "val", "test"))
feats = ["age", "sex", "preg", "diabetes", "epoc", "asma", "inmusupr", "hipertension", "otra_com", "cardiovascular", "obesidad", "renal_cronica", "tabaquismo"]
keep = np.zeros(len(df), bool)
for (s, st), g in df.groupby(["sector", "state"]):
    ix = g.index.values; tr = ix[per[ix] == "train"]; va = ix[per[ix] == "val"]; te = ix[per[ix] == "test"]
    if len(tr) >= 200 and len(va) >= 50 and len(te) >= 50 and df.y.values[tr].sum() >= 5:
        keep[ix] = True
d = df[keep].reset_index(drop=True); pr = per[keep]; X = d[feats].values.astype(float); y = d.y.values
mu, sd = X[pr == "train", 0].mean(), X[pr == "train", 0].std(); X[:, 0] = (X[:, 0] - mu) / sd
a, m, C = fit_select(X[pr == "train"], y[pr == "train"], X[pr == "val"], y[pr == "val"]); p = m.predict_proba(X[pr == "test"])[:, 1]; yt = y[pr == "test"]
ref = pd.read_csv(os.path.join(RES, "runs_T6_main.csv")); ref = ref[(ref.spec == "logreg") & (ref.seed == 0)].auroc.iloc[0]
age_sex = LogisticRegression(C=10, max_iter=2000).fit(X[pr == "train"][:, :2], y[pr == "train"]); a2 = auc_rank(yt, age_sex.predict_proba(X[pr == "test"][:, :2])[:, 1])
out["T6_from_processed_table"] = dict(clients_rows=int(keep.sum()), n_test=int((pr == "test").sum()), deaths_test=int(yt.sum()), auroc_rank=round(auc_rank(yt, p), 4),
                                      main_run_logreg_seed0=round(float(ref), 4), abs_diff=round(abs(auc_rank(yt, p) - float(ref)), 4), age_sex_only=round(a2, 4), C=C,
                                      max_abs_standardised_age_train_mean=round(float(abs(X[pr == "train", 0].mean())), 10))
print("T6 independent", out["T6_from_processed_table"])
json.dump(out, open(os.path.join(RES, "audit_independent.json"), "w"), indent=1)
