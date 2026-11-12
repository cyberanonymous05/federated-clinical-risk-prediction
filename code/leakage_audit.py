"""Hidden-flaw audit for all six tasks (added after a leaked registry field was found in T6).
Seed 0 of every task. Checks:
  1. split integrity: no row in two of train/validation/test; patient ids unique (T5); forward-in-time order of admission dates (T6)
  2. scaling fitted on training rows only (continuous training columns have mean 0)
  3. single-predictor screen: AUROC of every predictor alone (orientation from training rows, scored on test rows); anything above 0.85 is flagged
  4. permutation null, pooled logistic regression: training labels permuted, true test labels, AUROC must be about 0.5
  5. permutation null, federated training (FedAvg and HiPerC-FL, 30 rounds, a new permutation and a new network initialisation per draw): training labels
     permuted globally, validation and test labels true; checkpoint selection and the whole federated code path must not produce an AUROC above chance
     (a random function of correlated predictors has an AUROC far from 0.5 in either direction, so only the mean over draws is read)
  6. arrays dumped to data/audit_tmp/ for the independent re-implementation in audit_independent.py (no torch, sklearn only)
Writes results/leakage_audit.json.   Usage: python leakage_audit.py [T1 T2 ...]"""
import os, sys, json, time
import numpy as np, pandas as pd, torch
import data, fl
from metrics import auroc
from settings import SET

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("HIPERC_RESULTS", os.path.join(HERE, "..", "results"))
TMP = os.path.join(HERE, "..", "data", "audit_tmp"); os.makedirs(TMP, exist_ok=True)
TASKS = sys.argv[1:] or ["T1", "T2", "T3", "T4", "T5", "T6"]
sg = lambda z: 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def newton_lr(X, y, l2=1.0, iters=60):
    """Ridge logistic regression by Newton steps (numpy only)."""
    Xb = np.column_stack([np.ones(len(X)), X]); w = np.zeros(Xb.shape[1]); R = l2 * np.eye(Xb.shape[1]); R[0, 0] = 0
    for _ in range(iters):
        mu = sg(Xb @ w); W = mu * (1 - mu) + 1e-9
        step = np.linalg.solve(Xb.T @ (Xb * W[:, None]) + R, Xb.T @ (y - mu) - R @ w)
        w = w + step
        if np.max(np.abs(step)) < 1e-7:
            break
    return lambda Z: sg(np.column_stack([np.ones(len(Z)), Z]) @ w)


def cat(fed, k):
    return torch.cat([c[k] for c in fed]).numpy()


out = json.load(open(os.path.join(RES, "leakage_audit.json"))) if os.path.exists(os.path.join(RES, "leakage_audit.json")) else {}
for task in TASKS:
    t0 = time.time(); fed, _, info = data.build(task, 0); d = fed[0]["Xtr"].shape[1]
    Xtr, ytr, Xva, yva, Xte, yte = (cat(fed, k) for k in ("Xtr", "ytr", "Xva", "yva", "Xte", "yte")); ytr, yva, yte = ytr.astype(int), yva.astype(int), yte.astype(int)
    names = list(info.get("names") or [f"x{j}" for j in range(d)]); rep = dict(n_train=int(len(ytr)), n_val=int(len(yva)), n_test=int(len(yte)), n_features=int(d))
    # 1. split integrity
    itr, iva, ite = (np.concatenate([c[k] for c in fed]) for k in ("idx_tr", "idx_va", "idx_te"))
    rep["overlap_train_val_test"] = [int(len(set(itr) & set(iva))), int(len(set(itr) & set(ite))), int(len(set(iva) & set(ite)))]
    rep["duplicate_indices_within_split"] = int(len(itr) - len(set(itr)) + len(iva) - len(set(iva)) + len(ite) - len(set(ite)))
    if task in ("T5", "T5H24"):
        df = pd.read_csv(os.path.join(HERE, "..", "data", "sepsis_processed.csv")); rep["patient_ids_unique"] = bool(df.pid.is_unique)
    if task == "T6":
        df = pd.read_csv(os.path.join(HERE, "..", "data", "mx_processed.csv.gz")); adm = pd.to_datetime(df.adm)
        rep["latest_train_admission"] = str(adm.values[itr].max())[:10]; rep["earliest_val_admission"] = str(adm.values[iva].min())[:10]
        rep["latest_val_admission"] = str(adm.values[iva].max())[:10]; rep["earliest_test_admission"] = str(adm.values[ite].min())[:10]
        rep["forward_in_time_order_holds"] = bool(adm.values[itr].max() < adm.values[iva].min() and adm.values[iva].max() < adm.values[ite].min())
    # 2. scaling on training rows only: continuous columns (more than 2 distinct values) have training mean 0
    cont = [j for j in range(d) if len(np.unique(Xtr[:, j])) > 2]
    rep["continuous_columns"] = len(cont); rep["train_mean_abs_max_continuous"] = float(np.abs(Xtr[:, cont].mean(0)).max()) if cont else None
    rep["test_mean_abs_max_continuous"] = float(np.abs(Xte[:, cont].mean(0)).max()) if cont else None
    # 3. single-predictor screen
    sp = []
    for j in range(d):
        a_tr = auroc(ytr, Xtr[:, j]); a_te = auroc(yte, Xte[:, j])
        if np.isnan(a_tr) or np.isnan(a_te):
            continue
        sp.append((names[j] if j < len(names) else f"x{j}", a_te if a_tr >= 0.5 else 1 - a_te))
    sp.sort(key=lambda t: -t[1]); rep["single_predictor_top3"] = [(n, round(float(a), 3)) for n, a in sp[:3]]
    rep["single_predictor_max_auroc"] = round(float(sp[0][1]), 3); rep["flagged_above_0.85"] = [n for n, a in sp if a > 0.85]
    # 4. permutation null, pooled logistic regression
    rng = np.random.RandomState(0); null = []
    for _ in range(100):
        f = newton_lr(Xtr, rng.permutation(ytr)); null.append(auroc(yte, f(Xte)))
    rep["perm_null_logreg_auroc"] = dict(mean=round(float(np.mean(null)), 4), min=round(float(np.min(null)), 4), max=round(float(np.max(null)), 4), draws=100)
    rep["real_logreg_auroc_numpy"] = round(float(auroc(yte, newton_lr(Xtr, ytr)(Xte))), 4)
    # 5. permutation null, federated training
    best = json.load(open(os.path.join(RES, f"configs_{task}.json")))["best"]; clean = lambda c: {k: v for k, v in c.items() if not k.startswith("_")}
    draws = 8 if task == "T1" else 5; rep["perm_null_federated"] = {}
    for m in ("fedavg", "hiperc"):
        vals = []
        for k in range(draws):
            pl = ytr[np.random.RandomState(100 + k).permutation(len(ytr))]; o = 0; fedp = []
            for c in fed:
                n = len(c["ytr"]); cc = dict(c); cc["ytr"] = torch.tensor(pl[o:o + n], dtype=c["ytr"].dtype); o += n; fedp.append(cc)
            r = fl.run_method(m, fedp, d, clean(best[m]), k, min(SET[task]["rounds"], 30), 5)
            vals.append(float(auroc(yte, np.concatenate(r["test"]))))
        rep["perm_null_federated"][m] = dict(mean=round(float(np.mean(vals)), 4), min=round(float(np.min(vals)), 4), max=round(float(np.max(vals)), 4), draws=draws)
    # 6. dump for the independent re-implementation
    np.savez_compressed(os.path.join(TMP, f"arrays_{task}.npz"), Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, Xte=Xte, yte=yte)
    rep["seconds"] = round(time.time() - t0); out[task] = rep; print(task, json.dumps(rep), flush=True)
    json.dump(out, open(os.path.join(RES, "leakage_audit.json"), "w"), indent=1)
