"""Can a simple heterogeneity diagnostic, computed on training and validation data before any federated training, tell where personal models will help? (post hoc)
Three diagnostics were fixed before looking at their relation to the outcome (first split of every task, hospital = the environment of the task):
  D1 spread of the outcome rate between hospitals (standard deviation of hospital prevalence, percentage points)
  D2 covariate shift: macro one-versus-rest AUROC of a ridge logistic regression that predicts the hospital from the predictors (fitted on training rows, scored on validation rows; 0.5 = hospitals indistinguishable)
  D3 local-versus-pooled gap: mean over hospitals (weighted by validation size) of AUROC on that hospital's validation rows of a ridge logistic regression fitted on its own training rows minus the pooled one
Outcome (realised benefit of personalisation, mean over splits): G1 = HiPerC-FL minus HiPerC-FL with the global model only; G2 = best of Ditto, FedAvg with fine-tuning and HiPerC-FL minus FedAvg.
Six tasks, so only rank correlations are reported, descriptively.   Writes results/heterogeneity.json.   Usage: python heterogeneity_rule.py"""
import os, json
import numpy as np, pandas as pd
import data
from metrics import auroc
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "..", "results")
sg = lambda z: 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def lr(X, y, l2=10.0, iters=50):
    Xb = np.column_stack([np.ones(len(X)), X]); w = np.zeros(Xb.shape[1]); R = l2 * np.eye(Xb.shape[1]); R[0, 0] = 0
    for _ in range(iters):
        mu = sg(Xb @ w); W = mu * (1 - mu) + 1e-9
        step = np.linalg.solve(Xb.T @ (Xb * W[:, None]) + R, Xb.T @ (y - mu) - R @ w); w = w + step
        if np.max(np.abs(step)) < 1e-7:
            break
    return lambda Z: sg(np.column_stack([np.ones(len(Z)), Z]) @ w)


out = {}
for task in ["T1", "T2", "T3", "T4", "T5", "T6"]:
    fed, _, _ = data.build(task, 0); H = sorted({c["hosp"] for c in fed})
    cat = lambda hs, k: np.concatenate([c[k].numpy() for c in fed if c["hosp"] in hs])
    Xtr, ytr, Xva, yva = cat(H, "Xtr"), cat(H, "ytr").astype(int), cat(H, "Xva"), cat(H, "yva").astype(int)
    htr = np.concatenate([np.full(len(c["ytr"]), c["hosp"]) for c in fed]); hva = np.concatenate([np.full(len(c["yva"]), c["hosp"]) for c in fed])
    prev = [ytr[htr == h].mean() for h in H]; d1 = 100 * float(np.std(prev))
    d2 = float(np.mean([auroc((hva == h).astype(int), lr(Xtr, (htr == h).astype(int))(Xva)) for h in H]))
    pooled = lr(Xtr, ytr); gaps, wts = [], []
    for h in H:
        mtr, mva = htr == h, hva == h
        if yva[mva].sum() < 5 or (1 - yva[mva]).sum() < 5 or ytr[mtr].sum() < 5 or (1 - ytr[mtr]).sum() < 5:
            continue
        gaps.append(auroc(yva[mva], lr(Xtr[mtr], ytr[mtr])(Xva[mva])) - auroc(yva[mva], pooled(Xva[mva]))); wts.append(mva.sum())
    d3 = float(np.average(gaps, weights=wts))
    r = pd.read_csv(os.path.join(RES, f"runs_{task}_main.csv")); m = r.groupby("spec").auroc.mean()
    ft = pd.read_csv(os.path.join(RES, f"runs_{task}_ft.csv")).groupby("spec").auroc.mean(); m = pd.concat([m, ft[~ft.index.isin(m.index)]])
    g1 = float(m["hiperc"] - m["hiperc_global"]); g2 = float(max(m["ditto"], m["fedavg_ft"], m["hiperc"]) - m["fedavg"])
    out[task] = dict(hospitals=len(H), D1_prevalence_sd_pp=round(d1, 2), D2_hospital_identifiability=round(d2, 3), D3_local_minus_pooled=round(d3, 4), G1_personal_models=round(g1, 4), G2_best_personalised=round(g2, 4))
    print(task, out[task], flush=True)
T = list(out)
for d in ("D1_prevalence_sd_pp", "D2_hospital_identifiability", "D3_local_minus_pooled"):
    for g in ("G1_personal_models", "G2_best_personalised"):
        rho = spearmanr([out[t][d] for t in T], [out[t][g] for t in T]); out.setdefault("spearman", {})[f"{d} vs {g}"] = dict(rho=round(float(rho[0]), 2), p=round(float(rho[1]), 3))
json.dump(out, open(os.path.join(RES, "heterogeneity.json"), "w"), indent=1); print(json.dumps(out["spearman"], indent=1))
