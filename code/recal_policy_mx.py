"""T6 forward replay: does a simple recalibration policy repair the calibration that frozen models lose? (post hoc)
The primary models (seed 0, trained to 2021-03-31) are scored quarter by quarter. Policies (the model itself is never retrained; AUROC is unchanged by an increasing recalibration):
  frozen          no update
  intercept       before each quarter, shift the logit of the predicted risk by the intercept fitted on the previous quarter (April to June 2021 for the first)
  intercept+slope logistic recalibration (intercept and slope on the logit) fitted on the previous quarter
  triggered       as intercept+slope, but refitted only when the previous quarter's calibration slope (before recalibration) lay outside 0.8 to 1.2; otherwise the current recalibration is kept
Writes results/recal_policy_T6.csv and results/recal_policy_T6.json.   Usage: python recal_policy_mx.py"""
import os, json
import numpy as np, pandas as pd
import data
from metrics import auroc, ece, calib_slope_intercept

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "..", "results")
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))); sg = lambda z: 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def fit_int(y, z, iters=50):
    b = 0.0
    for _ in range(iters):
        mu = sg(z + b); step = (y - mu).sum() / max((mu * (1 - mu)).sum(), 1e-9); b += step
        if abs(step) < 1e-9:
            break
    return b


def fit_both(y, z, iters=50):
    X = np.column_stack([np.ones(len(z)), z]); w = np.array([0.0, 1.0])
    for _ in range(iters):
        mu = sg(X @ w); W = mu * (1 - mu) + 1e-12; step = np.linalg.solve(X.T @ (X * W[:, None]) + 1e-8 * np.eye(2), X.T @ (y - mu)); w = w + step
        if np.max(np.abs(step)) < 1e-9:
            break
    return w


fed, _, _ = data.build_mx(0); df = pd.read_csv(os.path.join(HERE, "..", "data", "mx_processed.csv.gz"), usecols=["adm"])
ite = np.concatenate([c["idx_te"] for c in fed]); q = pd.PeriodIndex(pd.to_datetime(df.adm.values[ite]), freq="Q").astype(str); q = np.where(q >= "2022Q4", "2022Q4+", q); Q = sorted(set(q))
rows = []
for spec in ("logreg", "centralized", "fedavg", "hiperc"):
    z0 = np.load(os.path.join(RES, "preds", "T6_main", f"{spec}_s0.npz")); y = z0["y"].astype(int); p = z0["p"].astype(float); prev_y, prev_z = z0["yv"].astype(int), lg(z0["pv"].astype(float))
    state = {"frozen": (0.0, 1.0), "intercept": (0.0, 1.0), "both": (0.0, 1.0), "triggered": (0.0, 1.0)}
    for k, qq in enumerate(Q):
        m = q == qq; yq, zq = y[m], lg(p[m])
        # update the policies using the previous quarter (before scoring this one)
        b = fit_int(prev_y, prev_z); state["intercept"] = (b, 1.0)
        w = fit_both(prev_y, prev_z); state["both"] = (w[0], w[1])
        sl, _ = calib_slope_intercept(prev_y, sg(state["triggered"][0] + state["triggered"][1] * prev_z))   # slope of the previous quarter under the recalibration then in force
        if k == 0 or not (0.8 <= sl <= 1.2):
            state["triggered"] = (w[0], w[1])
        for pol, (a_, s_) in state.items():
            pq = sg(a_ + s_ * zq); slope, icpt = calib_slope_intercept(yq, pq)
            rows.append(dict(spec=spec, policy=pol, quarter=qq, n=int(m.sum()), deaths=int(yq.sum()), obs=float(yq.mean()), mean_pred=float(pq.mean()), auroc=float(auroc(yq, pq)), ece=float(ece(yq, pq)), brier=float(np.mean((pq - yq) ** 2)), slope=slope, intercept=icpt))
        prev_y, prev_z = yq, zq
d = pd.DataFrame(rows); d.to_csv(os.path.join(RES, "recal_policy_T6.csv"), index=False)
w = d.assign(w=d.n).groupby(["spec", "policy"]).apply(lambda g: pd.Series(dict(ece=np.average(g.ece, weights=g.w), brier=np.average(g.brier, weights=g.w), cal_gap=np.average(np.abs(g.mean_pred - g.obs), weights=g.w), auroc=np.average(g.auroc, weights=g.w))), include_groups=False)
w.round(4).to_csv(os.path.join(RES, "recal_policy_T6_summary.csv")); print(w.round(4).to_string())
h = d[d.spec == "hiperc"].pivot_table(index="quarter", columns="policy", values="ece"); print(h.round(3).to_string())
json.dump(dict(summary={f"{a}|{b}": {k: round(float(v), 4) for k, v in r.items()} for (a, b), r in w.iterrows()}), open(os.path.join(RES, "recal_policy_T6.json"), "w"), indent=1)
