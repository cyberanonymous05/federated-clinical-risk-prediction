"""Recalibration analysis for T6 (Mexican COVID-19 registry, forward in time): mortality fell from 13.5% in the training period to 3.8% in the test period, so predicted risks drift upward. What do simple recalibration steps fitted on the later validation period do?
For every split and method, a logistic recalibration (intercept and slope on the logit of the predicted risk) is fitted on VALIDATION predictions only and applied to the
test predictions: (a) one pooled recalibration, (b) one recalibration per health-system sector.
Reported (mean over 10 splits): calibration slope and intercept, expected calibration error, Brier score and net benefit at 2%, 5% and 10% thresholds; AUROC is
reported to show that discrimination is essentially unchanged.  Usage: python recal_sepsis.py"""
import os, json
import numpy as np, pandas as pd
import data
from metrics import auroc, ece, calib_slope_intercept, net_benefit, treat_all_nb
from settings import SET

RES = os.environ.get("HIPERC_RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
SPECS = ["logreg", "lin_hiperc", "centralized", "fedavg", "fedavg_ft", "ditto", "hiperc"]
TH = [0.02, 0.05, 0.10]
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
sg = lambda z: 1 / (1 + np.exp(-z))


def fit(yv, pv, iters=50):
    """Logistic regression of the outcome on the logit of the predicted risk (intercept, slope), fitted by Newton steps (numpy only)."""
    X = np.column_stack([np.ones(len(pv)), lg(pv)]); w = np.array([0.0, 1.0])
    for _ in range(iters):
        mu = sg(X @ w); W = mu * (1 - mu) + 1e-12
        step = np.linalg.solve(X.T @ (X * W[:, None]) + 1e-8 * np.eye(2), X.T @ (yv - mu))
        w = w + step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w[0], w[1]


def metrics(y, p):
    sl, ic = calib_slope_intercept(y, p); nb = net_benefit(y, p, TH)
    return dict(auroc=auroc(y, p), slope=sl, intercept=ic, ece=ece(y, p), brier=float(np.mean((p - y) ** 2)), nb2=nb[0], nb5=nb[1], nb10=nb[2])


rows = []
for seed in SET["T6"]["seeds"]:
    fed, _, _ = data.build_mx(seed); sys_va = np.concatenate([np.full(len(c["yva"]), c["sector"]) for c in fed])
    for sp in SPECS:
        f = os.path.join(RES, "preds", "T6_main", f"{sp}_s{seed}.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f); y = z["y"].astype(int); p = z["p"].astype(float); yv = z["yv"].astype(int); pv = z["pv"].astype(float); st = z["hosp"].astype(int)
        assert len(sys_va) == len(yv)
        a, b = fit(yv, pv); p_pool = sg(a + b * lg(p))
        p_sys = p.copy()
        for s_ in range(5):
            a_, b_ = fit(yv[sys_va == s_], pv[sys_va == s_]); m = st == s_; p_sys[m] = sg(a_ + b_ * lg(p[m]))
        for kind, pp in (("none", p), ("pooled", p_pool), ("per system", p_sys)):
            rows.append(dict(seed=seed, spec=sp, recal=kind, **metrics(y, pp)))
df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, "recal_T6.csv"), index=False)
g = df.groupby(["spec", "recal"]).mean(numeric_only=True).drop(columns="seed")
print(g.round(4).to_string())
fh = df[df.spec == "hiperc"]; y_prev = np.mean(np.load(os.path.join(RES, "preds", "T6_main", "hiperc_s0.npz"))["y"])
print("treat-all NB at", TH, "(split 0):", np.round(treat_all_nb(np.load(os.path.join(RES, "preds", "T6_main", "hiperc_s0.npz"))["y"], TH), 4))
