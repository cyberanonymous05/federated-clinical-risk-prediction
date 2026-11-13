"""Federated model-class blending (post hoc; DEVIATIONS_FROM_PLAN.md row 38; protocol fixed beforehand). Components: pooled logistic regression (exactly federable by Newton iterations),
the HiPerC-FL network and the federated tree ensemble (fedgbm.py). Blend rules fitted on validation predictions of each split only: mean of logits, validation-selected component, and a
logistic stacker (intercept + 3 weights; the Newton sums are additive over clients, as in hical.py).
  python blend.py make | stats      make writes preds/<task>_main/blend_<rule>_s<seed>.npz ; stats writes results/blend_patient_level.csv, blend_calibration.csv"""
import os, sys, json
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.environ.get("HIPERC_RESULTS", os.path.join(HERE, "..", "results"))
COMP = ("logreg", "hiperc", "fedgbm"); RULES = ("mean", "select", "stack")
SEEDS = {"T1": list(range(20)), "T2": [0, 1, 2], "T3": [0, 1, 2], "T4": list(range(10)), "T5": list(range(10)), "T6": [0]}
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))); sg = lambda z: 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def stack_fit(Z, y, iters=25):
    X = np.column_stack([np.ones(len(Z)), Z]); w = np.array([0.0] + [1.0 / Z.shape[1]] * Z.shape[1])
    for _ in range(iters):
        mu = sg(X @ w); W = mu * (1 - mu) + 1e-12; step = np.linalg.solve(X.T @ (X * W[:, None]) + 1e-6 * np.eye(len(w)), X.T @ (y - mu)); w = w + step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w


def make():
    from metrics import auroc
    for task in ("T1", "T2", "T3", "T4", "T5", "T6"):
        od = os.path.join(RES, "preds", f"{task}_main"); n = 0
        for s in SEEDS[task]:
            fs = {c: os.path.join(od, f"{c}_s{s}.npz") for c in COMP}
            if not all(os.path.exists(f) for f in fs.values()):
                continue
            z = {c: np.load(f) for c, f in fs.items()}; y, yv = z["hiperc"]["y"].astype(float), z["hiperc"]["yv"].astype(float)
            assert all(np.array_equal(z[c]["y"].astype(float), y) and np.array_equal(z[c]["yv"].astype(float), yv) for c in COMP), (task, s)
            Zv = np.column_stack([lg(z[c]["pv"].astype(float)) for c in COMP]); Zt = np.column_stack([lg(z[c]["p"].astype(float)) for c in COMP])
            va = [auroc(yv, Zv[:, j]) for j in range(len(COMP))]; j = int(np.argmax(va)); w = stack_fit(Zv, yv)
            out = {"mean": (sg(Zv.mean(1)), sg(Zt.mean(1))), "select": (sg(Zv[:, j]), sg(Zt[:, j])), "stack": (sg(w[0] + Zv @ w[1:]), sg(w[0] + Zt @ w[1:]))}
            for r, (pv, pt) in out.items():
                np.savez_compressed(os.path.join(od, f"blend_{r}_s{s}.npz"), y=y.astype(np.int8), p=pt.astype(np.float32), yv=yv.astype(np.int8), pv=pv.astype(np.float32), thr=0.0, selected=COMP[j], w=w)
            n += 1
        print(task, n, "splits written")


def stats():
    from stats_patient import patient_scores, fast_delong
    from metrics import holm, ece, calib_slope_intercept
    from scipy import stats as st
    rows = []; cal = []
    for task in ("T1", "T2", "T3", "T4", "T5", "T6"):
        sd = SEEDS[task]; specs = list(COMP) + [f"blend_{r}" for r in RULES]; P = {sp: patient_scores(task, sp, sd, "first") for sp in specs}; base = P["hiperc"][0]
        for sp in specs:
            assert np.array_equal(P[sp][0], base), (task, sp)
        y = P["hiperc"][2]
        for r in RULES:
            b = f"blend_{r}"; a, c = fast_delong(P[b][1][None, :], y); se = float(np.sqrt(c[0, 0])); rows.append(dict(task=task, rule=r, comparator="_self", auroc=float(a[0]), lo=float(a[0] - 1.96 * se), hi=float(a[0] + 1.96 * se), n=len(y), events=int(y.sum())))
            res = []
            for sp in COMP:
                au, cov = fast_delong(np.vstack([P[b][1], P[sp][1]]), y); dl = au[0] - au[1]; se = float(np.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]))
                res.append(dict(task=task, rule=r, comparator=sp, auroc=float(au[1]), diff=float(dl), lo=float(dl - 1.96 * se), hi=float(dl + 1.96 * se), p=float(2 * st.norm.sf(abs(dl / se))) if se > 0 else 1.0, n=len(y), events=int(y.sum())))
            for rr, hh in zip(res, holm([x["p"] for x in res])):
                rr["p_holm"] = float(hh)
            rows += res
        for sp in specs:
            p = P[sp][1]; sl, ic = calib_slope_intercept(y.astype(int), p); cal.append(dict(task=task, spec=sp, brier=float(np.mean((p - y) ** 2)), ece=float(ece(y.astype(int), p)), slope=sl, intercept=ic, auroc=float(fast_delong(p[None, :], y)[0][0])))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "blend_patient_level.csv"), index=False); pd.DataFrame(cal).to_csv(os.path.join(RES, "blend_calibration.csv"), index=False)
    d = pd.DataFrame(rows); print(d[(d.rule == "mean") & (d.comparator != "_self")][["task", "comparator", "diff", "lo", "hi", "p_holm"]].round(4).to_string()); print(pd.DataFrame(cal).round(4).to_string())


if __name__ == "__main__":
    {"make": make, "stats": stats}[sys.argv[1]]()
