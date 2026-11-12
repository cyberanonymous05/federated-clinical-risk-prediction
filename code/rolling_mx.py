"""T6 pseudo-prospective replay (post hoc, added after the main results).
Two analyses on the Mexican COVID-19 registry sample, both strictly forward in time:
 (1) FROZEN: the models of the primary design (trained on admissions to 2021-03-31, selected on April-June 2021) are scored, without any update, on consecutive calendar
     quarters of the test period. This is what a silent trial of a frozen model would show: discrimination, observed and predicted mortality, calibration slope per quarter.
 (2) ROLLING ORIGIN: the same models are retrained at three later origins (train to 2020-12-31 / 2021-06-30 / 2021-12-31, validation the next quarter, test the following
     six months), with the configurations selected for the primary design (no re-tuning), 5 seeds. Patient-level DeLong comparisons on seed 0.
Writes results/frozen_quarters_T6.csv, results/rolling_T6.csv, results/rolling_T6_delong.csv, results/rolling_T6.json.   Usage: python rolling_mx.py"""
import os, sys, json, time
import numpy as np, pandas as pd
from multiprocessing import Pool
import data, fl
from metrics import full_metrics, youden_threshold, auroc, holm
from stats_patient import fast_delong
from settings import SET
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.environ.get("HIPERC_RESULTS", os.path.join(HERE, "..", "results"))
ORIGINS = {"A": ("2020-12-31", "2021-03-31", "2021-09-30"), "B": ("2021-06-30", "2021-09-30", "2022-03-31"), "C": ("2021-12-31", "2022-03-31", "2022-09-30")}
SPECS = ["logreg", "centralized", "fedavg", "ditto", "hiperc"]
sg = lambda z: 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def newton_lr(X, y, l2=1e-3, iters=60):
    Xb = np.column_stack([np.ones(len(X)), X]); w = np.zeros(Xb.shape[1]); R = l2 * np.eye(Xb.shape[1]); R[0, 0] = 0
    for _ in range(iters):
        mu = sg(Xb @ w); W = mu * (1 - mu) + 1e-9
        step = np.linalg.solve(Xb.T @ (Xb * W[:, None]) + R, Xb.T @ (y - mu) - R @ w); w = w + step
        if np.max(np.abs(step)) < 1e-8:
            break
    return lambda Z: sg(np.column_stack([np.ones(len(Z)), Z]) @ w)


def job(a):
    org, spec, seed = a
    fed, _, _ = data.build_mx(seed, origin=ORIGINS[org]); d = fed[0]["Xtr"].shape[1]
    f_saved = os.path.join(RES, "preds", f"T6_roll_{org}", f"{spec}_s{seed}.npz")
    if os.path.exists(f_saved):                                   # resume: predictions of a finished job are reused
        z = np.load(f_saved); y = z["y"].astype(int); p = z["p"].astype(float)
        return dict(origin=org, spec=spec, seed=seed, clients=len(fed), mean_pred=float(np.mean(p)), obs=float(y.mean()), deaths=int(y.sum()), **full_metrics(y, p, youden_threshold(z["yv"].astype(int), z["pv"].astype(float))))
    cfg = {} if spec == "clinical_lr" else {k: v for k, v in json.load(open(os.path.join(RES, "configs_T6.json")))["best"][spec].items() if not k.startswith("_")}
    if spec == "clinical_lr":
        Xtr = np.concatenate([c["Xtr"].numpy() for c in fed])[:, :2]; ytr = np.concatenate([c["ytr"].numpy() for c in fed]); f = newton_lr(Xtr, ytr)
        pv = np.concatenate([f(c["Xva"].numpy()[:, :2]) for c in fed]); p = np.concatenate([f(c["Xte"].numpy()[:, :2]) for c in fed]); r = dict(best_round=0, total_wan=0)
    else:
        r = fl.run_method(spec, fed, d, cfg, seed, SET["T6"]["rounds"], SET["T6"]["eval_every"]); pv = np.concatenate(r["val"]); p = np.concatenate(r["test"])
    yv = np.concatenate([c["yva"].numpy() for c in fed]); y = np.concatenate([c["yte"].numpy() for c in fed]).astype(int)
    out = os.path.join(RES, "preds", f"T6_roll_{org}"); os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, f"{spec}_s{seed}.npz"), y=y.astype(np.int8), p=p.astype(np.float32), yv=yv.astype(np.int8), pv=pv.astype(np.float32))
    return dict(origin=org, spec=spec, seed=seed, clients=len(fed), mean_pred=float(np.mean(p)), obs=float(y.mean()), deaths=int(y.sum()), **full_metrics(y, p, youden_threshold(yv, pv)))


def frozen():
    """Primary-design models scored on consecutive quarters of the test period (no update)."""
    fed, _, _ = data.build_mx(0); df = pd.read_csv(os.path.join(HERE, "..", "data", "mx_processed.csv.gz"), usecols=["adm"])
    ite = np.concatenate([c["idx_te"] for c in fed]); q = pd.PeriodIndex(pd.to_datetime(df.adm.values[ite]), freq="Q").astype(str); q = np.where(q >= "2022Q4", "2022Q4+", q)
    rows = []
    for spec in ("logreg", "centralized", "fedavg", "hiperc"):
        for seed in SET["T6"]["seeds"]:
            z = np.load(os.path.join(RES, "preds", "T6_main", f"{spec}_s{seed}.npz")); y = z["y"].astype(int); p = z["p"].astype(float)
            for qq in sorted(set(q)):
                m = q == qq
                if y[m].sum() < 5:
                    continue
                aucs, cov = fast_delong(p[m][None, :], y[m]); se = float(np.sqrt(cov[0, 0]))
                fm = full_metrics(y[m], p[m], 0.5)
                rows.append(dict(spec=spec, seed=seed, quarter=qq, n=int(m.sum()), deaths=int(y[m].sum()), obs=float(y[m].mean()), mean_pred=float(p[m].mean()), auroc=float(aucs[0]),
                                 lo=float(aucs[0] - 1.96 * se), hi=float(aucs[0] + 1.96 * se), slope=fm["cal_slope"], intercept=fm["cal_int"], ece=fm["ece"]))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, "frozen_quarters_T6.csv"), index=False); return df


def delong_origins():
    rows = []
    for org in ORIGINS:
        P = {}
        for sp in SPECS + ["clinical_lr"]:
            z = np.load(os.path.join(RES, "preds", f"T6_roll_{org}", f"{sp}_s0.npz")); P[sp] = z["p"].astype(float); y = z["y"].astype(int)
        comps = ["logreg", "centralized", "fedavg", "clinical_lr"]; res = []
        for c in comps:
            aucs, cov = fast_delong(np.vstack([P["hiperc"], P[c]]), y); dlt = aucs[0] - aucs[1]; se = float(np.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]))
            res.append(dict(origin=org, comparator=c, hiperc=float(aucs[0]), other=float(aucs[1]), diff=float(dlt), lo=float(dlt - 1.96 * se), hi=float(dlt + 1.96 * se),
                            p=float(2 * stats.norm.sf(abs(dlt / se))) if se > 0 else 1.0))
        adj = holm([r["p"] for r in res])
        for r, a in zip(res, adj):
            r["p_holm"] = float(a); rows.append(r)
        aucs, cov = fast_delong(P["hiperc"][None, :], y); rows.append(dict(origin=org, comparator="_hiperc_alone", hiperc=float(aucs[0]), other=np.nan, diff=np.nan, lo=float(aucs[0] - 1.96 * np.sqrt(cov[0, 0])),
                                                                              hi=float(aucs[0] + 1.96 * np.sqrt(cov[0, 0])), p=np.nan, p_holm=np.nan))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, "rolling_T6_delong.csv"), index=False); return df


def main():
    t0 = time.time(); fq = frozen(); print("frozen quarters done"); print(fq[fq.seed == 0].pivot_table(index="quarter", columns="spec", values="auroc").round(3).to_string(), flush=True)
    jobs = [(o, s, k) for o in ORIGINS for s in SPECS + ["clinical_lr"] for k in SET["T6"]["seeds"]]
    with Pool(9) as pl:
        rows = pl.map(job, jobs, chunksize=1)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, "rolling_T6.csv"), index=False)
    print("rolling done %.0fs" % (time.time() - t0)); print(df.pivot_table(index="spec", columns="origin", values="auroc", aggfunc="mean").round(3).to_string())
    dl = delong_origins(); print(dl.round(4).to_string())
    json.dump(dict(origins=ORIGINS), open(os.path.join(RES, "rolling_T6.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
