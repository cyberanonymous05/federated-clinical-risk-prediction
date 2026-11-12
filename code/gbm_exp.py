"""Pooled gradient-boosted trees as a further (non-federated) reference on all six tasks (post hoc).
Three steps, because scikit-learn's boosting and PyTorch cannot share one process on macOS (segmentation fault):
  python gbm_exp.py dump   (torch)  - the splits of every task and seed to data/audit_tmp/gbm/ (not archived)
  python gbm_exp.py fit    (no torch) - HistGradientBoostingClassifier on the pooled training rows; the configuration (leaf nodes 4, 8 or 16; 100 or 300 iterations, learning rate 0.05)
                            is chosen by validation AUROC within each split; predictions go to results/preds/<task>_main/gbm_s<seed>.npz and runs_<task>_gbm.csv
  python gbm_exp.py stats  (torch)  - patient-level AUROC with DeLong interval and paired differences from HiPerC-FL, pooled logistic regression and the pooled neural network
T6 uses one split (its sample is fixed; boosting is deterministic)."""
import os, sys, json, glob
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.environ.get("HIPERC_RESULTS", os.path.join(HERE, "..", "results")); TMP = os.path.join(HERE, "..", "data", "audit_tmp", "gbm")
TASKS = os.environ.get("GBM_TASKS", "T1 T2 T3 T4 T5 T6").split()          # GBM_TASKS=T1 restricts every mode to one task (used by the smoke test)
MODE = sys.argv[1] if len(sys.argv) > 1 else "dump"


def seeds_of(task):
    from settings import SET
    return [0] if task == "T6" else SET[task]["seeds"]


if MODE == "dump":
    import data
    os.makedirs(TMP, exist_ok=True)
    for task in TASKS:
        for s in seeds_of(task):
            fed, _, _ = data.build(task, s); cat = lambda k: np.concatenate([c[k].numpy() for c in fed])
            np.savez_compressed(os.path.join(TMP, f"{task}_s{s}.npz"), Xtr=cat("Xtr").astype(np.float32), ytr=cat("ytr").astype(np.int8), Xva=cat("Xva").astype(np.float32), yva=cat("yva").astype(np.int8),
                                Xte=cat("Xte").astype(np.float32), yte=cat("yte").astype(np.int8))
        print("dumped", task, flush=True)

if MODE == "fit":
    from multiprocessing import Pool
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from metrics import full_metrics, youden_threshold

    def job(a):
        task, s = a; z = np.load(os.path.join(TMP, f"{task}_s{s}.npz")); best = None; best_model = None
        for leaves in (4, 8, 16):
            m = HistGradientBoostingClassifier(max_leaf_nodes=leaves, learning_rate=0.05, max_iter=300, early_stopping=False, random_state=s).fit(z["Xtr"], z["ytr"])
            for k, pv in enumerate(m.staged_predict_proba(z["Xva"])):
                if k + 1 in (100, 300):
                    v = roc_auc_score(z["yva"], pv[:, 1])
                    if best is None or v > best[0]:
                        best = (v, leaves, k + 1); best_model = m
        v, leaves, it = best
        pv = pt = None
        for k, (a_, b_) in enumerate(zip(best_model.staged_predict_proba(z["Xva"]), best_model.staged_predict_proba(z["Xte"]))):
            if k + 1 == it:
                pv, pt = a_[:, 1], b_[:, 1]; break
        y = z["yte"].astype(int); thr = youden_threshold(z["yva"], pv)
        out = os.path.join(RES, "preds", f"{task}_main"); os.makedirs(out, exist_ok=True)
        np.savez_compressed(os.path.join(out, f"gbm_s{s}.npz"), y=y.astype(np.int8), p=pt.astype(np.float32), yv=z["yva"], pv=pv.astype(np.float32), thr=thr)
        return dict(task=task, spec="gbm", seed=s, leaves=leaves, iters=it, val_auc=v, **full_metrics(y, pt, thr))

    if __name__ == "__main__":
        jobs = [(t, s) for t in TASKS for s in seeds_of(t)]
        with Pool(4) as p:
            rows = p.map(job, jobs, chunksize=1)
        df = pd.DataFrame(rows)
        for t in TASKS:
            df[df.task == t].to_csv(os.path.join(RES, f"runs_{t}_gbm.csv"), index=False)
        print(df.groupby("task")[["auroc", "brier", "ece", "cal_slope"]].mean().round(4).to_string()); print(df.groupby("task")[["leaves", "iters"]].agg(lambda x: x.value_counts().to_dict()).to_string())


if MODE == "dumpx":                       # (torch) splits for zero-shot transport: train on the other hospitals, test on all rows of the held-out one
    import data
    os.makedirs(TMP, exist_ok=True)
    for task, hs, sds in (("T1", range(4), range(20)), ("T5", range(2), range(10)), ("T6", range(5), [0])):
        for h in hs:
            for s in sds:
                fed, ext, _ = data.build_heart(s, holdout=h) if task == "T1" else data.build(task, s, holdout=[h]); cat = lambda k: np.concatenate([c[k].numpy() for c in fed])
                np.savez_compressed(os.path.join(TMP, f"x_{task}_h{h}_s{s}.npz"), Xtr=cat("Xtr").astype(np.float32), ytr=cat("ytr").astype(np.int8), Xva=cat("Xva").astype(np.float32), yva=cat("yva").astype(np.int8),
                                    Xte=ext["X"].numpy().astype(np.float32), yte=ext["y"].numpy().astype(np.int8))
        print("dumped transport", task, flush=True)

if MODE == "fitx":
    from multiprocessing import Pool
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    def jobx(f):
        z = np.load(f); best = None
        for leaves in (4, 8, 16):
            m = HistGradientBoostingClassifier(max_leaf_nodes=leaves, learning_rate=0.05, max_iter=300, early_stopping=False, random_state=0).fit(z["Xtr"], z["ytr"])
            pv = list(m.staged_predict_proba(z["Xva"])); pt = list(m.staged_predict_proba(z["Xte"]))
            for it in (100, 300):
                v = roc_auc_score(z["yva"], pv[it - 1][:, 1])
                if best is None or v > best[0]:
                    best = (v, roc_auc_score(z["yte"], pt[it - 1][:, 1]))
        b = os.path.basename(f)[2:-4].split("_"); return dict(task=b[0], holdout=int(b[1][1:]), seed=int(b[2][1:]), spec="gbm", auroc=best[1])

    if __name__ == "__main__":
        with Pool(4) as p:
            rows = p.map(jobx, sorted(glob.glob(os.path.join(TMP, "x_*.npz"))), chunksize=1)
        df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, "loho_gbm.csv"), index=False)
        cmp = {}
        for t in ("T1", "T5", "T6"):
            ref = pd.read_csv(os.path.join(RES, f"loho_{t}.csv")); g = df[df.task == t]
            cmp[t] = dict(gbm=g.groupby("holdout").auroc.mean().mean(), logreg=ref[ref.spec == "logreg"].groupby("holdout").auroc.mean().mean(), hiperc=ref[ref.spec == "hiperc"].groupby("holdout").auroc.mean().mean(),
                          centralized=ref[ref.spec == "centralized"].groupby("holdout").auroc.mean().mean(),
                          gbm_by=g.groupby("holdout").auroc.mean().round(3).tolist(), logreg_by=ref[ref.spec == "logreg"].groupby("holdout").auroc.mean().round(3).tolist())
        print(json.dumps(cmp, indent=1, default=float)); json.dump(cmp, open(os.path.join(RES, "loho_gbm.json"), "w"), indent=1, default=float)

if MODE == "stats":
    from stats_patient import patient_scores, fast_delong
    from metrics import holm
    from scipy import stats
    rows = []
    for task in TASKS:
        sd = seeds_of(task) if task != "T6" else [0]; P = {}
        for sp in ("gbm", "hiperc", "logreg", "centralized"):
            ids, sc, yy = patient_scores(task, sp, sd if sp == "gbm" else (sd if task != "T6" else [0]), "first"); P[sp] = (ids, sc, yy)
        base = P["gbm"][0]
        for sp in P:
            assert np.array_equal(P[sp][0], base), (task, sp)
        y = P["gbm"][2]; a, c = fast_delong(P["gbm"][1][None, :], y); rows.append(dict(task=task, comparator="_gbm", auroc=float(a[0]), lo=float(a[0] - 1.96 * np.sqrt(c[0, 0])), hi=float(a[0] + 1.96 * np.sqrt(c[0, 0])), n=len(y), events=int(y.sum())))
        res = []
        for sp in ("hiperc", "logreg", "centralized"):
            aucs, cov = fast_delong(np.vstack([P["gbm"][1], P[sp][1]]), y); d = aucs[0] - aucs[1]; se = float(np.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]))
            res.append(dict(task=task, comparator=sp, auroc=float(aucs[1]), diff=float(d), lo=float(d - 1.96 * se), hi=float(d + 1.96 * se), p=float(2 * stats.norm.sf(abs(d / se))) if se > 0 else 1.0, n=len(y), events=int(y.sum())))
        for r, h in zip(res, holm([r["p"] for r in res])):
            r["p_holm"] = float(h)
        rows += res
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, "gbm_patient_level.csv"), index=False); print(df.round(4).to_string())
