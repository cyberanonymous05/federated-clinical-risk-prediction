"""Federated gradient-boosted trees by histogram aggregation (post hoc). Own implementation, numpy only.
Protocol (horizontal federation, no raw records leave a client):
  1. bin edges: every client sends per-feature quantiles of its training data; the server averages them weighted by client size (32 bins);
  2. per tree and level: every client sends, for each current node, feature and bin, the sum of gradients and Hessians of its own rows; the server adds them (optionally
     after a hospital server has added its clients' histograms, which is the two-tier variant), chooses the best split of every node and broadcasts it; clients route their rows;
  3. leaf values from the summed gradient and Hessian of the leaf; clients update their own scores.
Because histogram sums are additive, the model equals the model trained on the pooled rows with the same bin edges (checked by `equivalence`). Level-wise trees of depth 3 or 4,
learning rate 0.1, L2 weight 1, minimum Hessian 1; depth (3 or 4) and number of trees (100 or 200) are chosen by validation AUROC. Traffic counts histogram payloads
(2 values x 4 bytes per bin) uploaded by each client; the broadcast of splits is negligible.
  python fedgbm.py equivalence | run | stats       Writes results/runs_<task>_fedgbm.csv, preds/<task>_main/fedgbm_s<seed>.npz, results/fedgbm_patient_level.csv"""
import os, sys, json, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.environ.get("HIPERC_RESULTS", os.path.join(HERE, "..", "results"))
B, LR, LAM, MCW = 32, 0.1, 1.0, 1.0
TASKS = os.environ.get("GBM_TASKS", "T1 T2 T3 T4 T5 T6").split()
SEEDS = {"T1": list(range(20)), "T2": [0, 1, 2], "T3": [0, 1, 2], "T4": list(range(10)), "T5": list(range(10)), "T6": [0]}
sg = lambda z: 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def federated_edges(parts):
    qs = np.linspace(0, 1, B + 1)[1:-1]; N = sum(len(x) for x in parts)
    return sum(len(x) * np.quantile(x, qs, axis=0) for x in parts) / N


def binarize(X, edges):
    out = np.empty(X.shape, np.int16)
    for j in range(X.shape[1]):
        out[:, j] = np.searchsorted(edges[:, j], X[:, j], side="left")
    return out


def grow_tree(Xb, g, h, cid, K, depth):
    n, d = Xb.shape; node = np.zeros(n, np.int64); feats, thr = [], []; ar = np.arange(d)[None, :]; gw, hw = np.repeat(g, d), np.repeat(h, d)
    for level in range(depth):
        nn = 2 ** level; idx = ((((cid * nn + node) * d)[:, None] + ar) * B + Xb).ravel(); size = K * nn * d * B
        Hg = np.bincount(idx, weights=gw, minlength=size).reshape(K, nn, d, B); Hh = np.bincount(idx, weights=hw, minlength=size).reshape(K, nn, d, B)   # one histogram pair per client
        Hg, Hh = Hg.sum(0), Hh.sum(0)                                                                                                                    # the server adds them
        GL, HL = np.cumsum(Hg, axis=2), np.cumsum(Hh, axis=2); G, H = GL[:, :, -1:], HL[:, :, -1:]; GR, HR = G - GL, H - HL
        gain = GL ** 2 / (HL + LAM) + GR ** 2 / (HR + LAM) - G ** 2 / (H + LAM); gain = np.where((HL >= MCW) & (HR >= MCW), gain, -np.inf); gain[:, :, -1] = -np.inf
        flat = gain.reshape(nn, -1); best = flat.argmax(1); bg = flat[np.arange(nn), best]; f, t = best // B, best % B; none = ~np.isfinite(bg) | (bg <= 1e-9); f[none], t[none] = 0, B
        feats.append(f); thr.append(t); node = node * 2 + (Xb[np.arange(n), f[node]] > t[node])
    nl = 2 ** depth; Gl = np.bincount(node, weights=g, minlength=nl); Hl = np.bincount(node, weights=h, minlength=nl)
    return feats, thr, -Gl / (Hl + LAM), node


def route(Xb, feats, thr, leaf):
    node = np.zeros(len(Xb), np.int64); ar = np.arange(len(Xb))
    for f, t in zip(feats, thr):
        node = node * 2 + (Xb[ar, f[node]] > t[node])
    return leaf[node]


def fit(Xtr, ytr, cid, K, depth, n_trees, evals):
    """evals: dict name -> binned matrix; returns dict name -> {n_trees: raw scores} at the checkpoints in n_trees (a tuple)."""
    f = np.full(len(ytr), np.log(ytr.mean() / (1 - ytr.mean()))); f0 = f[0]; out = {k: {} for k in evals}; F = {k: np.full(len(v), f0) for k, v in evals.items()}
    for m in range(1, max(n_trees) + 1):
        p = sg(f); g, h = p - ytr, np.maximum(p * (1 - p), 1e-6); feats, thr, leaf, node = grow_tree(Xtr, g, h, cid, K, depth); f = f + LR * leaf[node]
        for k, X in evals.items():
            F[k] = F[k] + LR * route(X, feats, thr, leaf)
            if m in n_trees:
                out[k][m] = F[k].copy()
        if m in n_trees:
            out.setdefault("_train", {})[m] = f.copy()
    return out


def traffic(K_hosp, K_clients, depth, trees, d):
    per_client = trees * (2 ** depth - 1) * d * B * 2 * 4                         # bytes uploaded by one client over the whole run
    return dict(flat_MB=K_clients * per_client / 1e6, two_tier_MB=K_hosp * per_client / 1e6, per_client_MB=per_client / 1e6)


def job(a):
    import data
    from metrics import full_metrics, youden_threshold, auroc
    task, seed = a; fed, _, _ = data.build(task, seed); cat = lambda k: np.concatenate([c[k].numpy() for c in fed])
    Xtr, ytr, Xva, yva, Xte, yte = cat("Xtr"), cat("ytr").astype(int), cat("Xva"), cat("yva").astype(int), cat("Xte"), cat("yte").astype(int)
    cid = np.concatenate([np.full(len(c["ytr"]), i) for i, c in enumerate(fed)]); K = len(fed); Kh = len({c["hosp"] for c in fed}); d = Xtr.shape[1]
    edges = federated_edges([c["Xtr"].numpy() for c in fed]); Xb, Vb, Tb = binarize(Xtr, edges), binarize(Xva, edges), binarize(Xte, edges); best = None
    for depth in (3, 4):
        o = fit(Xb, ytr.astype(float), cid, K, depth, (100, 200), {"va": Vb, "te": Tb})
        for m in (100, 200):
            v = auroc(yva, o["va"][m])
            if best is None or v > best[0]:
                best = (v, depth, m, sg(o["va"][m]), sg(o["te"][m]))
    v, depth, m, pv, pt = best; thr = youden_threshold(yva, pv); od = os.path.join(RES, "preds", f"{task}_main"); os.makedirs(od, exist_ok=True)
    np.savez_compressed(os.path.join(od, f"fedgbm_s{seed}.npz"), y=yte.astype(np.int8), p=pt.astype(np.float32), yv=yva.astype(np.int8), pv=pv.astype(np.float32), thr=thr)
    return dict(task=task, spec="fedgbm", seed=seed, depth=depth, trees=m, val_auc=v, clients=K, hospitals=Kh, **traffic(Kh, K, depth, m, d), **full_metrics(yte, pt, thr))


def equivalence():
    import data
    fed, _, _ = data.build("T1", 0); cat = lambda k: np.concatenate([c[k].numpy() for c in fed]); Xtr, ytr = cat("Xtr"), cat("ytr").astype(float)
    cid = np.concatenate([np.full(len(c["ytr"]), i) for i, c in enumerate(fed)]); edges = federated_edges([Xtr]); Xb = binarize(Xtr, edges)       # same edges for both runs
    a = fit(Xb, ytr, cid, len(fed), 3, (20,), {"x": Xb})["x"][20]; b = fit(Xb, ytr, np.zeros(len(ytr), np.int64), 1, 3, (20,), {"x": Xb})["x"][20]
    from sklearn.metrics import roc_auc_score
    res = dict(max_abs_score_difference_12_clients_vs_pooled=float(np.max(np.abs(a - b))), train_auroc=float(roc_auc_score(ytr, a)))
    e2 = federated_edges([c["Xtr"].numpy() for c in fed]); res["max_abs_edge_difference_federated_vs_pooled_quantiles"] = float(np.max(np.abs(e2 - federated_edges([Xtr]))))
    json.dump(res, open(os.path.join(RES, "fedgbm_equivalence.json"), "w"), indent=1); print(res)


def stats():
    from stats_patient import patient_scores, fast_delong
    from metrics import holm
    from scipy import stats as st
    rows = []
    for task in TASKS:
        sd = SEEDS[task]; P = {sp: patient_scores(task, sp, sd, "first") for sp in ("fedgbm", "gbm", "hiperc", "logreg", "centralized")}; base = P["fedgbm"][0]
        for sp in P:
            assert np.array_equal(P[sp][0], base), (task, sp)
        y = P["fedgbm"][2]; a, c = fast_delong(P["fedgbm"][1][None, :], y); se = float(np.sqrt(c[0, 0])); rows.append(dict(task=task, comparator="_fedgbm", auroc=float(a[0]), lo=float(a[0] - 1.96 * se), hi=float(a[0] + 1.96 * se), n=len(y), events=int(y.sum())))
        res = []
        for sp in ("gbm", "hiperc", "logreg", "centralized"):
            au, cov = fast_delong(np.vstack([P["fedgbm"][1], P[sp][1]]), y); dl = au[0] - au[1]; se = float(np.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]))
            res.append(dict(task=task, comparator=sp, auroc=float(au[1]), diff=float(dl), lo=float(dl - 1.96 * se), hi=float(dl + 1.96 * se), p=float(2 * st.norm.sf(abs(dl / se))) if se > 0 else 1.0, n=len(y), events=int(y.sum())))
        for r, hh in zip(res, holm([r["p"] for r in res])):
            r["p_holm"] = float(hh)
        rows += res
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, "fedgbm_patient_level.csv"), index=False); print(df.round(4).to_string())


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "equivalence":
        equivalence()
    elif mode == "run":
        from multiprocessing import Pool
        jobs = [(t, s) for t in TASKS for s in SEEDS[t]]
        t0 = time.time()
        with Pool(4) as p:
            rows = p.map(job, jobs, chunksize=1)
        df = pd.DataFrame(rows)
        for t in TASKS:
            df[df.task == t].to_csv(os.path.join(RES, f"runs_{t}_fedgbm.csv"), index=False)
        print("done %.0fs" % (time.time() - t0)); print(df.groupby("task")[["auroc", "ece", "cal_slope", "flat_MB", "two_tier_MB", "depth", "trees"]].mean().round(3).to_string())
    elif mode == "stats":
        stats()
