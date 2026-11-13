"""HiCal: Hierarchical federated Calibration (post hoc; DEVIATIONS_FROM_PLAN.md row 37; protocol fixed in the project protocol written before any result was computed).
A frozen federated model gives a logit z. HiCal recalibrates it with p = sigmoid(a + b z + delta_hospital + delta_client) using only quantities that are sums over clients,
so the server never sees a record and the aggregation can run under secure aggregation:
  1. (a, b): federated Newton on the logistic recalibration model; every client returns a 2-vector gradient and a 2x2 Hessian per round (5 numbers), at most 8 rounds.
  2. hospital offsets then client offsets: from the sums U = sum(y - mu) and I = sum mu(1-mu), raw offset d = U / I; the between-group variance is the DerSimonian-Laird moment
     estimate, computed from the sums  S1 = sum I, S2 = sum I^2, Q = sum I d^2 - (sum I d)^2 / S1;  offset = d * tau2 I / (tau2 I + 1)  (empirical-Bayes shrinkage towards 0).
  3. monitoring (HiCal-T): refit only when the pooled observed/expected ratio of the last quarter leaves 0.80 to 1.25.
Modes (python hical.py <mode>):
  equiv   checks federated Newton against a pooled fit and the message-passing protocol against the vectorised code      -> results/hical_equivalence.json
  t6      T6 forward replay, seven policies x four model families                                                         -> results/hical_T6*.csv, hical_T6_boot.json
  static  one fit on the validation predictions of each split, scored on that split's test set, tasks T1 to T5           -> results/hical_static*.csv"""
import os, sys, json, glob
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.environ.get("HIPERC_RESULTS", os.path.join(HERE, "..", "results"))
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))); sg = lambda z: 1 / (1 + np.exp(-np.clip(z, -30, 30)))
BAND = (0.80, 1.25); SPECS = ("logreg", "centralized", "fedavg", "hiperc", "fedgbm"); POLS = ("frozen", "intercept", "both", "raw", "client", "hical", "hical_t")


def newton2(z, y, cid, K, max_rounds=8):
    """federated Newton for (a, b); per-client sums are formed first (what each client would send) and then added"""
    w = np.array([0.0, 1.0]); r = 0
    for r in range(1, max_rounds + 1):
        mu = sg(w[0] + w[1] * z); res = y - mu; W = mu * (1 - mu) + 1e-12
        g0, g1 = np.bincount(cid, res, K), np.bincount(cid, res * z, K); h00, h01, h11 = np.bincount(cid, W, K), np.bincount(cid, W * z, K), np.bincount(cid, W * z * z, K)
        g = np.array([g0.sum(), g1.sum()]); H = np.array([[h00.sum(), h01.sum()], [h01.sum(), h11.sum()]]); step = np.linalg.solve(H + 1e-8 * np.eye(2), g); w = w + step
        if np.max(np.abs(step)) < 1e-6:
            break
    return w, r


def newton_int(z, y, max_rounds=8):
    b = 0.0
    for r in range(1, max_rounds + 1):
        mu = sg(z + b); step = (y - mu).sum() / max((mu * (1 - mu)).sum(), 1e-9); b += step
        if abs(step) < 1e-6:
            break
    return b, r


def offsets(eta, y, grp, G, shrink=True):
    """raw offsets d = U / I per group and their empirical-Bayes shrinkage towards 0 (moment estimate of the between-group variance from additive sums)"""
    mu = sg(eta); U = np.bincount(grp, y - mu, G); I = np.bincount(grp, mu * (1 - mu), G); ok = I > 1e-9; d = np.zeros(G); d[ok] = U[ok] / I[ok]
    if not shrink or ok.sum() < 2:
        return (d if not shrink else np.zeros(G)), 0.0
    S1, S2 = I[ok].sum(), (I[ok] ** 2).sum(); Q = (I * d * d).sum() - (I * d).sum() ** 2 / S1; tau2 = max(0.0, (Q - (ok.sum() - 1)) / (S1 - S2 / S1)); out = np.zeros(G); out[ok] = d[ok] * tau2 * I[ok] / (tau2 * I[ok] + 1)
    return out, tau2


def fit_policy(pol, z, y, cid, K, hk, H):
    P = dict(a=0.0, b=1.0, dh=np.zeros(H), dk=np.zeros(K), rounds=0, tau_h=0.0, tau_c=0.0)
    if pol == "intercept":
        P["a"], P["rounds"] = newton_int(z, y)
    elif pol in ("both", "raw", "client", "hical"):
        w, P["rounds"] = newton2(z, y, cid, K); P["a"], P["b"] = w; eta = P["a"] + P["b"] * z
        if pol == "hical":
            P["dh"], P["tau_h"] = offsets(eta, y, hk[cid], H); eta = eta + P["dh"][hk[cid]]
        if pol in ("raw", "client", "hical"):
            P["dk"], P["tau_c"] = offsets(eta, y, cid, K, shrink=(pol != "raw"))
    return P


def predict(P, z, cid, hk):
    return sg(P["a"] + P["b"] * z + P["dh"][hk[cid]] + P["dk"][cid])


def uplink_bytes(rounds, hierarchical):
    """bytes sent by one client per update: 5 numbers per Newton round, then (U, I) for the hospital stage and 4 sums for the client stage (8-byte floats)"""
    return 8 * (5 * rounds + (2 + 4 if hierarchical else 4))


# ---------------------------------------------------------------- equivalence
def protocol_messages(parts, hk, H, K):
    """explicit message passing: parts[k] = (z, y) of client k. Clients send only the sums named in the docstring; the server adds them (hospital servers first) and broadcasts scalars."""
    w = np.array([0.0, 1.0])
    for _ in range(8):
        msg = []
        for z, y in parts:
            mu = sg(w[0] + w[1] * z); res = y - mu; W = mu * (1 - mu) + 1e-12; msg.append((res.sum(), (res * z).sum(), W.sum(), (W * z).sum(), (W * z * z).sum()))
        s = np.sum(msg, axis=0); step = np.linalg.solve(np.array([[s[2], s[3]], [s[3], s[4]]]) + 1e-8 * np.eye(2), s[:2]); w = w + step
        if np.max(np.abs(step)) < 1e-6:
            break

    def stage(etas, grp_of_client, G, shrink=True):
        U = np.zeros(G); I = np.zeros(G)
        for k, (z, y) in enumerate(parts):                                                    # each client sends (U, I); the group server adds
            mu = sg(etas[k]); U[grp_of_client[k]] += (y - mu).sum(); I[grp_of_client[k]] += (mu * (1 - mu)).sum()
        d = np.where(I > 1e-9, U / np.maximum(I, 1e-9), 0.0); ok = I > 1e-9; S1, S2 = I[ok].sum(), (I[ok] ** 2).sum(); Q = (I * d * d).sum() - (I * d).sum() ** 2 / S1
        tau2 = max(0.0, (Q - (ok.sum() - 1)) / (S1 - S2 / S1)) if ok.sum() > 1 else 0.0
        return np.where(ok, d * tau2 * I / (tau2 * I + 1), 0.0), tau2
    eta0 = [w[0] + w[1] * z for z, _ in parts]; dh, th = stage(eta0, hk, H)                                       # hospital stage: groups = hospitals
    eta1 = [e + dh[hk[k]] for k, e in enumerate(eta0)]; dk, tc = stage(eta1, np.arange(K), K)                   # client stage: groups = clients
    return w, dh, dk, th, tc


def equiv():
    import data
    fed, _, _ = data.build_mx(0); z0 = np.load(os.path.join(RES, "preds", "T6_main", "hiperc_s0.npz")); yv = z0["yv"].astype(float); zv = lg(z0["pv"].astype(float))
    n = [len(c["yva"]) for c in fed]; cid = np.repeat(np.arange(len(fed)), n); K = len(fed); hk = np.array([c["hosp"] for c in fed]); H = int(hk.max()) + 1; assert len(cid) == len(yv)
    P = fit_policy("hical", zv, yv, cid, K, hk, H)
    X = np.column_stack([np.ones(len(zv)), zv]); w = np.array([0.0, 1.0])                                          # independent pooled fit (as in recal_policy_mx.py)
    for _ in range(50):
        mu = sg(X @ w); W = mu * (1 - mu) + 1e-12; step = np.linalg.solve(X.T @ (X * W[:, None]) + 1e-8 * np.eye(2), X.T @ (yv - mu)); w = w + step
        if np.max(np.abs(step)) < 1e-10:
            break
    off = np.cumsum([0] + n); parts = [(zv[off[k]:off[k + 1]], yv[off[k]:off[k + 1]]) for k in range(K)]; w2, dh, dk, th, tc = protocol_messages(parts, hk, H, K)
    pa = sg(P["a"] + P["b"] * zv + P["dh"][hk[cid]] + P["dk"][cid]); pb = sg(w2[0] + w2[1] * zv + dh[hk[cid]] + dk[cid])
    res = dict(newton_vs_pooled_max_abs_param_diff=float(np.max(np.abs(w - np.array([P["a"], P["b"]])))), newton_rounds=int(P["rounds"]), protocol_vs_vectorised_max_abs_prob_diff=float(np.max(np.abs(pa - pb))),
               tau_h=float(P["tau_h"]), tau_c=float(P["tau_c"]), uplink_bytes_per_client_per_update=uplink_bytes(P["rounds"], True), clients=K, hospitals=H)
    json.dump(res, open(os.path.join(RES, "hical_equivalence.json"), "w"), indent=1); print(res)


# ---------------------------------------------------------------- T6 forward replay
def boot_diff(pa, pb, y, cid, K, kind, B=2000, seed=0):
    """client-clustered bootstrap of the mean per-patient loss difference (a minus b)"""
    if kind == "brier":
        d = (pa - y) ** 2 - (pb - y) ** 2
    else:
        f = lambda p: -(y * np.log(np.clip(p, 1e-9, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-9, 1)))
        d = f(pa) - f(pb)
    s, n = np.bincount(cid, d, K), np.bincount(cid, minlength=K).astype(float); rng = np.random.default_rng(seed); c = rng.multinomial(K, np.ones(K) / K, size=B).astype(float); est = (c @ s) / (c @ n)
    return float(s.sum() / n.sum()), float(np.quantile(est, 0.025)), float(np.quantile(est, 0.975))


def t6():
    import data
    from metrics import auroc, ece, calib_slope_intercept
    fed, _, _ = data.build_mx(0); K = len(fed); hk = np.array([c["hosp"] for c in fed]); H = int(hk.max()) + 1
    df = pd.read_csv(os.path.join(HERE, "..", "data", "mx_processed.csv.gz"), usecols=["adm", "y"]); ite = np.concatenate([c["idx_te"] for c in fed]); cte = np.concatenate([np.full(len(c["idx_te"]), i) for i, c in enumerate(fed)])
    cva = np.concatenate([np.full(len(c["yva"]), i) for i, c in enumerate(fed)]); q = pd.PeriodIndex(pd.to_datetime(df.adm.values[ite]), freq="Q").astype(str); q = np.where(q >= "2022Q4", "2022Q4+", q); Q = sorted(set(q))
    rows = []; allp = {}; yall = None
    for spec in SPECS:
        z0 = np.load(os.path.join(RES, "preds", "T6_main", f"{spec}_s0.npz")); y = z0["y"].astype(int); p = z0["p"].astype(float); assert np.array_equal(y, df.y.values[ite]); prev = (lg(z0["pv"].astype(float)), z0["yv"].astype(float), cva)
        yall = y; P = {pol: dict(a=0.0, b=1.0, dh=np.zeros(H), dk=np.zeros(K), rounds=0) for pol in POLS}; fits = {pol: 0 for pol in POLS}; rnds = {pol: [] for pol in POLS}; allp[spec] = {pol: np.zeros(len(y)) for pol in POLS}
        for k, qq in enumerate(Q):
            m = q == qq; zq, yq, cq = lg(p[m]), y[m].astype(float), cte[m]
            for pol in POLS:
                if pol == "frozen":
                    continue
                if pol == "hical_t":
                    pp = predict(P[pol], prev[0], prev[2], hk); oe = prev[1].sum() / max(pp.sum(), 1e-9)
                    if k > 0 and BAND[0] <= oe <= BAND[1]:
                        continue
                    P[pol] = fit_policy("hical", prev[0], prev[1], prev[2], K, hk, H)
                else:
                    P[pol] = fit_policy(pol, prev[0], prev[1], prev[2], K, hk, H)
                fits[pol] += 1; rnds[pol].append(P[pol]["rounds"])
            for pol in POLS:
                pq = predict(P[pol], zq, cq, hk); allp[spec][pol][m] = pq; sl, ic = calib_slope_intercept(yq, pq)
                rows.append(dict(spec=spec, policy=pol, quarter=qq, n=int(m.sum()), deaths=int(yq.sum()), obs=float(yq.mean()), mean_pred=float(pq.mean()), auroc=float(auroc(yq, pq)), ece=float(ece(yq, pq)), brier=float(np.mean((pq - yq) ** 2)), slope=sl, intercept=ic))
            prev = (zq, yq, cq)
        for pol in POLS:
            rows.append(dict(spec=spec, policy=pol, quarter="_fits", n=fits[pol], deaths=0, obs=np.nan, mean_pred=float(np.mean(rnds[pol])) if rnds[pol] else np.nan))
    d = pd.DataFrame(rows); d.to_csv(os.path.join(RES, "hical_T6.csv"), index=False)
    # overall summary
    S = []; boot = {}; dq = d[d.quarter != "_fits"]
    for spec in SPECS:
        y = yall
        for pol in POLS:
            pp = allp[spec][pol]; g = dq[(dq.spec == spec) & (dq.policy == pol)]; ft = d[(d.spec == spec) & (d.policy == pol) & (d.quarter == "_fits")].iloc[0]
            O = np.bincount(cte, y, K); E = np.bincount(cte, pp, K); ok = O >= 20; oe = O[ok] / E[ok]
            S.append(dict(spec=spec, policy=pol, brier=float(np.mean((pp - y) ** 2)), logloss=float(-np.mean(y * np.log(np.clip(pp, 1e-9, 1)) + (1 - y) * np.log(np.clip(1 - pp, 1e-9, 1)))), auroc=float(auroc(y, pp)), ece_pooled=float(ece(y, pp)),
                          ece_q=float(np.average(g.ece, weights=g.n)), cal_gap=float(np.average(np.abs(g.mean_pred - g.obs), weights=g.n)), clients_ge20=int(ok.sum()), share_oe_ok=float(np.mean((oe >= BAND[0]) & (oe <= BAND[1]))), mean_abs_log_oe=float(np.mean(np.abs(np.log(np.clip(oe, 1e-3, 1e3))))),
                          fits=int(ft.n), mean_rounds=float(ft.mean_pred), uplink_bytes_per_update=uplink_bytes(max(ft.mean_pred if ft.mean_pred == ft.mean_pred else 0, 0), pol in ("hical", "hical_t"))))
        for a, b in (("hical", "both"), ("hical", "client"), ("hical", "intercept"), ("hical", "frozen"), ("client", "both"), ("raw", "both"), ("hical_t", "hical"), ("hical_t", "both")):
            for kind in ("brier", "logloss"):
                boot[f"{spec}|{a}|{b}|{kind}"] = boot_diff(allp[spec][a], allp[spec][b], y.astype(float), cte, K, kind)
    pd.DataFrame(S).to_csv(os.path.join(RES, "hical_T6_summary.csv"), index=False); json.dump({k: [round(x, 6) for x in v] for k, v in boot.items()}, open(os.path.join(RES, "hical_T6_boot.json"), "w"), indent=1)
    s = pd.DataFrame(S); print(s[s.spec == "hiperc"].round(4).to_string()); print({k: [round(x, 5) for x in v] for k, v in boot.items() if k.startswith("hiperc")})


# ---------------------------------------------------------------- static use, T1 to T5
def static():
    import data
    from metrics import auroc, ece, calib_slope_intercept
    rows = []
    for task in os.environ.get("HICAL_TASKS", "T1 T2 T3 T4 T5").split():
        seeds = sorted(int(os.path.basename(f).split("_s")[1][:-4]) for f in glob.glob(os.path.join(RES, "preds", f"{task}_main", "hiperc_s*.npz")))
        for seed in seeds:
            fed, _, _ = data.build(task, seed); K = len(fed); hk = np.array([c["hosp"] for c in fed]); H = int(hk.max()) + 1
            cva = np.concatenate([np.full(len(c["yva"]), i) for i, c in enumerate(fed)]); cte = np.concatenate([np.full(len(c["yte"]), i) for i, c in enumerate(fed)])
            for spec in SPECS:
                f = os.path.join(RES, "preds", f"{task}_main", f"{spec}_s{seed}.npz")
                if not os.path.exists(f):
                    continue
                z0 = np.load(f); y = z0["y"].astype(float); p = z0["p"].astype(float); yv = z0["yv"].astype(float); zv = lg(z0["pv"].astype(float))
                if len(y) != len(cte) or len(yv) != len(cva):
                    print("skip (size)", task, spec, seed); continue
                if not np.array_equal(y, np.concatenate([c["yte"].numpy() for c in fed]).astype(float)):
                    print("skip (order)", task, spec, seed); continue
                zt = lg(p)
                for pol in ("frozen", "both", "hical"):
                    P = fit_policy(pol, zv, yv, cva, K, hk, H) if pol != "frozen" else dict(a=0.0, b=1.0, dh=np.zeros(H), dk=np.zeros(K)); pq = predict(P, zt, cte, hk); sl, ic = calib_slope_intercept(y.astype(int), pq)
                    rows.append(dict(task=task, spec=spec, seed=seed, policy=pol, n=len(y), events=int(y.sum()), clients=K, brier=float(np.mean((pq - y) ** 2)), logloss=float(-np.mean(y * np.log(np.clip(pq, 1e-9, 1)) + (1 - y) * np.log(np.clip(1 - pq, 1e-9, 1)))),
                                     ece=float(ece(y.astype(int), pq)), slope=sl, intercept=ic, auroc=float(auroc(y.astype(int), pq))))
        print("done", task, len(seeds), "splits")
    d = pd.DataFrame(rows); d.to_csv(os.path.join(RES, "hical_static.csv"), index=False)
    m = d.groupby(["task", "spec", "policy"])[["brier", "logloss", "ece", "slope", "intercept", "auroc"]].mean().reset_index(); m.to_csv(os.path.join(RES, "hical_static_summary.csv"), index=False); print(m[m.spec == "hiperc"].round(4).to_string())


if __name__ == "__main__":
    {"equiv": equiv, "t6": t6, "static": static}[sys.argv[1]]()
