"""Pre-emptive validity checks: (A) implementation sanity tests of the federated algorithms, (B) split / leakage audit,
(C) preprocessing sensitivity on T1. Usage: python validity_checks.py   -> results/validity_checks.json"""
import os, json, itertools
import numpy as np, pandas as pd, torch
import data, fl
from metrics import auroc

RES = os.environ.get("HIPERC_RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
out = {"A": {}, "B": {}, "C": {}}


def cfg(**k):
    return {**dict(epochs=2, lr=0.1, bs=32, frac=1.0), **k}


def final_vec(obj, R=5):
    for t in range(1, R + 1):
        obj.round(t)
    return obj.g.clone()


# ---------------- A. implementation sanity tests
fed, _, _ = data.build("T1", 0); d = fed[0]["Xtr"].shape[1]
one = [dict(fed[0], hosp=0)]                                   # a single client
# A1 FedAvg with one client == plain SGD on that client
fa = fl.FedAvg(one, d, cfg(), 3); g = final_vec(fa)
gen = torch.Generator().manual_seed(3); w = fl.init_vec(d, 3)
for _ in range(5):
    w, _, _ = fl.sgd(w, one[0]["Xtr"], one[0]["ytr"], 2, 0.1, 32, gen)
out["A"]["fedavg_one_client_equals_sgd"] = float((g - w).abs().max())
# A2 FedProx with mu = 0 == FedAvg
g1 = final_vec(fl.FedAvg(fed, d, cfg(), 4)); g2 = final_vec(fl.FedProx(fed, d, cfg(mu=0.0), 4))
out["A"]["fedprox_mu0_equals_fedavg"] = float((g1 - g2).abs().max())
# A3 SCAFFOLD with one client == SGD (control variates cancel)
g3 = final_vec(fl.Scaffold(one, d, cfg(), 3))
out["A"]["scaffold_one_client_equals_sgd"] = float((g3 - w).abs().max())
# A4 HierFAVG with tau = 1 and full participation == FedAvg (clients are ordered by hospital)
g4 = final_vec(fl.HierFAVG(fed, d, cfg(tau=1), 4))
out["A"]["hierfavg_tau1_equals_fedavg"] = float((g1 - g4).abs().max())
# A5 HiPerC (global mode, lam_c = 0, tau = 1) == HierFAVG (tau = 1)
g5 = final_vec(fl.HiPerC(fed, d, cfg(tau=1, lam_c=0.0, mode="global"), 4))
out["A"]["hiperc_global_lamc0_tau1_equals_hierfavg"] = float((g4 - g5).abs().max())
# A6 V-REx weights: reduce to size weights at lam_c = 0, sum to one, up-weight the high-loss hospital
o = fl.HiPerC(fed, d, cfg(tau=5, lam_c=0.0), 4)
res = {0: (None, 0.60, 100.0), 1: (None, 0.40, 100.0)}
w0 = o.global_weights(res); o.lam_c = 1.0; w1 = o.global_weights(res); sw = sum(w1.values()); nrm = {h: v / sw for h, v in w1.items()}
out["A"]["vrex_lamc0_weights_are_size_weights"] = float(abs(w0[0] / sum(w0.values()) - 0.5))
out["A"]["vrex_weights_sum_to_one"] = float(abs(sum(o.last_w.values()) - 1.0))
out["A"]["vrex_upweights_high_loss_hospital"] = bool(nrm[0] > 0.5)
# A7 analytic gradient == finite differences
w = fl.init_vec(d, 0); X, y = fed[0]["Xtr"][:20], fed[0]["ytr"][:20]; _, gr = fl.loss_grad(w, X, y); e = 1e-3; idx = [0, 7, 100, 2100, 4200]
fd = []
for i in idx:
    wp, wm = w.clone(), w.clone(); wp[i] += e; wm[i] -= e
    fd.append(((fl.bce(wp, X, y) - fl.bce(wm, X, y)) / (2 * e)).item())
out["A"]["gradient_vs_finite_difference_max_abs"] = float(np.max(np.abs(np.array(fd) - gr[idx].numpy())))
# A8 per-sample DP gradient with a huge clip norm and no noise == mean gradient
gdp = fl.dp_grad(w, X, y, 1e9, 0.0, torch.Generator().manual_seed(0))
out["A"]["dp_grad_no_clip_no_noise_equals_mean_gradient"] = float((gdp - gr).abs().max())
# A9 Ditto: personal model gets closer to the global model as lambda grows
dist = []
for lam in (0.01, 0.1, 1.0, 5.0):
    o = fl.Ditto(fed, d, cfg(lam=lam, lr=0.05), 4)
    for t in range(1, 6):
        o.round(t)
    dist.append(float(np.mean([(o.v[i] - o.g).norm().item() for i in range(len(fed))])))
out["A"]["ditto_distance_to_global_by_lambda"] = dist
out["A"]["ditto_distance_decreases_with_lambda"] = bool(all(dist[i] >= dist[i + 1] for i in range(len(dist) - 1)))
# A10 communication ledger: flat FedAvg bytes = 2 * participating clients * P * 4 per round
o = fl.FedAvg(fed, d, cfg(), 4); final_vec(o, 5)
out["A"]["ledger_fedavg_bytes_match_formula"] = float(abs(o.wan - 5 * len(fed) * 2 * fl.n_params(d) * 4.0))

# ---------------- B. split / leakage audit
for task in ("T1", "T2", "T3", "T4"):
    rep = {}
    for seed in (0, 1, 2):
        fedx, _, _ = data.build(task, seed)
        tr = np.concatenate([c["idx_tr"] for c in fedx]); va = np.concatenate([c["idx_va"] for c in fedx]); te = np.concatenate([c["idx_te"] for c in fedx])
        rep[seed] = dict(train_val_overlap=int(len(set(tr) & set(va))), train_test_overlap=int(len(set(tr) & set(te))), val_test_overlap=int(len(set(va) & set(te))),
                         duplicate_rows_within_train=int(len(tr) - len(set(tr))), n_train=int(len(tr)), n_val=int(len(va)), n_test=int(len(te)))
        # standardised continuous column: mean ~ 0 on the TRAINING rows (scaling fitted on training rows only)
        Xtr = torch.cat([c["Xtr"] for c in fedx]); rep[seed]["train_col0_mean_abs"] = float(Xtr[:, 0].mean().abs())
    out["B"][task] = rep

# ---------------- C. preprocessing sensitivity on T1 (same tuned configs, 10 seeds)
best = json.load(open(os.path.join(RES, "configs_T1.json")))["best"]
cl = lambda c: {k: v for k, v in c.items() if not k.startswith("_")}
variants = {"main": (), "no_missingness_indicators": "ind", "without_ca_thal_slope": ("ca", "thal", "slope"), "without_chol": ("chol",)}
rows = []
for name, drop in variants.items():
    for seed in range(10):
        if drop == "ind":
            fedv, _, info = data.build_heart(seed); keep = [j for j, n in enumerate(info["names"]) if not n.startswith("miss_")]
            for c in fedv:
                for s_ in ("Xtr", "Xva", "Xte"):
                    c[s_] = c[s_][:, keep]
        else:
            fedv, _, info = data.build_heart(seed, drop=drop)
        dd = fedv[0]["Xtr"].shape[1]; y = np.concatenate([c["yte"].numpy() for c in fedv])
        for m in ("logreg", "fedavg", "hierfavg", "hiperc"):
            r = fl.run_method(m, fedv, dd, cl(best[m]) if m != "logreg" else {}, seed, 200, 5)
            rows.append(dict(variant=name, method=m, seed=seed, auroc=auroc(y, np.concatenate(r["test"]))))
df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, "validity_T1_preprocessing.csv"), index=False)
tab = df.groupby(["variant", "method"]).auroc.agg(["mean", "std"]).round(3)
out["C"] = {f"{v}|{m}": [float(x.loc[(v, m), "mean"]), float(x.loc[(v, m), "std"])] for x in [tab] for (v, m) in tab.index}
pv = df.pivot_table(index=["variant", "seed"], columns="method", values="auroc")
out["C_hiperc_minus_fedavg"] = {v: [float((pv.loc[v].hiperc - pv.loc[v].fedavg).mean()), float((pv.loc[v].hiperc - pv.loc[v].fedavg).std())] for v in variants}
out["C_hiperc_minus_logreg"] = {v: [float((pv.loc[v].hiperc - pv.loc[v].logreg).mean()), float((pv.loc[v].hiperc - pv.loc[v].logreg).std())] for v in variants}
json.dump(out, open(os.path.join(RES, "validity_checks.json"), "w"), indent=1)
print(json.dumps(out["A"], indent=1)); print(json.dumps({t: {s: {k: v for k, v in r.items() if "overlap" in k or "duplicate" in k} for s, r in rr.items()} for t, rr in out["B"].items()})[:600])
print(tab.unstack().to_string()); print(out["C_hiperc_minus_fedavg"]); print(out["C_hiperc_minus_logreg"])
