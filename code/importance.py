"""E11: permutation importance (drop in pooled test AUROC when one attribute is shuffled) for HiPerC-FL personal models
and for pooled logistic regression, seeds 0-2, attribute groups (one-hot levels and missingness flags grouped)."""
import os, json, re
import numpy as np
import data, fl
from metrics import auroc
from sklearn.linear_model import LogisticRegression

RES = os.environ.get("HIPERC_RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
cfgs = {t: json.load(open(os.path.join(RES, f"configs_{t}.json")))["best"] for t in ("T1", "T2", "T3")}
from settings import SET


def groups_of(names):
    g = {}
    for j, n in enumerate(names):
        base = n.replace("miss_", "").split("=")[0]
        g.setdefault(base, []).append(j)
    return g


def perm_imp(predict_fn, Xs, y, groups, reps=10, seed=0):
    rng = np.random.RandomState(seed); base = auroc(y, predict_fn(Xs)); out = {}
    for g, cols in groups.items():
        d = []
        for _ in range(reps):
            Xp = Xs.copy(); idx = rng.permutation(len(Xs)); Xp[:, cols] = Xs[idx][:, cols]
            d.append(base - auroc(y, predict_fn(Xp)))
        out[g] = float(np.mean(d))
    return out


res = {}
for t in ("T1", "T2", "T3"):
    cl = {k: v for k, v in cfgs[t]["hiperc"].items() if not k.startswith("_")}
    imp_h, imp_l = [], []
    for seed in range(3):
        fed, _, info = data.build(t, seed); d = fed[0]["Xtr"].shape[1]
        obj = fl.HiPerC(fed, d, cl, seed)
        for r in range(1, SET[t]["rounds"] + 1):
            obj.round(r)
        X = np.concatenate([c["Xte"].numpy() for c in fed]); y = np.concatenate([c["yte"].numpy() for c in fed])
        cid = np.concatenate([np.full(len(c["yte"]), i) for i, c in enumerate(fed)])
        import torch
        def pred_h(Xs):
            p = np.zeros(len(Xs))
            for i in range(len(fed)):
                m = cid == i; p[m] = fl.predict_prob(obj.v[i], torch.tensor(Xs[m], dtype=torch.float32))
            return p
        Xtr = np.concatenate([c["Xtr"].numpy() for c in fed]); ytr = np.concatenate([c["ytr"].numpy() for c in fed])
        lr = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, ytr)
        grp = groups_of(info["names"])
        imp_h.append(perm_imp(pred_h, X, y, grp, reps=5, seed=seed))
        imp_l.append(perm_imp(lambda Z: lr.predict_proba(Z)[:, 1], X, y, grp, reps=5, seed=seed))
    keys = list(imp_h[0].keys())
    mh = {k: np.mean([i[k] for i in imp_h]) for k in keys}; ml = {k: np.mean([i[k] for i in imp_l]) for k in keys}
    top = sorted(keys, key=lambda k: -mh[k])[:8]
    res[t] = [[k, f"{mh[k]:.3f}", f"{ml[k]:.3f}"] for k in top]
    print(t, [(k, round(mh[k], 3), round(ml[k], 3)) for k in top])
json.dump(res, open(os.path.join(RES, "importance.json"), "w"), indent=1)
