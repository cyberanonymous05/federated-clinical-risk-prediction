"""Patient-level inference (addresses the optimism of seed-based intervals).

Every run saves test predictions in a deterministic order, so each prediction can be mapped back to its patient row.
For each method, a patient's out-of-sample score is the average of the scores from all splits in which that patient
was a test case (T1: about 4 of 20 splits; T2/T3: 0-3 of 10 splits). Methods are then compared on the *same patients*:
  * AUROC with a DeLong 95% interval (T1 additionally with a 2000-replicate patient bootstrap),
  * paired DeLong test and interval for the difference HiPerC-FL minus comparator (Holm-adjusted over comparators),
  * non-inferiority: lower 95% limit of the difference versus -0.02.
These intervals describe uncertainty from the finite number of patients. They do not capture dependence among patients
within a hospital or uncertainty about other populations.
Usage: python stats_patient.py T1|T2|T3"""
import sys, os, glob
import numpy as np, pandas as pd
from scipy import stats
import data
from metrics import holm

RES = os.environ.get("HIPERC_RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
TAB = os.path.join(RES, "tables")
SPECS = ["clinical_lr", "logreg", "centralized", "local", "fedavg", "fedavg_ft", "fedprox", "scaffold", "hierfavg", "ifca",
         "perfedavg", "pfedme", "ditto", "hiperc"]
EXTRA = ["lin_fedavg", "lin_hierfavg", "lin_hiperc", "lin_hiperc_lamc0"]     # added experiment (linear predictor), if available


def midrank(x):
    j = np.argsort(x); z = x[j]; n = len(x); t = np.zeros(n); i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1; i = k
    out = np.empty(n); out[j] = t
    return out


def fast_delong(P, y):
    """P: (k, n) scores, y: (n,) 0/1. Returns AUCs (k,) and covariance (k, k)  [Sun & Xu 2014]."""
    pos = y == 1; m = pos.sum(); n = (~pos).sum(); k = P.shape[0]
    X = P[:, pos]; Y = P[:, ~pos]
    tx = np.array([midrank(r) for r in X]); ty = np.array([midrank(r) for r in Y])
    tzx = np.array([midrank(np.concatenate([X[i], Y[i]]))[:m] for i in range(k)])
    aucs = tzx.sum(1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tzx - tx) / n
    tzy = np.array([midrank(np.concatenate([X[i], Y[i]]))[m:] for i in range(k)])
    v10 = 1.0 - (tzy - ty) / m
    sx = np.cov(v01) if k > 1 else np.atleast_2d(np.var(v01, ddof=1))
    sy = np.cov(v10) if k > 1 else np.atleast_2d(np.var(v10, ddof=1))
    return aucs, sx / m + sy / n


def patient_scores(task, spec, seeds, mode="first"):
    """Return (patient index array, score, label) for one method.
    mode='first': score from the first split (lowest seed) in which the patient was a test case = ONE single-model prediction
                  per patient (primary analysis; no ensembling advantage for high-variance models).
    mode='mean' : average over all splits in which the patient was a test case (sensitivity analysis; ensembling favours
                  high-variance models such as MLPs over logistic regression)."""
    S = {}; C = {}; Y = {}
    for s in seeds:
        f = os.path.join(RES, "preds", f"{task}_main", f"{spec}_s{s}.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f); fed, _, _ = data.build(task, s)
        idx = np.concatenate([c["idx_te"] for c in fed]); yy = np.concatenate([c["yte"].numpy() for c in fed]).astype(int)
        assert len(idx) == len(z["p"]) and np.array_equal(yy, z["y"].astype(int)), f"order mismatch {task} {spec} {s}"
        for i, p, y in zip(idx, z["p"], yy):
            if mode == "first" and i in S:
                continue
            S[i] = S.get(i, 0.0) + float(p); C[i] = C.get(i, 0) + 1; Y[i] = y
    ids = np.array(sorted(S)); return ids, np.array([S[i] / C[i] for i in ids]), np.array([Y[i] for i in ids])


def main(task, mode="first", ref="hiperc"):
    seeds = list(range(20 if task == "T1" else 10)); cache = {}
    for sp in SPECS + [e for e in EXTRA if os.path.exists(os.path.join(RES, "preds", f"{task}_main", f"{e}_s0.npz"))]:
        ids, p, y = patient_scores(task, sp, seeds, mode)
        if len(ids):
            cache[sp] = (ids, p, y)
    common = sorted(set.intersection(*[set(v[0]) for v in cache.values()]))
    keep = {sp: np.isin(v[0], common) for sp, v in cache.items()}
    ref_ids = cache["hiperc"][0][keep["hiperc"]]
    P = np.array([cache[sp][1][keep[sp]] for sp in cache]); y = cache["hiperc"][2][keep["hiperc"]]
    for sp in cache:                                     # same patients in same order
        assert np.array_equal(cache[sp][0][keep[sp]], ref_ids)
    names = list(cache); n_pat = len(y)
    aucs, cov = fast_delong(P, y)
    rows = []; hi = names.index(ref)
    rng = np.random.RandomState(0)
    boot = None
    if task == "T1":
        from sklearn.metrics import roc_auc_score
        boot = np.zeros((2000, len(names)))
        for b in range(2000):
            ii = rng.randint(0, n_pat, n_pat)
            if len(np.unique(y[ii])) < 2:
                boot[b] = np.nan; continue
            boot[b] = [roc_auc_score(y[ii], P[k][ii]) for k in range(len(names))]
    pv = []
    for k, sp in enumerate(names):
        se = np.sqrt(cov[k, k]); r = dict(spec=sp, n_patients=n_pat, auroc=aucs[k], ci_lo=aucs[k] - 1.96 * se, ci_hi=aucs[k] + 1.96 * se)
        if boot is not None:
            r["boot_lo"], r["boot_hi"] = np.nanpercentile(boot[:, k], [2.5, 97.5])
        if sp != ref:
            d = aucs[hi] - aucs[k]; var = cov[hi, hi] + cov[k, k] - 2 * cov[hi, k]
            sd = np.sqrt(max(var, 1e-18)); z = d / sd
            r.update(diff=d, diff_lo=d - 1.96 * sd, diff_hi=d + 1.96 * sd, p=2 * (1 - stats.norm.cdf(abs(z))))
            if boot is not None:
                bd = boot[:, hi] - boot[:, k]; r["diff_boot_lo"], r["diff_boot_hi"] = np.nanpercentile(bd, [2.5, 97.5])
            pv.append(r["p"])
        rows.append(r)
    df = pd.DataFrame(rows)
    m = df.spec != ref; df.loc[m, "p_holm"] = holm(df.loc[m, "p"].values)
    df.to_csv(os.path.join(TAB, f"{task}_E10b_patient_level" + ("" if mode == "first" else "_ensemble") + ("" if ref == "hiperc" else f"_ref_{ref}") + ".csv"), index=False)
    pd.set_option("display.width", 220)
    print(task, mode, "patients", n_pat, "prevalence %.3f" % y.mean())
    print(df[["spec", "auroc", "ci_lo", "ci_hi", "diff", "diff_lo", "diff_hi", "p_holm"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "first", sys.argv[3] if len(sys.argv) > 3 else "hiperc")
