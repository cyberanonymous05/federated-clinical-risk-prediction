"""Aggregate raw runs into summary tables.  Usage: python analyze.py T1|T2|T3
Writes results/tables/<task>_*.csv ; downstream summaries read these files."""
import sys, os, glob, json
import numpy as np
import pandas as pd
from scipy import stats
from metrics import t_ci, holm, auroc

RES = os.environ.get("HIPERC_RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
TAB = os.path.join(RES, "tables"); os.makedirs(TAB, exist_ok=True)

NAMES = dict(local="Local-only (per hospital)", logreg="Logistic regression (pooled)", centralized="MLP (pooled)",
             clinical_lr="Clinical-style LR (few predictors)", fedavg="FedAvg", fedavg_ft="FedAvg + local fine-tuning", fedprox="FedProx",
             scaffold="SCAFFOLD", ditto="Ditto", pfedme="pFedMe", perfedavg="Per-FedAvg",
             ifca="IFCA (clustered FL)", hierfavg="HierFAVG", hiperc="HiPerC-FL",
             hiperc_lamc0="HiPerC-FL, no invariance term (λ~c~ = 0)", hiperc_flat="HiPerC-FL, flat (τ = 1)",
             hiperc_hospital="HiPerC-FL, hospital model only", hiperc_global="HiPerC-FL, global model only")
ORDER = ["clinical_lr", "logreg", "centralized", "local", "fedavg", "fedavg_ft", "fedprox", "scaffold", "hierfavg", "ifca",
         "perfedavg", "pfedme", "ditto", "hiperc"]
ABL = ["hiperc", "hiperc_lamc0", "hiperc_flat", "hiperc_hospital", "hiperc_global"]
METRICS = ["auroc", "auprc", "acc", "sensitivity", "specificity", "f1", "brier", "ece", "cal_slope", "cal_int"]


def load(task, which="main"):
    df = pd.read_csv(os.path.join(RES, f"runs_{task}_{which}.csv"))
    f = os.path.join(RES, f"runs_{task}_clinical.csv")
    if which == "main" and os.path.exists(f):
        df = pd.concat([df, pd.read_csv(f)], ignore_index=True)
    g = os.path.join(RES, f"runs_{task}_ft.csv")
    if which == "main" and os.path.exists(g):
        df = pd.concat([df, pd.read_csv(g)], ignore_index=True)
    return df


def fmt(m, s, nd=3):
    return f"{m:.{nd}f} ± {s:.{nd}f}"


def table_main(df, task):
    rows = []
    for sp in ORDER:
        d = df[df.spec == sp]
        if d.empty:
            continue
        r = dict(Method=NAMES[sp], spec=sp, n_seeds=len(d))
        for m in METRICS:
            r[m] = fmt(d[m].mean(), d[m].std())
        lo, hi = t_ci(d.auroc.values); r["auroc_ci95"] = f"[{lo:.3f}, {hi:.3f}]"
        r["wan_MB"] = round(d.wan_MB.mean(), 1); r["lan_MB"] = round(d.lan_MB.mean(), 1)
        r["auroc_mean"] = d.auroc.mean()
        rows.append(r)
    t = pd.DataFrame(rows); t.to_csv(os.path.join(TAB, f"{task}_E1_main.csv"), index=False)
    return t


def paired(df, a, others, metric="auroc"):
    """Paired (by seed) difference a - b, Wilcoxon signed-rank, Holm-adjusted."""
    rows = []; A = df[df.spec == a].set_index("seed")[metric]
    for b in others:
        B = df[df.spec == b].set_index("seed")[metric]
        idx = A.index.intersection(B.index); x, y = A[idx].values, B[idx].values
        diff = x - y
        try:
            p = stats.wilcoxon(x, y).pvalue if np.any(diff != 0) else 1.0
        except ValueError:
            p = 1.0
        lo, hi = t_ci(diff)
        rows.append(dict(comparator=NAMES[b], spec=b, mean_diff=diff.mean(), ci_lo=lo, ci_hi=hi, p_wilcoxon=p,
                         wins=int((diff > 0).sum()), n=len(diff)))
    t = pd.DataFrame(rows); t["p_holm"] = holm(t.p_wilcoxon.values)
    return t


def table_hosp(df, task):
    cols = [c for c in df.columns if c.startswith("auroc_h")]
    rows = []
    for sp in ORDER:
        d = df[df.spec == sp]
        if d.empty:
            continue
        r = dict(Method=NAMES[sp], spec=sp)
        for c in cols:
            r[c] = fmt(d[c].mean(), d[c].std())
        rows.append(r)
    t = pd.DataFrame(rows); t.to_csv(os.path.join(TAB, f"{task}_E2_per_hospital.csv"), index=False)
    return t


def table_ablation(df, task):
    full = df[df.spec == "hiperc"].set_index("seed")
    rows = []
    for sp in ABL:
        d = df[df.spec == sp].set_index("seed")
        if d.empty:
            continue
        idx = full.index.intersection(d.index)
        diff = d.loc[idx, "auroc"].values - full.loc[idx, "auroc"].values
        p = stats.wilcoxon(d.loc[idx, "auroc"].values, full.loc[idx, "auroc"].values).pvalue if np.any(diff != 0) and sp != "hiperc" else np.nan
        rows.append(dict(Variant=NAMES[sp], spec=sp, auroc=fmt(d.auroc.mean(), d.auroc.std()),
                         brier=fmt(d.brier.mean(), d.brier.std()), ece=fmt(d.ece.mean(), d.ece.std()),
                         d_auroc_vs_full=diff.mean(), p_wilcoxon=p, wan_MB=round(d.wan_MB.mean(), 1)))
    t = pd.DataFrame(rows); t.to_csv(os.path.join(TAB, f"{task}_E4_ablation.csv"), index=False)
    return t


def comm_to_target(task, df):
    """WAN/LAN MB needed to reach 99% of FedAvg's best validation AUROC (per seed; NaN if never reached)."""
    out = {}
    d = os.path.join(RES, "preds", f"{task}_main")
    for sp in ["fedavg", "fedprox", "scaffold", "hierfavg", "ditto", "hiperc", "hiperc_flat"]:
        vals = []
        for s in sorted(df.seed.unique()):
            f = os.path.join(d, f"{sp}_s{s}.npz"); g = os.path.join(d, f"fedavg_s{s}.npz")
            if not (os.path.exists(f) and os.path.exists(g)):
                continue
            cf = np.load(f)["curve"]; cg = np.load(g)["curve"]
            target = 0.99 * cg[:, 1].max()
            hit = np.where(cf[:, 1] >= target)[0]
            vals.append((cf[hit[0], 0], cf[hit[0], 2] / 1e6, cf[hit[0], 3] / 1e6) if len(hit) else (np.nan,) * 3)
        v = np.array(vals, dtype=float)
        out[sp] = dict(Method=NAMES[sp], reached=int((~np.isnan(v[:, 0])).sum()), n=len(v),
                       rounds=np.nanmean(v[:, 0]), wan_MB=np.nanmean(v[:, 1]), lan_MB=np.nanmean(v[:, 2]))
    t = pd.DataFrame(out.values()); t.to_csv(os.path.join(TAB, f"{task}_E8_comm_to_target.csv"), index=False)
    return t


def groups(task, meta):
    """Subgroup labels from saved test metadata."""
    g = {}
    sex = meta["sex"]
    g["sex"] = np.where(sex == 1, "male", "female")
    age = meta["age"]
    if task in ("T1", "T4"):
        g["age"] = np.where(age < 45, "<45", np.where(age < 60, "45-59", ">=60"))
    else:
        g["age"] = np.where(age <= 5, "18-44", np.where(age <= 9, "45-64", "65+"))
    if task not in ("T1", "T4"):
        g["income"] = np.array([f"band{int(h) + 1}" for h in meta["hosp"]])
    elif task == "T1":
        g["hospital"] = np.array([f"h{int(h)}" for h in meta["hosp"]])
    else:
        g["cycle"] = np.array([f"c{int(h)}" for h in meta["hosp"]])
    return g


def fairness(task, df, specs=("logreg", "fedavg", "ditto", "hierfavg", "hiperc")):
    d = os.path.join(RES, "preds", f"{task}_main"); rows = []; gaprows = []
    for sp in specs:
        for s in sorted(df.seed.unique()):
            f = os.path.join(d, f"{sp}_s{s}.npz")
            if not os.path.exists(f):
                continue
            z = np.load(f); y, p, thr = z["y"], z["p"], float(z["thr"])
            meta = {k: z[k] for k in z.files if k in ("sex", "age", "hosp", "income")}
            for dim, lab in groups(task, meta).items():
                per = {}
                for gname in np.unique(lab):
                    m = lab == gname; yy, pp = y[m], p[m]
                    pred = pp >= thr
                    tpr = pred[yy == 1].mean() if (yy == 1).any() else np.nan
                    fpr = pred[yy == 0].mean() if (yy == 0).any() else np.nan
                    per[gname] = (auroc(yy, pp), tpr, fpr, int(m.sum()), float(yy.mean()))
                    rows.append(dict(spec=sp, seed=s, dim=dim, group=gname, auroc=per[gname][0], tpr=tpr, fpr=fpr,
                                     n=int(m.sum()), prev=float(yy.mean())))
                a = np.array([v[0] for v in per.values()], float); t_ = np.array([v[1] for v in per.values()], float)
                f_ = np.array([v[2] for v in per.values()], float)
                gaprows.append(dict(spec=sp, seed=s, dim=dim, auroc_gap=np.nanmax(a) - np.nanmin(a),
                                    eo_diff=max(np.nanmax(t_) - np.nanmin(t_), np.nanmax(f_) - np.nanmin(f_))))
    r = pd.DataFrame(rows); r.to_csv(os.path.join(TAB, f"{task}_E6_subgroups_raw.csv"), index=False)
    ag = r.groupby(["spec", "dim", "group"]).agg(auroc=("auroc", "mean"), auroc_sd=("auroc", "std"),
                                                  tpr=("tpr", "mean"), fpr=("fpr", "mean"), n=("n", "mean"),
                                                  prev=("prev", "mean")).reset_index()
    ag.to_csv(os.path.join(TAB, f"{task}_E6_subgroups.csv"), index=False)
    gp = pd.DataFrame(gaprows).groupby(["spec", "dim"]).agg(auroc_gap=("auroc_gap", "mean"), auroc_gap_sd=("auroc_gap", "std"),
                                                             eo_diff=("eo_diff", "mean"), eo_diff_sd=("eo_diff", "std")).reset_index()
    gp.to_csv(os.path.join(TAB, f"{task}_E6_gaps.csv"), index=False)
    return ag, gp


def main(task):
    df = load(task)
    t1 = table_main(df, task); print(t1[["Method", "auroc", "auroc_ci95", "auprc", "brier", "ece", "wan_MB"]].to_string(index=False))
    others = [s for s in ORDER if s != "hiperc" and s in set(df.spec)]
    pt = paired(df, "hiperc", others); pt.to_csv(os.path.join(TAB, f"{task}_E10_paired_vs_hiperc.csv"), index=False)
    print(pt.round(4).to_string(index=False))
    print(table_hosp(df, task).to_string(index=False))
    print(table_ablation(df, task).round(4).to_string(index=False))
    print(comm_to_target(task, df).round(1).to_string(index=False))
    ag, gp = fairness(task, df); print(gp.round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])


def stratification(task, specs=("logreg", "fedavg", "hiperc")):
    """Observed event rate per predicted-risk quintile and share of all cases in the top quintile (mean over seeds)."""
    d = os.path.join(RES, "preds", f"{task}_main"); out = []
    for sp in specs:
        er, share = [], []
        for f in sorted(glob.glob(os.path.join(d, f"{sp}_s*.npz"))):
            z = np.load(f); y, p = z["y"], z["p"]
            q = np.quantile(p, [0.2, 0.4, 0.6, 0.8]); b = np.digitize(p, q)
            er.append([y[b == k].mean() for k in range(5)]); share.append(y[b == 4].sum() / y.sum())
        er = np.mean(er, axis=0)
        out.append(dict(spec=sp, **{f"q{k + 1}": er[k] for k in range(5)}, top_quintile_share=float(np.mean(share))))
    t = pd.DataFrame(out); t.to_csv(os.path.join(TAB, f"{task}_E7_stratification.csv"), index=False)
    return t
