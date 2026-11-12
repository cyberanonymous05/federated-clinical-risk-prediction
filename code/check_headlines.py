"""One-command check of the headline results from the raw data files and the saved predictions, for a reader who wants to verify the results without running
the experiments. It uses only numpy, pandas and scipy (no torch, no code from fl.py, data.py or metrics.py) and prints PASS or FAIL for each claim.
  python -I check_headlines.py      (from the code directory; needs the processed data in ../data and ../results/preds, i.e. after reproduce.sh or with results_preds.zip unpacked)"""
import os, glob, json
import numpy as np, pandas as pd
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, ".."); RES = os.path.join(ROOT, "results"); DATA = os.path.join(ROOT, "data")
fails = []


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + ("  " + detail if detail else ""));  fails.append(name) if not ok else None


def auc(y, p):
    r = rankdata(p); n1 = y.sum(); n0 = len(y) - n1; return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


# 1. data counts and prevalences
cdc = pd.read_csv(os.path.join(DATA, "cdc_diabetes.csv")); check("T2/T3 BRFSS: 253,680 adults", len(cdc) == 253680, str(len(cdc)))
check("T2 diabetes prevalence 13.9%", abs(100 * cdc.Diabetes_binary.mean() - 13.9) < 0.05, f"{100 * cdc.Diabetes_binary.mean():.2f}%"); check("T3 high blood pressure prevalence 42.9%", abs(100 * cdc.HighBP.mean() - 42.9) < 0.05, f"{100 * cdc.HighBP.mean():.2f}%")
t1 = [pd.read_csv(f, header=None, na_values="?") for f in glob.glob(os.path.join(DATA, "processed.*.data"))]; n1 = sum(len(x) for x in t1); p1 = sum((x.iloc[:, -1] > 0).sum() for x in t1)
check("T1 heart disease: 920 patients, 55.3% with disease", n1 == 920 and abs(100 * p1 / n1 - 55.3) < 0.05, f"{n1}, {100 * p1 / n1:.2f}%")
nh = pd.read_csv(os.path.join(DATA, "nhanes_processed.csv")); check("T4 NHANES: 36,666 adults, 18.5% with diabetes", len(nh) == 36666 and abs(100 * nh.y.mean() - 18.5) < 0.05, f"{len(nh)}, {100 * nh.y.mean():.2f}%")
sp = pd.read_csv(os.path.join(DATA, "sepsis_processed.csv"), usecols=["y", "system", "pid"]); check("T5 sepsis: 30,827 patients, 1,580 events, unique ids", len(sp) == 30827 and sp.y.sum() == 1580 and sp.pid.is_unique, f"{len(sp)}, {sp.y.sum()}")
mx = pd.read_csv(os.path.join(DATA, "mx_processed.csv.gz"), usecols=["period", "y"]); g = mx.groupby("period").y.agg(["size", "mean"])
check("T6 sample: 354,120 patients; mortality 13.5% (train) and 3.8% (test period, before the client filter)", len(mx) == 354120 and abs(100 * g.loc["train", "mean"] - 13.5) < 0.05 and abs(100 * g.loc["test", "mean"] - 3.8) < 0.05, f"{len(mx)}, {100 * g.loc['train', 'mean']:.2f}%, {100 * g.loc['test', 'mean']:.2f}%")

# 2. seed-level AUROC recomputed from the saved predictions against the run tables, and federated versus pooled network (within 0.02)
for t in ["T1", "T2", "T3", "T4", "T5", "T6"]:
    runs = pd.read_csv(os.path.join(RES, f"runs_{t}_main.csv")); worst = 0.0; means = {}
    for spec in ("hiperc", "fedavg", "logreg", "centralized"):
        v = []
        for f in sorted(glob.glob(os.path.join(RES, "preds", f"{t}_main", f"{spec}_s*.npz"))):
            z = np.load(f); s = int(os.path.basename(f).split("_s")[1][:-4]); a = auc(z["y"].astype(int), z["p"].astype(float)); v.append(a); worst = max(worst, abs(a - runs[(runs.spec == spec) & (runs.seed == s)].auroc.iloc[0]))
        means[spec] = np.mean(v)
    check(f"{t}: recomputed AUROC equals the run table (max difference {worst:.5f})", worst < 1e-4)
    check(f"{t}: federated HiPerC-FL within 0.02 AUROC of the pooled neural network", abs(means["hiperc"] - means["centralized"]) < 0.02, f"{means['hiperc']:.3f} vs {means['centralized']:.3f}")
    if t in ("T1", "T2", "T3", "T4", "T5", "T6"):
        w = {"fedavg": [], "hiperc": []}                                    # traffic to the central server until 99% of FedAvg's best validation AUROC (the traffic measure used here), per seed
        for f in sorted(glob.glob(os.path.join(RES, "preds", f"{t}_main", "fedavg_s*.npz"))):
            sd = os.path.basename(f).split("_s")[1]; cg = np.load(f)["curve"]; cf = np.load(os.path.join(RES, "preds", f"{t}_main", "hiperc_s" + sd))["curve"]; target = 0.99 * cg[:, 1].max()
            for name, c in (("fedavg", cg), ("hiperc", cf)):
                hit = np.where(c[:, 1] >= target)[0]; w[name].append(c[hit[0], 2] / 1e6 if len(hit) else np.nan)
        ratio = np.nanmean(w["fedavg"]) / np.nanmean(w["hiperc"]); expected = {"T1": 33, "T2": 11, "T3": 9, "T4": 2, "T5": 22, "T6": 29}[t]
        check(f"{t}: traffic to the central server until 99% of FedAvg's best validation AUROC, ratio {ratio:.0f} (expected {expected})", abs(ratio - expected) <= 1.0)

# 3. modelled decision impact: fewer patients flagged than the conventional-predictor model (expected: 13% to 27%)
im = pd.read_csv(os.path.join(RES, "impact.csv")).groupby(["task", "spec"]).flagged.mean(); fewer = {t: 100 * (im[(t, "clinical_lr")] - im[(t, "hiperc")]) / im[(t, "clinical_lr")] for t in ("T2", "T4", "T5", "T6")}
check("modelled impact: 13% to 27% fewer patients flagged", 12.5 <= min(fewer.values()) and max(fewer.values()) <= 27.5, ", ".join(f"{k} {v:.0f}%" for k, v in fewer.items()))
print("\n%d check(s) failed" % len(fails) if fails else "\nall checks passed")
