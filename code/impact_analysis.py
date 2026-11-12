"""Modelled decision impact at a fixed operating point (NOT measured patient outcomes). For each task and method the risk threshold is chosen on VALIDATION predictions so that
80% of the validation cases are flagged (sensitivity 0.80), then applied unchanged to the test predictions of the same split. Reported (mean over splits): share of patients flagged
(workload), sensitivity and positive predictive value reached on the test data, number needed to evaluate per true case found (1 / PPV), cases missed per 1000 patients,
and net benefit at a task-specific probability threshold as net true positives per 1000 patients (net benefit x 1000).  Usage: python impact_analysis.py T2 T4 T5 T6"""
import os, sys
import numpy as np, pandas as pd
from metrics import net_benefit, treat_all_nb
from settings import SET

RES = os.environ.get("HIPERC_RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
SPECS = ["clinical_lr", "logreg", "lin_hiperc", "centralized", "hiperc"]
NBTH = {"T2": 0.10, "T4": 0.10, "T5": 0.05, "T6": 0.05}            # decision threshold for the net-benefit column: act when predicted risk is at least this
SENS = 0.80


def one(path):
    z = np.load(path); y = z["y"].astype(int); p = z["p"].astype(float); yv = z["yv"].astype(int); pv = z["pv"].astype(float)
    t = np.quantile(pv[yv == 1], 1 - SENS)                       # threshold giving sensitivity 0.80 on validation data
    flag = p >= t; tp = int((flag & (y == 1)).sum()); fp = int((flag & (y == 0)).sum()); pos = int(y.sum())
    sens = tp / pos; ppv = tp / max(tp + fp, 1)
    nb = float(net_benefit(y, p, [NBTH_]) [0])
    return dict(flagged=flag.mean(), sens=sens, ppv=ppv, nne=1 / max(ppv, 1e-9), missed_per_1000=1000 * (pos - tp) / len(y), nb_per_1000=1000 * nb,
                treat_all_per_1000=1000 * float(treat_all_nb(y, [NBTH_])[0]), prevalence=y.mean())


rows = []
for task in sys.argv[1:] or ["T2", "T4", "T5", "T6"]:
    NBTH_ = NBTH[task]
    for sp in SPECS:
        for seed in SET[task]["seeds"]:
            f = os.path.join(RES, "preds", f"{task}_main", f"{sp}_s{seed}.npz")
            if os.path.exists(f):
                rows.append(dict(task=task, spec=sp, seed=seed, **one(f)))
df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, "impact.csv"), index=False)
print(df.groupby(["task", "spec"])[["flagged", "sens", "ppv", "nne", "missed_per_1000", "nb_per_1000", "treat_all_per_1000", "prevalence"]].mean().round(3).to_string())
