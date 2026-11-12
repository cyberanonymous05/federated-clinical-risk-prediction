"""Patient-level (DeLong) version of the component ablation, so that the claim 'personal models help on T1' does not rest on the
split-level Wilcoxon test (which describes stability only). Same single out-of-sample score per patient as stats_patient.py.
Difference = variant minus full HiPerC-FL; Holm-adjusted over the four variants within a task. Usage: python ablation_patient.py"""
import os
import numpy as np, pandas as pd
from scipy import stats
from stats_patient import patient_scores, fast_delong, RES
from metrics import holm

VAR = {"hiperc_global": "global model only (no personal models)", "hiperc_hospital": "hospital model only", "hiperc_flat": "flat aggregation (τ = 1)",
       "hiperc_lamc0": "no invariance penalty (λc = 0)"}
rows = []
import sys
for task in (sys.argv[1:] or ["T1", "T2", "T3", "T4"]):
    seeds = list(range(20 if task == "T1" else 10)); ids0, p0, y0 = patient_scores(task, "hiperc", seeds); out = []
    for v in VAR:
        ids, p, y = patient_scores(task, v, seeds); assert np.array_equal(ids, ids0) and np.array_equal(y, y0)
        a, C = fast_delong(np.vstack([p, p0]), y0); d = a[0] - a[1]; var = C[0, 0] + C[1, 1] - 2 * C[0, 1]; se = np.sqrt(var)
        pv = 2 * (1 - stats.norm.cdf(abs(d) / se)) if se > 0 else 1.0
        out.append(dict(task=task, variant=v, n=len(y0), auroc=a[0], full=a[1], diff=d, lo=d - 1.96 * se, hi=d + 1.96 * se, p=pv))
    ph = holm([o["p"] for o in out])
    for o, q in zip(out, ph):
        o["p_holm"] = q; rows.append(o)
df = pd.DataFrame(rows); f_ = os.path.join(RES, "tables", "ablation_patient_level.csv")
if sys.argv[1:] and os.path.exists(f_):
    old = pd.read_csv(f_); df = pd.concat([old[~old.task.isin(df.task.unique())], df], ignore_index=True)
df.to_csv(f_, index=False); print(df.round(4).to_string())
