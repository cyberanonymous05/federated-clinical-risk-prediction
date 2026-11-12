"""E3: leave-one-hospital-out external validation on T1 (UCI Heart Disease, 4 real hospitals).
Train on 3 hospitals, test on ALL rows of the held-out hospital (zero-shot: the global model, no local data).
Usage: python loho.py"""
import os, json, time
import numpy as np
import pandas as pd
from multiprocessing import Pool
import data, fl
from metrics import full_metrics, youden_threshold, auroc
from settings import SET

RES = os.environ.get("HIPERC_RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))


def job(args):
    name, method, cfg, h, seed = args
    fed, ext, info = data.build_heart(seed, holdout=h)
    d = fed[0]["Xtr"].shape[1]
    r = fl.run_method(method, fed, d, cfg, seed, SET["T1"]["rounds"], SET["T1"]["eval_every"], ext_X=ext["X"])
    y = ext["y"].numpy(); p = r["ext"]
    yv = np.concatenate([c["yva"].numpy() for c in fed]); pv = np.concatenate(r["val"])
    thr = youden_threshold(yv, pv)          # threshold from internal validation only
    row = dict(spec=name, method=method, holdout=h, seed=seed, **full_metrics(y, p, thr))
    return row


def main():
    best = json.load(open(os.path.join(RES, "configs_T1.json")))["best"]
    clean = lambda c: {k: v for k, v in c.items() if not k.startswith("_")}
    sp = {m: (m, clean(best[m])) for m in ("logreg", "centralized", "fedavg", "fedprox", "scaffold",
                                           "perfedavg", "hierfavg")}
    h = clean(best["hiperc"])
    sp["hiperc_lamc0"] = ("hiperc", clean(best["hiperc_stage1_lamc0"]))
    sp["hiperc"] = ("hiperc", h)
    sp["hiperc_lamc10"] = ("hiperc", {**h, "lam_c": 10.0})
    jobs = [(n, m, c, ho, s) for n, (m, c) in sp.items() for ho in range(4) for s in SET["T1"]["seeds"]]
    t0 = time.time()
    with Pool(9) as p:
        rows = p.map(job, jobs, chunksize=1)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, "loho_T1.csv"), index=False)
    print("LOHO done %.0fs" % (time.time() - t0))
    print(df.pivot_table(index="spec", columns="holdout", values="auroc", aggfunc="mean").round(3).to_string())
    print(df.groupby("spec").auroc.mean().round(3).sort_values().to_string())


if __name__ == "__main__":
    main()
