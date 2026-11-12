"""Federated linear models on T5 (sepsis): does the advantage of pooled logistic regression come from the model class or from not federating?
Same protocol as linear_exp.py (validation-only tuning on seeds 0-1), then all splits, then both cross-system directions.  Usage: python linear_sepsis.py"""
import os, json, time
import numpy as np, pandas as pd
from multiprocessing import Pool
import data, fl
import linear_exp as LX
from metrics import full_metrics, youden_threshold
from settings import SET

RES = os.environ.get("HIPERC_RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
TASK = "T5"


def loho_job(a):
    name, m, cfg, h, seed = a
    fed, ext, _ = data.build_sepsis(seed, holdout=[h]); d = fed[0]["Xtr"].shape[1]
    r = fl.run_method(m, fed, d, cfg, seed, SET[TASK]["rounds"], SET[TASK]["eval_every"], ext_X=ext["X"])
    yv = np.concatenate([c["yva"].numpy() for c in fed]); pv = np.concatenate(r["val"])
    return dict(spec=name, method=m, holdout=h, seed=seed, **full_metrics(ext["y"].numpy(), r["ext"], youden_threshold(yv, pv)))


def main():
    t0 = time.time(); best = LX.tune(TASK)
    names = {"fedavg": "lin_fedavg", "hierfavg": "lin_hierfavg", "hiperc": "lin_hiperc", "hiperc_lamc0": "lin_hiperc_lamc0"}
    meth = {"fedavg": "fedavg", "hierfavg": "hierfavg", "hiperc": "hiperc", "hiperc_lamc0": "hiperc"}
    json.dump({TASK: best}, open(os.path.join(RES, "configs_linear_T5.json"), "w"), indent=1)
    jobs = [(TASK, names[k], meth[k], best[k], s) for k in names for s in SET[TASK]["seeds"]]
    with Pool(9) as p:
        rows = p.map(LX.run_job, jobs, chunksize=1)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, f"runs_{TASK}_linear.csv"), index=False)
    print("standard done %.0fs" % (time.time() - t0)); print(df.groupby("spec")[["auroc", "brier", "ece"]].mean().round(4))
    jobs = [(names[k], meth[k], best[k], h, s) for k in names for h in (0, 1) for s in SET[TASK]["seeds"]]
    with Pool(9) as p:
        rows = p.map(loho_job, jobs, chunksize=1)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, f"loho_{TASK}_linear.csv"), index=False)
    print(df.pivot_table(index="spec", columns="holdout", values="auroc", aggfunc="mean").round(3))


if __name__ == "__main__":
    main()
