"""Added experiment: leave-one-income-group-out on BRFSS (T2, T3).
Eight real income groups are the environments (7 for training, 1 held out), i.e. more environments than the 3-5 used before.
Tests whether the invariance penalty (lam_c) helps when more environments are available. Configurations are those tuned for the
five-band federation (not re-tuned). Usage: python loho_cdc.py"""
import os, json, time
import numpy as np, pandas as pd
from multiprocessing import Pool
import data, fl
from metrics import full_metrics, youden_threshold
from settings import SET, MAIN_METHODS

RES = os.environ.get("HIPERC_RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
SEEDS = list(range(5))


def job(a):
    task, name, m, cfg, h, seed = a
    fed, ext, _ = data.build(task, seed, regions="income8", holdout=h); d = fed[0]["Xtr"].shape[1]
    r = fl.run_method(m, fed, d, cfg, seed, SET[task]["rounds"], SET[task]["eval_every"], ext_X=ext["X"])
    yv = np.concatenate([c["yva"].numpy() for c in fed]); pv = np.concatenate(r["val"])
    return dict(task=task, spec=name, method=m, holdout=h, seed=seed, **full_metrics(ext["y"].numpy(), r["ext"], youden_threshold(yv, pv)))


def main():
    t0 = time.time(); rows = []
    for task in ("T2", "T3"):
        best = json.load(open(os.path.join(RES, f"configs_{task}.json")))["best"]
        cl = lambda c: {k: v for k, v in c.items() if not k.startswith("_")}
        sp = {"logreg": ("logreg", {}), "fedavg": ("fedavg", cl(best["fedavg"])), "hierfavg": ("hierfavg", cl(best["hierfavg"])),
              "hiperc_lamc0": ("hiperc", cl(best["hiperc_stage1_lamc0"])), "hiperc": ("hiperc", cl(best["hiperc"])),
              "hiperc_lamc10": ("hiperc", {**cl(best["hiperc"]), "lam_c": 10.0})}
        jobs = [(task, n, m, c, h, s) for n, (m, c) in sp.items() for h in range(8) for s in SEEDS]
        with Pool(9) as p:
            rows += p.map(job, jobs, chunksize=1)
        print(task, "done %.0fs" % (time.time() - t0), flush=True)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, "loho_cdc.csv"), index=False)
    print(df.groupby(["task", "spec"]).auroc.mean().unstack(0).round(4))


if __name__ == "__main__":
    main()
