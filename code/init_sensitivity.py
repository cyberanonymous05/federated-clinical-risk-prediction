"""Sensitivity to weight initialisation (found while cross-checking against FLamby: PyTorch's default initialisation changed T1 AUROC at a
fixed round by ~0.009). Re-run the main methods with PyTorch-default initialisation (other settings as tuned) on T1 and T4.
Usage: python init_sensitivity.py"""
import os, json, time
import numpy as np, pandas as pd
from multiprocessing import Pool
import data, fl
from metrics import auroc
from settings import SET

RES = os.environ.get("HIPERC_RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))


def job(a):
    task, name, m, cfg, seed, init = a
    fed, _, _ = data.build(task, seed); d = fed[0]["Xtr"].shape[1]
    r = fl.run_method(m, fed, d, {**cfg, "init": init}, seed, SET[task]["rounds"], SET[task]["eval_every"])
    y = np.concatenate([c["yte"].numpy() for c in fed]); p = np.concatenate(r["test"])
    return dict(task=task, method=name, init=init, seed=seed, auroc=auroc(y, p))


def main():
    t0 = time.time(); jobs = []
    for task in ("T1", "T4"):
        best = json.load(open(os.path.join(RES, f"configs_{task}.json")))["best"]; cl = lambda c: {k: v for k, v in c.items() if not k.startswith("_")}
        ft = json.load(open(os.path.join(RES, "configs_ft.json")))[task]
        for init in ("he", "torch"):
            for name in ("fedavg", "hierfavg", "ditto", "hiperc", "centralized"):
                for s in SET[task]["seeds"]:
                    jobs.append((task, name, name, cl(best[name]), s, init))
            for s in SET[task]["seeds"]:
                jobs.append((task, "fedavg_ft", "fedavg_ft", ft, s, init))
    with Pool(9) as p:
        rows = p.map(job, jobs, chunksize=1)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, "init_sensitivity.csv"), index=False)
    print("done %.0fs" % (time.time() - t0)); print(df.groupby(["task", "init", "method"]).auroc.mean().unstack(2).round(4).to_string())
    pv = df.pivot_table(index=["task", "init", "seed"], columns="method", values="auroc")
    print((pv.hiperc - pv.fedavg).groupby(level=[0, 1]).agg(["mean", "std"]).round(4))


if __name__ == "__main__":
    main()
