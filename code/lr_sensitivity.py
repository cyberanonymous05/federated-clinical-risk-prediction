"""Reviewer-anticipated sensitivity: the tuned learning rate sat at the upper edge of the grid for several methods. Re-run FedAvg,
hierarchical FedAvg and HiPerC-FL (other hyper-parameters as tuned) at larger learning rates (5 seeds) and compare.
Usage: python lr_sensitivity.py"""
import os, json, time
import numpy as np, pandas as pd
from multiprocessing import Pool
import data, fl
from metrics import auroc
from settings import SET

RES = os.environ.get("HIPERC_RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
HIGH = {"T1": [0.6, 1.0], "T2": [0.2, 0.5, 1.0], "T3": [0.2, 0.5, 1.0], "T4": [0.6, 1.0]}


def job(a):
    task, name, m, cfg, seed = a
    fed, _, _ = data.build(task, seed); d = fed[0]["Xtr"].shape[1]
    r = fl.run_method(m, fed, d, cfg, seed, SET[task]["rounds"], SET[task]["eval_every"])
    y = np.concatenate([c["yte"].numpy() for c in fed]); p = np.concatenate(r["test"])
    return dict(task=task, method=name, lr=cfg["lr"], seed=seed, auroc=auroc(y, p), val_auc=r["val_auc"], finite=bool(np.isfinite(p).all()))


def main():
    t0 = time.time(); jobs = []
    for task in ("T1", "T2", "T3", "T4"):
        best = json.load(open(os.path.join(RES, f"configs_{task}.json")))["best"]; cl = lambda c: {k: v for k, v in c.items() if not k.startswith("_")}
        for lr in HIGH[task]:
            for name in ("fedavg", "hierfavg", "hiperc"):
                for s in range(5):
                    jobs.append((task, name, name, {**cl(best[name]), "lr": lr}, s))
        for name in ("fedavg", "hierfavg", "hiperc"):                       # the tuned value, same 5 seeds, for reference
            for s in range(5):
                jobs.append((task, name, name, cl(best[name]), s))
    with Pool(9) as p:
        rows = p.map(job, jobs, chunksize=1)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, "lr_sensitivity.csv"), index=False)
    print("done %.0fs" % (time.time() - t0))
    g = df.groupby(["task", "method", "lr"]).agg(auroc=("auroc", "mean"), val=("val_auc", "mean"), finite=("finite", "all")).round(4)
    print(g.unstack(1).to_string())


if __name__ == "__main__":
    main()
