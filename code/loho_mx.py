"""External validation across institution types (T6): train on four health-system sectors, test on the held-out sector in the later test period (zero-shot).
Both directions, 10 splits, configurations tuned on internal validation data of the standard protocol.  Usage: python loho_sepsis.py"""
import os, sys, json, time
import numpy as np, pandas as pd
from multiprocessing import Pool
import data, fl
from metrics import full_metrics, youden_threshold
from settings import SET

TASK = "T6"
RES = os.environ.get("HIPERC_RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))


def job(a):
    name, method, cfg, h, seed = a
    fed, ext, _ = data.build_mx(seed, holdout=[h]); d = fed[0]["Xtr"].shape[1]
    r = fl.run_method(method, fed, d, cfg, seed, SET[TASK]["rounds"], SET[TASK]["eval_every"], ext_X=ext["X"])
    yv = np.concatenate([c["yva"].numpy() for c in fed]); pv = np.concatenate(r["val"])
    return dict(spec=name, method=method, holdout=h, seed=seed, **full_metrics(ext["y"].numpy(), r["ext"], youden_threshold(yv, pv)))


def main():
    best = json.load(open(os.path.join(RES, f"configs_{TASK}.json")))["best"]; clean = lambda c: {k: v for k, v in c.items() if not k.startswith("_")}
    sp = {m: (m, clean(best[m])) for m in ("logreg", "centralized", "fedavg", "hierfavg", "ditto", "hiperc")}
    sp["hiperc_lamc0"] = ("hiperc", clean(best["hiperc_stage1_lamc0"]))
    jobs = [(n, m, c, h, s) for n, (m, c) in sp.items() for h in range(5) for s in SET[TASK]["seeds"]]
    t0 = time.time()
    with Pool(9) as p:
        rows = p.map(job, jobs, chunksize=1)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, f"loho_{TASK}.csv"), index=False)
    print("done %.0fs" % (time.time() - t0)); print(df.pivot_table(index="spec", columns="holdout", values="auroc", aggfunc="mean").round(3).to_string())


if __name__ == "__main__":
    main()
