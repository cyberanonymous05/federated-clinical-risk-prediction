"""Added baseline: FedAvg + local fine-tuning. The FedAvg configuration is the one already tuned; the number of
fine-tuning epochs is chosen on validation data (seeds 0-1). Then 10/20 seeds as for all methods. Usage: python ft_exp.py"""
import os, json, time
import numpy as np, pandas as pd
from multiprocessing import Pool
import data, fl, linear_exp
from settings import SET

RES = os.environ.get("HIPERC_RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))


def tune_job(a):
    task, cfg, seed = a
    fed, _, _ = data.build(task, seed); d = fed[0]["Xtr"].shape[1]
    return json.dumps(cfg, sort_keys=True), fl.run_method("fedavg_ft", fed, d, cfg, seed, SET[task]["rounds"], SET[task]["eval_every"])["val_auc"]


def main():
    import sys
    t0 = time.time(); tasks = sys.argv[1:] or ["T1", "T2", "T3", "T4"]
    fcfg = os.path.join(RES, "configs_ft.json"); cfgs = json.load(open(fcfg)) if os.path.exists(fcfg) else {}
    for task in tasks:
        best = json.load(open(os.path.join(RES, f"configs_{task}.json")))["best"]
        base = {k: v for k, v in best["fedavg"].items() if not k.startswith("_")}
        grid = [{**base, "ft_epochs": e} for e in ((1, 3, 10, 30) if task == "T5" else (1, 3, 10) if task in ("T1", "T4") else (1, 3))]
        with Pool(9) as p:
            out = p.map(tune_job, [(task, c, s) for c in grid for s in (0, 1)], chunksize=1)
        sc = {}
        for c, v in out:
            sc.setdefault(c, []).append(v)
        bc = json.loads(max(sc, key=lambda c: np.mean(sc[c]))); cfgs[task] = bc
        jobs = [(task, "fedavg_ft", "fedavg_ft", bc, s) for s in SET[task]["seeds"]]
        with Pool(9) as p:
            rows = p.map(linear_exp.run_job, jobs, chunksize=1)
        df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, f"runs_{task}_ft.csv"), index=False)
        print(task, "ft_epochs", bc["ft_epochs"], "AUROC %.4f +- %.4f" % (df.auroc.mean(), df.auroc.std()), "%.0fs" % (time.time() - t0), flush=True)
    json.dump(cfgs, open(fcfg, "w"), indent=1)


if __name__ == "__main__":
    main()
