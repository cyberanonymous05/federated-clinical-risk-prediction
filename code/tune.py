"""Validation-only hyper-parameter selection. Usage: python tune.py T1|T2|T3"""
import sys, json, os, time
import numpy as np
from multiprocessing import Pool
import data, fl
from settings import SET, TUNE_SEEDS, MAIN_METHODS, grid, TAU

RES = os.environ.get("HIPERC_RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))


def job(args):
    task, method, cfg, seed = args
    fed, _, info = data.build(task, seed)
    d = fed[0]["Xtr"].shape[1]
    r = fl.run_method(method, fed, d, cfg, seed, SET[task]["rounds"], SET[task]["eval_every"])
    return (task, method, json.dumps(cfg, sort_keys=True), seed, r["val_auc"])


def main(task):
    os.makedirs(RES, exist_ok=True)
    t0 = time.time()
    jobs = [(task, m, c, s) for m in MAIN_METHODS for c in grid(m, task) for s in TUNE_SEEDS]
    with Pool(9) as p:
        out = p.map(job, jobs, chunksize=1)
    table = {}
    for _, m, c, s, v in out:
        table.setdefault(m, {}).setdefault(c, []).append(v)
    best = {}
    for m, d in table.items():
        sc = {c: float(np.mean(v)) for c, v in d.items()}
        c = max(sc, key=sc.get); best[m] = json.loads(c)
        best[m]["_val_auc"] = sc[c]; best[m]["_n_configs"] = len(sc)
    # stage 2 for HiPerC-FL: causal-regulariser coefficient on top of the best stage-1 config
    hb = {k: v for k, v in best["hiperc"].items() if not k.startswith("_")}
    jobs = [(task, "hiperc", {**hb, "lam_c": lc}, s) for lc in (0.0, 0.3, 1.0, 3.0, 10.0) for s in TUNE_SEEDS]
    with Pool(9) as p:
        out2 = p.map(job, jobs, chunksize=1)
    sc2 = {}
    for _, m, c, s, v in out2:
        sc2.setdefault(json.loads(c)["lam_c"], []).append(v)
    sc2 = {k: float(np.mean(v)) for k, v in sc2.items()}
    best["hiperc_stage1_lamc0"] = dict(best["hiperc"])           # used for the lam_c = 0 ablation
    lc = max(sc2, key=sc2.get)
    best["hiperc"] = {**hb, "lam_c": lc, "_val_auc": sc2[lc], "_n_configs": best["hiperc"]["_n_configs"] + 5}
    best["_lamc_scores"] = {str(k): v for k, v in sc2.items()}
    json.dump(dict(best=best, table={m: {c: float(np.mean(v)) for c, v in d.items()} for m, d in table.items()}),
              open(os.path.join(RES, f"configs_{task}.json"), "w"), indent=1)
    print(task, "tuning done %.0fs" % (time.time() - t0))
    for m, c in best.items():
        print(" ", m, c)


if __name__ == "__main__":
    main(sys.argv[1])
