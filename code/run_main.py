"""Main experiments (E1, E2, E4, E8 raw data). Usage: python run_main.py T1|T2|T3 [spec_set]

spec sets
  main      : all methods with tuned configs (E1/E2) + HiPerC-FL ablations (E4)
  skew      : Dirichlet label-skew sensitivity (T2/T3 only)
Writes  results/runs_<task>_<set>.csv  and  results/preds/<task>_<set>/<spec>_s<seed>.npz
"""
import sys, os, json, time
import numpy as np
import pandas as pd
from multiprocessing import Pool
import data, fl
from metrics import youden_threshold, full_metrics, auroc
from settings import SET, MAIN_METHODS

RES = os.environ.get("HIPERC_RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))


def load_cfg(task):
    return json.load(open(os.path.join(RES, f"configs_{task}.json")))["best"]


def specs_for(task, which):
    best = load_cfg(task)
    clean = lambda c: {k: v for k, v in c.items() if not k.startswith("_")}
    sp = {}
    if which in ("main", "skew"):
        for m in MAIN_METHODS:
            sp[m] = (m, clean(best[m]))
    if which == "main":
        h = clean(best["hiperc"])
        sp["hiperc_lamc0"] = ("hiperc", clean(best["hiperc_stage1_lamc0"]))
        sp["hiperc_flat"] = ("hiperc", {**h, "tau": 1})
        sp["hiperc_hospital"] = ("hiperc", {**h, "mode": "hospital"})
        sp["hiperc_global"] = ("hiperc", {**h, "mode": "global"})
    if which == "skew":
        sp = {k: v for k, v in sp.items() if k in ("fedavg", "ditto", "hierfavg", "hiperc", "local", "ifca")}
    return sp


def job(args):
    task, which, name, method, cfg, seed = args
    kw = {}
    if which == "skew":
        kw = dict(alpha=ALPHA)
    fed, _, info = data.build(task, seed, **kw)
    d = fed[0]["Xtr"].shape[1]
    t0 = time.time()
    r = fl.run_method(method, fed, d, cfg, seed, SET[task]["rounds"], SET[task]["eval_every"])
    yv = np.concatenate([c["yva"].numpy() for c in fed]); pv = np.concatenate(r["val"])
    y = np.concatenate([c["yte"].numpy() for c in fed]); p = np.concatenate(r["test"])
    thr = youden_threshold(yv, pv)
    row = dict(task=task, spec=name, method=method, seed=seed, best_round=r["round"], val_auc=r["val_auc"],
               wan_MB=r["total_wan"] / 1e6, lan_MB=r["total_lan"] / 1e6, secs=time.time() - t0,
               **full_metrics(y, p, thr))
    wc = [auroc(c["yte"].numpy(), pp) for c, pp in zip(fed, r["test"])]
    row["auroc_within_client"] = float(np.nanmean(wc))      # prevalence-independent: ranks patients inside one client only
    hosp = np.concatenate([c["meta_te"]["hosp"] for c in fed])
    for h in range(info["n_hosp"]):
        m = hosp == h
        row[f"auroc_h{h}"] = auroc(y[m], p[m])
    meta = {k: np.concatenate([c["meta_te"][k] for c in fed]) for k in fed[0]["meta_te"]}
    out = os.path.join(RES, "preds", f"{task}_{which}"); os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, f"{name}_s{seed}.npz"), y=y.astype(np.int8), p=p.astype(np.float32),
                        yv=yv.astype(np.int8), pv=pv.astype(np.float32), thr=thr,
                        curve=np.array(r["curve"], dtype=float), **meta)
    return row


ALPHA = 0.5


def main(task, which="main"):
    os.makedirs(RES, exist_ok=True)
    sp = specs_for(task, which)
    seeds = SET[task]["seeds"] if which == "main" else SET[task]["seeds"][:5]
    jobs = [(task, which, n, m, c, s) for n, (m, c) in sp.items() for s in seeds]
    order = {"ditto": 0, "pfedme": 0, "hiperc": 0, "ifca": 1, "perfedavg": 1}
    jobs.sort(key=lambda j: order.get(j[3], 2))
    t0 = time.time()
    with Pool(9) as p:
        rows = p.map(job, jobs, chunksize=1)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, f"runs_{task}_{which}.csv"), index=False)
    print(task, which, "done in %.0fs" % (time.time() - t0), len(df), "runs")
    g = df.groupby("spec")[["auroc", "auprc", "brier", "ece", "wan_MB"]].agg(["mean", "std"]).round(3)
    print(g.to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "main")
