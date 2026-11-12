"""NHANES (T4) added experiments: (A) linear-predictor variants under the standard split protocol,
(B) temporal external validation: train on cycles 2005-2012 (0-3), test on 2013-2014, 2015-2016, 2017-2018 (4-6).
Configurations: tuned on validation data (tune.py T4 for the MLP methods; linear_exp.tune for the linear ones).
Usage: python nhanes_exp.py"""
import os, json, time
import numpy as np, pandas as pd
from multiprocessing import Pool
import data, fl, linear_exp
from metrics import auroc, calib_slope_intercept, youden_threshold
from settings import SET

RES = os.environ.get("HIPERC_RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
HOLD = [4, 5, 6]


def temporal_job(a):
    name, m, cfg, seed = a
    fed, ext, _ = data.build("T4", seed, holdout=HOLD); d = fed[0]["Xtr"].shape[1]
    r = fl.run_method(m, fed, d, cfg, seed, SET["T4"]["rounds"], SET["T4"]["eval_every"], ext_X=ext["X"])
    y = ext["y"].numpy(); p = np.asarray(r["ext"]); rows = []
    for c in HOLD:
        k = ext["cycle"] == c; yy, pp = y[k], p[k]; sl, ic = calib_slope_intercept(yy, pp)
        rows.append(dict(spec=name, method=m, seed=seed, cycle=c, auroc=auroc(yy, pp), brier=float(np.mean((pp - yy) ** 2)), cal_slope=sl, cal_int=ic,
                         mean_pred=float(pp.mean()), prev=float(yy.mean()), n=int(k.sum())))
    return rows


def main():
    t0 = time.time()
    best = json.load(open(os.path.join(RES, "configs_T4.json")))["best"]
    cl = lambda c: {k: v for k, v in c.items() if not k.startswith("_")}
    # (A) linear variants, standard protocol
    lb = linear_exp.tune("T4"); json.dump(lb, open(os.path.join(RES, "configs_T4_linear.json"), "w"), indent=1)
    names = {"fedavg": "lin_fedavg", "hierfavg": "lin_hierfavg", "hiperc": "lin_hiperc", "hiperc_lamc0": "lin_hiperc_lamc0"}
    meth = {"fedavg": "fedavg", "hierfavg": "hierfavg", "hiperc": "hiperc", "hiperc_lamc0": "hiperc"}
    jobs = [("T4", names[k], meth[k], lb[k], s) for k in names for s in SET["T4"]["seeds"]]
    with Pool(9) as p:
        rows = p.map(linear_exp.run_job, jobs, chunksize=1)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "runs_T4_linear.csv"), index=False); print("A done %.0fs" % (time.time() - t0), flush=True)
    # (B) temporal external validation
    sp = {"logreg": ("logreg", {}), "centralized": ("centralized", cl(best["centralized"])), "fedavg": ("fedavg", cl(best["fedavg"])),
          "hierfavg": ("hierfavg", cl(best["hierfavg"])), "ditto": ("ditto", cl(best["ditto"])), "hiperc_lamc0": ("hiperc", cl(best["hiperc_stage1_lamc0"])),
          "hiperc": ("hiperc", cl(best["hiperc"])), "lin_fedavg": ("fedavg", lb["fedavg"]), "lin_hierfavg": ("hierfavg", lb["hierfavg"]),
          "lin_hiperc": ("hiperc", lb["hiperc"])}
    jobs = [(n, m, c, s) for n, (m, c) in sp.items() for s in SET["T4"]["seeds"]]
    with Pool(9) as p:
        out = p.map(temporal_job, jobs, chunksize=1)
    df = pd.DataFrame([r for o in out for r in o]); df.to_csv(os.path.join(RES, "nhanes_temporal.csv"), index=False)
    print("B done %.0fs" % (time.time() - t0)); print(df.groupby("spec")[["auroc", "brier", "cal_int"]].mean().round(4))


if __name__ == "__main__":
    main()
