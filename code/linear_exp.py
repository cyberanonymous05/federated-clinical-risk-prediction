"""Additional experiment (added after the main results):
does the federated *framework* keep its benefits when the predictor is a plain logistic regression (the model that
transported best to a held-out hospital)?  Same protocol as the MLP runs: validation-only tuning on seeds 0-1,
then all seeds; T1 additionally leave-one-hospital-out.   Usage: python linear_exp.py"""
import os, json, time
import numpy as np, pandas as pd
from multiprocessing import Pool
import data, fl
from metrics import full_metrics, youden_threshold, auroc
from settings import SET, TAU

RES = os.environ.get("HIPERC_RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
LRS = [0.03, 0.1, 0.3]


def base(task, lr):
    s = SET[task]
    return dict(lr=lr, epochs=s["epochs"], bs=s["bs"], frac=s["frac"], model="linear")


def grid(task):
    g = {"fedavg": [base(task, lr) for lr in LRS], "hierfavg": [{**base(task, lr), "tau": TAU} for lr in LRS],
         "hiperc": [{**base(task, lr), "tau": TAU, "lam_p": m, "lam_c": 0.0} for lr in LRS for m in (0.01, 0.1, 1.0)]}
    return g


def tune_job(a):
    task, m, cfg, seed = a
    fed, _, _ = data.build(task, seed); d = fed[0]["Xtr"].shape[1]
    return m, json.dumps(cfg, sort_keys=True), fl.run_method(m, fed, d, cfg, seed, SET[task]["rounds"], SET[task]["eval_every"])["val_auc"]


def tune(task):
    jobs = [(task, m, c, s) for m, cs in grid(task).items() for c in cs for s in (0, 1)]
    with Pool(9) as p:
        out = p.map(tune_job, jobs, chunksize=1)
    sc = {}
    for m, c, v in out:
        sc.setdefault(m, {}).setdefault(c, []).append(v)
    best = {m: json.loads(max(d, key=lambda c: np.mean(d[c]))) for m, d in sc.items()}
    hb = best["hiperc"]
    out2 = []
    with Pool(9) as p:
        out2 = p.map(tune_job, [(task, "hiperc", {**hb, "lam_c": lc}, s) for lc in (0.0, 0.3, 1.0, 3.0, 10.0) for s in (0, 1)], chunksize=1)
    s2 = {}
    for m, c, v in out2:
        s2.setdefault(json.loads(c)["lam_c"], []).append(v)
    lc = max(s2, key=lambda k: np.mean(s2[k]))
    best["hiperc_lamc0"] = dict(hb); best["hiperc"] = {**hb, "lam_c": lc}
    return best


def run_job(a):
    task, name, m, cfg, seed = a
    fed, _, info = data.build(task, seed); d = fed[0]["Xtr"].shape[1]
    r = fl.run_method(m, fed, d, cfg, seed, SET[task]["rounds"], SET[task]["eval_every"])
    yv = np.concatenate([c["yva"].numpy() for c in fed]); pv = np.concatenate(r["val"])
    y = np.concatenate([c["yte"].numpy() for c in fed]); p = np.concatenate(r["test"])
    row = dict(task=task, spec=name, method=m, seed=seed, best_round=r["round"], val_auc=r["val_auc"], wan_MB=r["total_wan"] / 1e6,
               lan_MB=r["total_lan"] / 1e6, secs=0, **full_metrics(y, p, youden_threshold(yv, pv)))
    meta = {k: np.concatenate([c["meta_te"][k] for c in fed]) for k in fed[0]["meta_te"]}
    out = os.path.join(RES, "preds", f"{task}_main"); os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, f"{name}_s{seed}.npz"), y=y.astype(np.int8), p=p.astype(np.float32), yv=yv.astype(np.int8),
                        pv=pv.astype(np.float32), thr=youden_threshold(yv, pv), curve=np.array(r["curve"], dtype=float), **meta)
    return row


def loho_job(a):
    name, m, cfg, h, seed = a
    fed, ext, _ = data.build_heart(seed, holdout=h); d = fed[0]["Xtr"].shape[1]
    r = fl.run_method(m, fed, d, cfg, seed, SET["T1"]["rounds"], SET["T1"]["eval_every"], ext_X=ext["X"])
    yv = np.concatenate([c["yva"].numpy() for c in fed]); pv = np.concatenate(r["val"])
    return dict(spec=name, method=m, holdout=h, seed=seed, **full_metrics(ext["y"].numpy(), r["ext"], youden_threshold(yv, pv)))


def main():
    t0 = time.time(); cfgs = {}
    for task in ("T1", "T2", "T3"):
        best = tune(task); cfgs[task] = best
        names = {"fedavg": "lin_fedavg", "hierfavg": "lin_hierfavg", "hiperc": "lin_hiperc", "hiperc_lamc0": "lin_hiperc_lamc0"}
        meth = {"fedavg": "fedavg", "hierfavg": "hierfavg", "hiperc": "hiperc", "hiperc_lamc0": "hiperc"}
        jobs = [(task, names[k], meth[k], best[k], s) for k in names for s in SET[task]["seeds"]]
        with Pool(9) as p:
            rows = p.map(run_job, jobs, chunksize=1)
        df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, f"runs_{task}_linear.csv"), index=False)
        print(task, "done %.0fs" % (time.time() - t0)); print(df.groupby("spec")[["auroc", "brier", "ece", "wan_MB"]].mean().round(4))
    json.dump(cfgs, open(os.path.join(RES, "configs_linear.json"), "w"), indent=1)
    b = cfgs["T1"]; names = {"fedavg": "lin_fedavg", "hierfavg": "lin_hierfavg", "hiperc_lamc0": "lin_hiperc_lamc0", "hiperc": "lin_hiperc"}
    meth = {"fedavg": "fedavg", "hierfavg": "hierfavg", "hiperc": "hiperc", "hiperc_lamc0": "hiperc"}
    jobs = [(names[k], meth[k], b[k], h, s) for k in names for h in range(4) for s in SET["T1"]["seeds"]]
    with Pool(9) as p:
        rows = p.map(loho_job, jobs, chunksize=1)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, "loho_T1_linear.csv"), index=False)
    print(df.pivot_table(index="spec", columns="holdout", values="auroc", aggfunc="mean").round(3)); print(df.groupby("spec").auroc.mean().round(3))


if __name__ == "__main__":
    main()
