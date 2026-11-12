"""E9: privacy-utility trade-off with sample-level DP-SGD on the shared model (T2).
Noise multiplier z is calibrated by bisection so that the worst client's (epsilon, delta=1e-5) after R rounds, as given by
the Opacus RDP accountant (Poisson-sampling approximation), is <= target.  Personal models stay local and are not noised.
V-REx is switched off here (lam_c=0) because the per-hospital risk scalar is not privatised.
Usage: python dp_exp.py"""
import os, json, time, math
import numpy as np, pandas as pd
from multiprocessing import Pool
from opacus.accountants import RDPAccountant
import data, fl
from metrics import full_metrics, youden_threshold
from settings import SET

RES = os.environ.get("HIPERC_RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
DELTA = 1e-5; TARGETS = [1.0, 3.0, 8.0]; C = 1.0; SEEDS = list(range(5))


def worst_eps(z, ns, R, frac, E, bs):
    e = 0.0
    for n in ns:
        acc = RDPAccountant()
        steps = int(round(frac * R)) * E * math.ceil(n / bs)
        acc.history = [(z, min(1.0, bs / n), steps)]
        e = max(e, acc.get_epsilon(delta=DELTA))
    return e


def calibrate(ns, R, frac, E, bs, target):
    lo, hi = 0.3, 200.0
    for _ in range(40):
        mid = math.sqrt(lo * hi)
        if worst_eps(mid, ns, R, frac, E, bs) > target:
            lo = mid
        else:
            hi = mid
    return hi


def job(args):
    name, method, cfg, seed, eps, z = args
    fed, _, info = data.build("T2", seed); d = fed[0]["Xtr"].shape[1]
    r = fl.run_method(method, fed, d, {**cfg, "sdp_z": z, "sdp_C": C}, seed, SET["T2"]["rounds"], SET["T2"]["eval_every"])
    yv = np.concatenate([c["yva"].numpy() for c in fed]); pv = np.concatenate(r["val"])
    y = np.concatenate([c["yte"].numpy() for c in fed]); p = np.concatenate(r["test"])
    return dict(spec=name, method=method, eps_target=eps, z=z, seed=seed, **full_metrics(y, p, youden_threshold(yv, pv)))


def main():
    best = json.load(open(os.path.join(RES, "configs_T2.json")))["best"]
    cl = lambda c: {k: v for k, v in c.items() if not k.startswith("_")}
    s = SET["T2"]; fed, _, _ = data.build("T2", 0)
    ns = [len(c["ytr"]) for c in fed]
    zs = {e: calibrate(ns, s["rounds"], s["frac"], s["epochs"], s["bs"], e) for e in TARGETS}
    print("noise multipliers", {e: round(z, 3) for e, z in zs.items()},
          {e: round(worst_eps(z, ns, s["rounds"], s["frac"], s["epochs"], s["bs"]), 3) for e, z in zs.items()})
    sp = {"fedavg": ("fedavg", cl(best["fedavg"])), "hiperc": ("hiperc", {**cl(best["hiperc"]), "lam_c": 0.0})}
    jobs = [(n, m, c, sd, e, zs[e]) for n, (m, c) in sp.items() for e in TARGETS for sd in SEEDS]
    with Pool(9) as p:
        rows = p.map(job, jobs, chunksize=1)
    df = pd.DataFrame(rows); df["z"] = df.z.round(3)
    df.to_csv(os.path.join(RES, "dp_T2.csv"), index=False)
    json.dump({str(e): z for e, z in zs.items()}, open(os.path.join(RES, "dp_noise_multipliers.json"), "w"))
    print(df.groupby(["spec", "eps_target"])[["auroc", "brier", "ece"]].agg(["mean", "std"]).round(3).to_string())


if __name__ == "__main__":
    main()
