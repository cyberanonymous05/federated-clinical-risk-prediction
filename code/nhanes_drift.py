"""NHANES real-drift stream: the seven survey cycles arrive one after another. Six persistent clients (two hospitals, three clients each)
receive a random share of every cycle's participants. At stage t (cycle t) each model is first evaluated on a held-out 30% evaluation
pool of that cycle (prequential, before it has seen cycle t), then updated with the 70% training pool.
Methods: frozen_fedavg, frozen_hiperc (cycle 0 only), retrain_fedavg (each cycle only), cumulative_fedavg (all cycles so far),
hiperc_naive, hiperc_continual (replay + anchor).  Usage: python nhanes_drift.py"""
import os, json, time
import numpy as np, pandas as pd, torch
from multiprocessing import Pool
import data, fl, drift
from metrics import auroc, calib_slope_intercept

RES = os.environ.get("HIPERC_RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
K = 6


def job(args):
    seed, cfg_h, cfg_f = args
    fed0, _, info = data.build("T4", seed); raw = info["raw"]; Xs, y, cyc = raw["Xs"], raw["y"], raw["cyc"]
    rng = np.random.RandomState(500 + seed); d = Xs.shape[1]
    stages = []
    for c in range(7):
        idx = np.where(cyc == c)[0]; rng.shuffle(idx); ntr = int(0.7 * len(idx)); tr, te = idx[:ntr], idx[ntr:]
        trp = data._round_robin(tr, y, K, rng); tep = data._round_robin(te, y, K, rng)
        stages.append((trp, tep))
    def staged(c, cum=None):
        trp, tep = stages[c]; out = []
        for k in range(K):
            ii = trp[k] if cum is None else np.concatenate([stages[q][0][k] for q in range(c + 1)])
            out.append(dict(hosp=k % 2, Xtr=torch.tensor(Xs[ii], dtype=torch.float32), ytr=torch.tensor(y[ii], dtype=torch.float32),
                            Xte=torch.tensor(Xs[tep[k]], dtype=torch.float32), yte=torch.tensor(y[tep[k]], dtype=torch.float32),
                            Xva=torch.tensor(Xs[tep[k]][:10], dtype=torch.float32), yva=torch.tensor(y[tep[k]][:10], dtype=torch.float32)))
        return out
    hp = dict(cfg_h, epochs=1, frac=1.0, tau=5); fa = dict(cfg_f, epochs=1, frac=1.0)
    f0 = staged(0)
    objs = {"frozen_fedavg": fl.FedAvg(f0, d, fa, seed), "retrain_fedavg": fl.FedAvg(f0, d, fa, seed), "cumulative_fedavg": fl.FedAvg(f0, d, fa, seed),
            "frozen_hiperc": fl.HiPerC(f0, d, hp, seed), "hiperc_naive": drift.HiPerCContinual(f0, d, hp, seed, replay=False),
            "hiperc_continual": drift.HiPerCContinual(f0, d, hp, seed, replay=True)}
    t = {k: 0 for k in objs}; rows = []
    for c in range(7):
        fed_c = staged(c); fed_cum = staged(c, cum=True); yS = np.concatenate([x["yte"].numpy() for x in fed_c])
        for k, o in objs.items():
            o.fed = fed_cum if k == "cumulative_fedavg" else fed_c
            o.n = np.array([len(x["ytr"]) for x in o.fed], dtype=float)
            o.fed_eval = fed_c
            saved = o.fed; o.fed = fed_c
            pb = np.concatenate(o.predict("te")) if c > 0 else None
            o.fed = saved
            ab = auroc(yS, pb) if c > 0 else np.nan
            sl, ic = calib_slope_intercept(yS, pb) if c > 0 else (np.nan, np.nan)
            frozen = k.startswith("frozen")
            if not (frozen and c > 0):
                for _ in range(40 if c == 0 else 8):
                    t[k] += 1; o.round(t[k])
                if hasattr(o, "remember"):
                    o.remember()
            o.fed = fed_c
            pa = np.concatenate(o.predict("te"))
            rows.append(dict(seed=seed, cycle=c, method=k, auroc_before=ab, cal_int_before=ic, auroc_after=auroc(yS, pa),
                             brier_before=float(np.mean((pb - yS) ** 2)) if c > 0 else np.nan, prev=float(yS.mean())))
    return rows


def main():
    best = json.load(open(os.path.join(RES, "configs_T4.json")))["best"]
    ch = {k: v for k, v in best["hiperc"].items() if not k.startswith("_")}; cf = {k: v for k, v in best["fedavg"].items() if not k.startswith("_")}
    import sys
    same = len(sys.argv) > 1 and sys.argv[1] == "samelr"      # sensitivity: HiPerC-FL uses FedAvg's learning rate
    if same:
        ch = dict(ch, lr=cf["lr"])
    t0 = time.time()
    with Pool(9) as p:
        rows = sum(p.map(job, [(s, ch, cf) for s in range(10)], chunksize=1), [])
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, "nhanes_drift_samelr.csv" if same else "nhanes_drift.csv"), index=False)
    print("nhanes drift done %.0fs" % (time.time() - t0))
    print(df[df.cycle > 0].groupby("method")[["auroc_before", "brier_before", "cal_int_before"]].mean().round(4))


if __name__ == "__main__":
    main()
