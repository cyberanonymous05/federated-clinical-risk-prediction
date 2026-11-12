"""E5: 12-stage simulated population drift on T2 (diabetes, BRFSS) with an explicit, reproducible protocol.

Protocol (stage t = 1..12, s = (t-1)/11):
  Each record in a client's pool receives sampling weight
        w = exp( gamma * (2s-1) * z_age ) * ( 1 + 0.3 * (2s-1) * (2y-1) )
  z_age = standardised BRFSS age band, gamma = 0.8.  At s=0 the population is young-weighted with 30% fewer positives
  than the base rate; at s=1 it is old-weighted with 30% more positives. This is covariate + prior-probability shift
  (NOT a change of P(y|x)).  Per stage and client: a training chunk of 6% of the client's train pool is drawn without
  replacement with these weights; the evaluation sample of the stage is the client's test pool re-sampled (with
  replacement, same size) with the same weights.
Methods
  frozen_fedavg      FedAvg on stage-1 chunks only, never updated
  retrain_fedavg     FedAvg warm-started on each new chunk (8 rounds), forgets older chunks
  frozen_hiperc      HiPerC-FL on stage-1 only
  hiperc_naive       HiPerC-FL, personal models fine-tuned on the new chunk only (no replay, no anchor)
  hiperc_continual   HiPerC-FL, personal models fine-tuned on new chunk + reservoir replay buffer (500/client)
                     with an L2 anchor to the previous personal model
Metrics: AUROC on stage t evaluation sample before adapting to chunk t (prequential, primary) and after.
Usage: python drift.py
"""
import os, json, time
import numpy as np
import pandas as pd
import torch
from multiprocessing import Pool
import data, fl
from metrics import auroc
from settings import SET

RES = os.environ.get("HIPERC_RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
GAMMA, PREV, CH, STAGES = 0.8, 0.3, 0.06, 12


def weights(age, y, s, mu, sd):
    z = (age - mu) / sd
    return np.exp(GAMMA * (2 * s - 1) * z) * (1 + PREV * (2 * s - 1) * (2 * y - 1))


class HiPerCContinual(fl.HiPerC):
    """HiPerC-FL with continual personal-model adaptation (replay + anchor)."""
    def __init__(self, *a, replay=True, anchor=1.0, buf=500, **k):
        super().__init__(*a, **k)
        self.replay, self.anchor, self.buf_size = replay, anchor, buf
        self.buf = [None] * self.N

    def round(self, t):
        S = self.sample()
        old = [p.clone() for p in self.phi]
        res = self.hospital_round(S)
        for i in S:
            c = self.fed[i]; X, y = c["Xtr"], c["ytr"]
            if self.replay and self.buf[i] is not None:
                bx, by = self.buf[i]
                m = min(len(by), len(y)); sel = torch.randperm(len(by), generator=self.gen)[:m]
                X = torch.cat([X, bx[sel]]); y = torch.cat([y, by[sel]])
                mu = self.lam_p + self.anchor
                center = (self.lam_p * old[self.hosp[i]] + self.anchor * self.v[i]) / mu
            else:
                mu, center = self.lam_p, old[self.hosp[i]]
            self.v[i], _, _ = fl.sgd(self.v[i], X, y, self.E, self.lr, self.bs, self.gen, mu=mu, center=center)
        for h, r in res.items():
            self.phi[h] = r[0]
        if t % self.tau == 0:
            self.sync(res)

    def remember(self):
        """Reservoir-style update of the replay buffer with the current chunk."""
        if not self.replay:
            return
        for i, c in enumerate(self.fed):
            X, y = c["Xtr"], c["ytr"]
            if self.buf[i] is not None:
                X = torch.cat([self.buf[i][0], X]); y = torch.cat([self.buf[i][1], y])
            sel = torch.randperm(len(y), generator=self.gen)[:self.buf_size]
            self.buf[i] = (X[sel], y[sel])


def stage_s(schedule, stage):
    """Population position s in [0,1] for a stage (see module docstring for the base protocol)."""
    lin = (stage - 1) / (STAGES - 1)
    if schedule == "linear":
        return lin
    if schedule == "cyclic":                         # young -> old -> young again
        return 1 - abs(2 * lin - 1)
    if schedule == "abrupt":                         # young for stages 1-6, old from stage 7
        return 0.0 if stage <= STAGES // 2 else 1.0
    return 0.5                                       # "criteria": stationary population, labelling rule changes


CRIT_Q = 0.12      # flip probability at the last stage in the "criteria" scenario


def job(args):
    seed, cfg, schedule = args
    ch_frac = 0.005 if schedule.endswith("_small") else CH          # '_small': scarce new data per stage (0.5% of the pool)
    base_schedule = schedule.replace("_small", "")
    fed0, _, info = data.build("T2", seed)
    d = fed0[0]["Xtr"].shape[1]; raw = info["raw"]
    rng = np.random.RandomState(1000 + seed)
    age_all = raw["age"].astype(float); mu, sd = age_all.mean(), age_all.std()
    C = fed0                                        # base pools
    bmi_all = info["raw"]["X"][:, info["raw"]["feat"].index("BMI")]; bmi_med = np.median(bmi_all)
    def flip(y, idx_rows, stage):
        """'criteria' scenario: a new diagnostic rule labels more high-BMI people as diabetic (P(y|x) changes)."""
        if base_schedule != "criteria":
            return y
        q = CRIT_Q * (stage - 1) / (STAGES - 1)
        y = y.clone(); hi = torch.tensor(bmi_all[idx_rows] > bmi_med)
        u = torch.tensor(rng.rand(len(y)) < q)
        y[(y == 0) & hi & u] = 1.0
        return y
    def draw(s, stage):
        chunks, evals = [], []
        for c in C:
            idx = c["idx_tr"]; w = weights(age_all[idx], raw["y"][idx], s, mu, sd); w /= w.sum()
            m = int(ch_frac * len(idx)); pick = rng.choice(len(idx), m, replace=False, p=w)
            chunks.append(pick)
            ide = c["idx_te"]; we = weights(age_all[ide], raw["y"][ide], s, mu, sd); we /= we.sum()
            evals.append(rng.choice(len(ide), len(ide), replace=True, p=we))
        return chunks, evals
    def staged(chunks, evals, stage):
        out = []
        for c, ch, ev in zip(C, chunks, evals):
            out.append(dict(hosp=c["hosp"], Xtr=c["Xtr"][ch], ytr=flip(c["ytr"][ch], c["idx_tr"][ch], stage),
                            Xte=c["Xte"][ev], yte=flip(c["yte"][ev], c["idx_te"][ev], stage),
                            Xva=c["Xva"][:10], yva=c["yva"][:10]))
        return out
    hp = dict(lr=cfg["lr"], epochs=1, bs=cfg["bs"], frac=cfg["frac"], tau=5, lam_p=cfg["lam_p"], lam_c=cfg["lam_c"])
    fa = dict(lr=cfg["lr_fedavg"], epochs=1, bs=cfg["bs"], frac=cfg["frac"])
    def mk(kind):
        if kind.endswith("fedavg"):
            return fl.FedAvg(C, d, fa, seed)
        if kind == "hiperc_continual":
            return HiPerCContinual(C, d, hp, seed, replay=True)
        if kind == "hiperc_naive":
            return HiPerCContinual(C, d, hp, seed, replay=False)
        return fl.HiPerC(C, d, hp, seed)
    kinds = ["frozen_fedavg", "retrain_fedavg", "frozen_hiperc", "hiperc_naive", "hiperc_continual"]
    objs = {k: mk(k) for k in kinds}
    t = {k: 0 for k in kinds}; rows = []
    for stage in range(1, STAGES + 1):
        s = stage_s(base_schedule, stage)
        chunks, evals = draw(s, stage); fedS = staged(chunks, evals, stage)
        yS = np.concatenate([c["yte"].numpy() for c in fedS])
        for k, o in objs.items():
            o.fed = fedS; o.n = np.array([len(c["ytr"]) for c in fedS], dtype=float)
            o.predict_before = np.concatenate(o.predict("te")) if stage > 1 else None
            a_before = auroc(yS, o.predict_before) if stage > 1 else np.nan
            frozen = k.startswith("frozen")
            nr = 40 if stage == 1 else 8
            if not (frozen and stage > 1):
                for _ in range(nr):
                    t[k] += 1; o.round(t[k])
                if hasattr(o, "remember"):
                    o.remember()
            a_after = auroc(yS, np.concatenate(o.predict("te")))
            rows.append(dict(seed=seed, stage=stage, s=s, method=k, auroc_before=a_before, auroc_after=a_after,
                             prev=float(yS.mean()), schedule=schedule))
    return rows


def main():
    best = json.load(open(os.path.join(RES, "configs_T2.json")))["best"]
    cfg = {k: v for k, v in best["hiperc"].items() if not k.startswith("_")}
    cfg["lr_fedavg"] = best["fedavg"]["lr"]
    t0 = time.time()
    import sys
    todo = sys.argv[1:] or ["linear", "cyclic", "abrupt", "criteria", "cyclic_small", "abrupt_small", "criteria_small"]
    for schedule in todo:
        jobs = [(s, cfg, schedule) for s in range(5)]
        with Pool(5) as p:
            rows = sum(p.map(job, jobs, chunksize=1), [])
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(RES, "drift_T2.csv" if schedule == "linear" else f"drift_T2_{schedule}.csv"), index=False)
        print(schedule, "done %.0fs" % (time.time() - t0))
        print(df[df.stage > 1].groupby("method")[["auroc_before", "auroc_after"]].mean().round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
