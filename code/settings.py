"""Experiment settings shared by all scripts."""
SET = {
    "T1": dict(rounds=200, eval_every=5, epochs=2, bs=32, frac=1.0, lrs=[0.03, 0.1, 0.3], seeds=list(range(20))),
    "T2": dict(rounds=100, eval_every=5, epochs=2, bs=256, frac=0.5, lrs=[0.02, 0.05, 0.1], seeds=list(range(10))),
    "T4": dict(rounds=100, eval_every=5, epochs=2, bs=64, frac=1.0, lrs=[0.03, 0.1, 0.3], seeds=list(range(10))),
    "T5": dict(rounds=100, eval_every=5, epochs=2, bs=128, frac=1.0, lrs=[0.03, 0.1, 0.3], seeds=list(range(10))),
    "T6": dict(rounds=100, eval_every=5, epochs=2, bs=128, frac=0.3, lrs=[0.03, 0.1, 0.3], seeds=list(range(5))),
    "T5H24": dict(rounds=100, eval_every=5, epochs=2, bs=128, frac=1.0, lrs=[0.03, 0.1, 0.3], seeds=list(range(10))),
    "T3": dict(rounds=100, eval_every=5, epochs=2, bs=256, frac=0.5, lrs=[0.02, 0.05, 0.1], seeds=list(range(10))),
}
TUNE_SEEDS = [0, 1]          # validation-only hyper-parameter selection (test splits never used)
TAU = 5                      # hospital -> global synchronisation period (rounds)

MAIN_METHODS = ["local", "logreg", "centralized", "fedavg", "fedprox", "scaffold", "ditto", "pfedme",
                "perfedavg", "ifca", "hierfavg", "hiperc"]


def grid(method, task):
    """Hyper-parameter grid (<= 9 configs per method; HiPerC-FL gets a 2nd stage for lam_c, see tune.py)."""
    s = SET[task]; out = []
    for lr in s["lrs"]:
        base = dict(lr=lr, epochs=s["epochs"], bs=s["bs"], frac=s["frac"])
        if method in ("fedavg", "scaffold", "perfedavg", "hierfavg"):
            out.append({**base, **({"tau": TAU} if method == "hierfavg" else {})})
        elif method == "fedprox":
            out += [{**base, "mu": m} for m in (0.01, 0.1, 1.0)]
        elif method == "ditto":
            out += [{**base, "lam": m} for m in (0.01, 0.1, 1.0)]
        elif method == "pfedme":
            out += [{**base, "lam": m} for m in (1.0, 5.0) if lr * m <= 0.5]
        elif method == "ifca":
            out += [{**base, "K": k} for k in (2, 3)]
        elif method == "hiperc":
            out += [{**base, "tau": TAU, "lam_p": m, "lam_c": 0.0} for m in (0.01, 0.1, 1.0)]
        elif method in ("local", "centralized"):
            out.append(base)
        elif method == "logreg":
            out = [dict()]
    return out


import os as _os
if _os.environ.get("REPRO_SMOKE"):          # smoke test: 2 seeds, 20 rounds (checks that the pipeline runs end to end; numbers are not the paper's)
    for _t in SET:
        SET[_t]["seeds"] = SET[_t]["seeds"][:2]; SET[_t]["rounds"] = min(SET[_t]["rounds"], 20)
