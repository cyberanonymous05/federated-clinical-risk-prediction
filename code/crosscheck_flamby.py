"""Independent cross-check of the federated algorithms against the FLamby reference implementations (Ogier du Terrail et al., 2022).
Same data and splits (T1), same network, same learning rate, same number of rounds, same number of local updates per round; only the
FL code differs (FLamby strategies vs. ours). The FLamby repository is cloned to /tmp/flamby_try/FLamby (not installed).
Compares the test AUROC of the final global model after a fixed number of rounds (no checkpoint selection) over 20 seeds.
Usage: python crosscheck_flamby.py"""
import os, sys, json, time
import numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
sys.path.insert(0, "/tmp/flamby_try/FLamby")
from flamby.strategies.fed_avg import FedAvg as FLFedAvg
from flamby.strategies.fed_prox import FedProx as FLFedProx
import data, fl
from metrics import auroc
from scipy import stats

RES = os.environ.get("HIPERC_RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
ROUNDS, LR, BS, UPDATES, MU = 60, 0.1, 32, 4, 0.1
ALIGN_INIT = (len(sys.argv) > 1 and sys.argv[1] == 'same_init')


class Net(nn.Module):
    def __init__(self, d):
        super().__init__(); self.f = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.f(x).squeeze(-1)


def load_vec(net, vec):
    """Copy our flat parameter vector (fl.init_vec layout) into the nn.Sequential so both implementations start from identical weights."""
    i = 0; lin = [m for m in net.f if isinstance(m, nn.Linear)]
    with torch.no_grad():
        for m in lin:
            fi, fo = m.in_features, m.out_features
            m.weight.copy_(vec[i:i + fi * fo].view(fi, fo).T); i += fi * fo
            m.bias.copy_(vec[i:i + fo]); i += fo
    return net


def run_flamby(fed, d, seed, which):
    torch.manual_seed(seed)
    dls = [DataLoader(TensorDataset(c["Xtr"], c["ytr"]), batch_size=BS, shuffle=True) for c in fed]
    kw = dict(training_dataloaders=dls, model=(load_vec(Net(d), fl.init_vec(d, seed)) if ALIGN_INIT else Net(d)), loss=nn.BCEWithLogitsLoss(), optimizer_class=torch.optim.SGD, learning_rate=LR,
              num_updates=UPDATES, nrounds=ROUNDS, seed=seed)
    s = FLFedAvg(**kw) if which == "fedavg" else FLFedProx(**kw, mu=MU)
    m = s.run()[0].eval(); X = torch.cat([c["Xte"] for c in fed]); y = torch.cat([c["yte"] for c in fed]).numpy()
    with torch.no_grad():
        return auroc(y, torch.sigmoid(m(X)).numpy())


def run_ours(fed, d, seed, which):
    class FixedUpdates(fl.FedAvg):                            # same number of local updates per round as FLamby
        def local(self, i, w0, shared=True, **kw):
            c = self.fed[i]; w = w0.clone(); gen = self.gen; n = len(c["ytr"]); steps = 0; center = w0
            while steps < UPDATES:
                perm = torch.randperm(n, generator=gen)
                for j in range(0, n, BS):
                    b = perm[j:j + BS]; _, g = fl.loss_grad(w, c["Xtr"][b], c["ytr"][b])
                    if which == "fedprox":
                        g = g + MU * (w - center)
                    w = w - LR * g; steps += 1
                    if steps >= UPDATES:
                        break
            return w, 0.0, steps
    o = FixedUpdates(fed, d, dict(epochs=2, lr=LR, bs=BS, frac=1.0), seed)
    for t in range(1, ROUNDS + 1):
        o.round(t)
    X = torch.cat([c["Xte"] for c in fed]); y = torch.cat([c["yte"] for c in fed]).numpy()
    return auroc(y, fl.predict_prob(o.g, X))


rows = []; t0 = time.time()
for seed in range(20):
    fed, _, _ = data.build("T1", seed); d = fed[0]["Xtr"].shape[1]
    for which in ("fedavg", "fedprox"):
        rows.append(dict(seed=seed, algo=which, flamby=run_flamby(fed, d, seed, which), ours=run_ours(fed, d, seed, which)))
df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, "crosscheck_flamby" + ("_same_init" if ALIGN_INIT else "_default_init") + ".csv"), index=False)
out = {}
for which, g in df.groupby("algo"):
    dd = (g.ours - g.flamby).values; t = stats.ttest_1samp(dd, 0.0)
    out[which] = dict(n=len(g), flamby_mean=float(g.flamby.mean()), flamby_sd=float(g.flamby.std()), ours_mean=float(g.ours.mean()), ours_sd=float(g.ours.std()),
                      diff_mean=float(dd.mean()), diff_sd=float(dd.std()), p=float(t.pvalue), corr=float(np.corrcoef(g.ours, g.flamby)[0, 1]),
                      ci=[float(dd.mean() - 2.093 * dd.std() / np.sqrt(len(dd))), float(dd.mean() + 2.093 * dd.std() / np.sqrt(len(dd)))])
json.dump(out, open(os.path.join(RES, "crosscheck_flamby" + ("_same_init" if ALIGN_INIT else "_default_init") + ".json"), "w"), indent=1)
print(json.dumps(out, indent=1)); print("%.0fs" % (time.time() - t0))
