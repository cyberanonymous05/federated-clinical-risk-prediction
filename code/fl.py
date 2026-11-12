"""Federated learning algorithms (single-process simulation) for tabular clinical prediction.

All methods share one MLP (d -> 64 -> 32 -> 1) stored as a flat parameter vector, plain SGD, the same local
epochs and batch size, so differences come from the federation strategy, not the architecture.

Methods: local, centralized, fedavg, fedprox, scaffold, ditto, pfedme, perfedavg, ifca, hierfavg, hiperc.
Every FL method keeps a communication ledger (bytes on the WAN = to the central server, bytes on the
LAN = client <-> hospital) computed from actual tensor sizes.
"""
import math
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

torch.set_num_threads(1)
H1, H2 = 64, 32


# ------------------------------------------------------------------ model
MODEL = "mlp"                      # "mlp" (default, all main experiments) or "linear" (logistic regression trained by SGD)
INIT = "he"                        # "he" (default, all main experiments) or "torch" (PyTorch's nn.Linear default; sensitivity analysis)


def set_init(m):
    global INIT
    INIT = m


def set_model(m):
    global MODEL
    MODEL = m


def n_params(d):
    if MODEL == "linear":
        return d + 1
    return d * H1 + H1 + H1 * H2 + H2 + H2 + 1


def init_vec(d, seed):
    g = torch.Generator().manual_seed(10_000 + seed)
    if MODEL == "linear":
        return torch.cat([torch.randn(d, generator=g) * 0.01, torch.zeros(1)])
    parts = []
    for fi, fo in ((d, H1), (H1, H2), (H2, 1)):
        if INIT == "torch":                                       # U(-1/sqrt(fan_in), 1/sqrt(fan_in)) for weights and biases
            b = 1.0 / math.sqrt(fi)
            parts.append((torch.rand(fi * fo, generator=g) * 2 - 1) * b); parts.append((torch.rand(fo, generator=g) * 2 - 1) * b)
        else:
            parts.append(torch.randn(fi * fo, generator=g) * math.sqrt(2.0 / fi))
            parts.append(torch.zeros(fo))
    return torch.cat(parts)


def logits(w, X):
    d = X.shape[1]
    if MODEL == "linear":
        return X @ w[:d] + w[d]
    i = 0
    W1 = w[i:i + d * H1].view(d, H1); i += d * H1
    b1 = w[i:i + H1]; i += H1
    W2 = w[i:i + H1 * H2].view(H1, H2); i += H1 * H2
    b2 = w[i:i + H2]; i += H2
    W3 = w[i:i + H2].view(H2, 1); i += H2
    b3 = w[i:i + 1]
    h = torch.relu(X @ W1 + b1)
    h = torch.relu(h @ W2 + b2)
    return (h @ W3 + b3).squeeze(-1)


def bce(w, X, y):
    return torch.nn.functional.binary_cross_entropy_with_logits(logits(w, X), y)


def loss_grad(w, X, y):
    w = w.detach().requires_grad_(True)
    l = bce(w, X, y)
    g, = torch.autograd.grad(l, w)
    return l.item(), g


@torch.no_grad()
def predict_prob(w, X):
    return torch.sigmoid(logits(w, X)).numpy()


def _loss1(w, x, y):
    return torch.nn.functional.binary_cross_entropy_with_logits(logits(w, x.unsqueeze(0)), y.unsqueeze(0))


_psg = torch.func.vmap(torch.func.grad(_loss1), in_dims=(None, 0, 0))


def dp_grad(w, X, y, C, z, gen):
    """DP-SGD gradient: per-sample clipping to norm C, Gaussian noise N(0, (zC)^2), mean over the batch."""
    g = _psg(w.detach(), X, y)
    g = g * torch.clamp(C / (g.norm(dim=1, keepdim=True) + 1e-12), max=1.0)
    return (g.sum(0) + z * C * torch.randn(g.shape[1], generator=gen)) / len(y)


def sgd(w0, X, y, epochs, lr, bs, gen, mu=0.0, center=None, corr=None, wd=0.0, dp=None):
    """Local SGD. mu/center: proximal term; corr: additive gradient correction (SCAFFOLD);
    dp=(C, z): sample-level DP-SGD."""
    w = w0.clone(); n = len(y); losses = []; steps = 0
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            if dp is not None:
                g = dp_grad(w, X[b], y[b], dp[0], dp[1], gen)
                with torch.no_grad():
                    l = bce(w, X[b], y[b]).item()
            else:
                l, g = loss_grad(w, X[b], y[b])
            if mu > 0:
                g = g + mu * (w - center)
            if corr is not None:
                g = g + corr
            if wd > 0:
                g = g + wd * w
            w = w - lr * g
            losses.append(l); steps += 1
    return w, float(np.mean(losses)), steps


def pooled_auc(fed, preds, split):
    y = np.concatenate([c["y" + split].numpy() for c in fed])
    p = np.concatenate(preds)
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")


# ------------------------------------------------------------------ DP / compression helpers
def clip_to(v, C):
    nrm = v.norm().item()
    return v * min(1.0, C / (nrm + 1e-12))


def topk_sparsify(v, frac):
    k = max(1, int(frac * v.numel()))
    idx = torch.topk(v.abs(), k).indices
    out = torch.zeros_like(v); out[idx] = v[idx]
    return out, k


# ------------------------------------------------------------------ base class
class FL:
    name = "base"

    def __init__(self, fed, d, cfg, seed):
        set_model(cfg.get("model", "mlp")); set_init(cfg.get("init", "he"))
        self.fed, self.d, self.cfg, self.seed = fed, d, cfg, seed
        self.N = len(fed); self.P = n_params(d)
        self.n = np.array([len(c["ytr"]) for c in fed], dtype=float)
        self.hosp = np.array([c["hosp"] for c in fed])
        self.H = int(self.hosp.max()) + 1
        self.gen = torch.Generator().manual_seed(seed)
        self.rng = np.random.RandomState(seed)
        self.wan = 0.0; self.lan = 0.0
        self.E = cfg.get("epochs", 2); self.lr = cfg.get("lr", 0.03); self.bs = cfg.get("bs", 32)
        self.frac = cfg.get("frac", 1.0)
        self.dp_z = cfg.get("dp_z", 0.0); self.dp_C = cfg.get("dp_C", 1.0)       # client-level DP (unused)
        self.sdp_z = cfg.get("sdp_z", 0.0); self.sdp_C = cfg.get("sdp_C", 1.0)   # sample-level DP-SGD
        self.topk = cfg.get("topk", 0.0)
        self.resid = {}
        self.theta0 = init_vec(d, seed)

    # ---- helpers
    def sample(self):
        if self.frac >= 1.0:
            return list(range(self.N))
        k = max(1, int(round(self.frac * self.N)))
        return sorted(self.rng.choice(self.N, k, replace=False).tolist())

    def local(self, i, w0, shared=True, **kw):
        """shared=False: personal-model training (stays on the client, no DP needed)."""
        c = self.fed[i]
        if shared and self.sdp_z > 0:
            kw["dp"] = (self.sdp_C, self.sdp_z)
        return sgd(w0, c["Xtr"], c["ytr"], self.E, self.lr, self.bs, self.gen, **kw)

    def uplink(self, i, delta):
        """Optional top-k sparsification with error feedback; returns (delta_hat, bytes)."""
        if self.topk > 0:
            r = self.resid.get(i, torch.zeros_like(delta))
            sp, k = topk_sparsify(delta + r, self.topk)
            self.resid[i] = delta + r - sp
            return sp, k * 8.0
        return delta, self.P * 4.0

    def aggregate(self, deltas, weights, denom_n=None):
        """Weighted mean of deltas; with DP: uniform clipped sum + Gaussian noise."""
        if self.dp_z > 0:
            tot = sum(clip_to(dl, self.dp_C) for dl in deltas)
            tot = tot + torch.randn(self.P, generator=self.gen) * self.dp_z * self.dp_C
            return tot / max(denom_n or len(deltas), 1)
        wt = np.asarray(weights, dtype=float); wt = wt / wt.sum()
        return sum(float(a) * dl for a, dl in zip(wt, deltas))

    # ---- to override
    def round(self, t): raise NotImplementedError
    def predict(self, split): raise NotImplementedError

    def run(self, R, eval_every=5, ext_X=None):
        best_v, best = -1.0, None; curve = []
        for t in range(1, R + 1):
            self.round(t)
            if t % eval_every == 0 or t == R:
                pv = self.predict("va"); va = pooled_auc(self.fed, pv, "va")
                curve.append((t, va, self.wan, self.lan))
                if (not np.isnan(va)) and va > best_v:
                    best_v = va; best = dict(val=pv, test=self.predict("te"), round=t,
                                             wan=self.wan, lan=self.lan)
                    if ext_X is not None:                      # external (held-out) hospital: global model only
                        best["ext"] = predict_prob(self.g, ext_X)
        best["curve"] = curve; best["val_auc"] = best_v
        best["total_wan"] = self.wan; best["total_lan"] = self.lan
        return best


# ------------------------------------------------------------------ flat methods
class FedAvg(FL):
    name = "fedavg"

    def __init__(self, *a, mu=0.0, **k):
        super().__init__(*a, **k); self.g = self.theta0.clone(); self.mu = self.cfg.get("mu", 0.0)

    def round(self, t):
        S = self.sample(); ds, ws = [], []
        for i in S:
            w, _, _ = self.local(i, self.g, mu=self.mu, center=self.g)
            dl, nb = self.uplink(i, w - self.g)
            ds.append(dl); ws.append(self.n[i]); self.wan += self.P * 4.0 + nb
        self.g = self.g + self.aggregate(ds, ws, denom_n=max(1, round(self.frac * self.N)))

    def predict(self, split):
        return [predict_prob(self.g, c["X" + split]) for c in self.fed]


class FedProx(FedAvg):
    name = "fedprox"


class FedAvgFT(FedAvg):
    """FedAvg followed by local fine-tuning: the simplest personalisation baseline. Training is plain FedAvg; at prediction time each
    client fine-tunes the global model for ft_epochs on its own training data (never transmitted)."""
    name = "fedavg_ft"

    def predict(self, split):
        out = []; E = self.cfg.get("ft_epochs", 1)
        for i, c in enumerate(self.fed):
            gen = torch.Generator().manual_seed(7 + i)
            w, _, _ = sgd(self.g, c["Xtr"], c["ytr"], E, self.lr, self.bs, gen)
            out.append(predict_prob(w, c["X" + split]))
        return out


class Scaffold(FL):
    name = "scaffold"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.g = self.theta0.clone(); self.c = torch.zeros(self.P)
        self.ci = [torch.zeros(self.P) for _ in range(self.N)]

    def round(self, t):
        S = self.sample(); dws, dcs = [], []
        for i in S:
            w, _, K = self.local(i, self.g, corr=self.c - self.ci[i])
            ci_new = self.ci[i] - self.c + (self.g - w) / (K * self.lr)
            dws.append(w - self.g); dcs.append(ci_new - self.ci[i]); self.ci[i] = ci_new
            self.wan += 2 * 2 * self.P * 4.0
        self.g = self.g + sum(dws) / len(S)
        self.c = self.c + (len(S) / self.N) * sum(dcs) / len(S)

    def predict(self, split):
        return [predict_prob(self.g, c["X" + split]) for c in self.fed]


class Ditto(FL):
    name = "ditto"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.g = self.theta0.clone(); self.v = [self.theta0.clone() for _ in range(self.N)]
        self.lam = self.cfg.get("lam", 0.1)

    def round(self, t):
        S = self.sample(); ds, ws = [], []
        for i in S:
            w, _, _ = self.local(i, self.g)
            self.v[i], _, _ = self.local(i, self.v[i], shared=False, mu=self.lam, center=self.g)
            ds.append(w - self.g); ws.append(self.n[i]); self.wan += 2 * self.P * 4.0
        self.g = self.g + self.aggregate(ds, ws)

    def predict(self, split):
        return [predict_prob(self.v[i], c["X" + split]) for i, c in enumerate(self.fed)]


class PFedMe(FL):
    name = "pfedme"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.g = self.theta0.clone(); self.lam = self.cfg.get("lam", 5.0); self.K = 3

    def _inner(self, theta, w, X, y):
        for _ in range(self.K):
            _, gr = loss_grad(theta, X, y)
            theta = theta - self.lr * (gr + self.lam * (theta - w))
        return theta

    def round(self, t):
        S = self.sample(); ws_, ns = [], []
        for i in S:
            c = self.fed[i]; w = self.g.clone(); n = len(c["ytr"])
            for _ in range(self.E):
                perm = torch.randperm(n, generator=self.gen)
                for j in range(0, n, self.bs):
                    b = perm[j:j + self.bs]
                    theta = self._inner(w.clone(), w, c["Xtr"][b], c["ytr"][b])
                    w = w - self.lr * self.lam * (w - theta)
            ws_.append(w); ns.append(self.n[i]); self.wan += 2 * self.P * 4.0
        a = np.asarray(ns) / np.sum(ns)
        self.g = sum(float(x) * w for x, w in zip(a, ws_))

    def predict(self, split):
        out = []
        for c in self.fed:
            theta = self.g.clone(); n = len(c["ytr"])
            perm = torch.randperm(n, generator=torch.Generator().manual_seed(1))
            for j in range(0, n, self.bs):
                b = perm[j:j + self.bs]
                _, gr = loss_grad(theta, c["Xtr"][b], c["ytr"][b])
                theta = theta - self.lr * (gr + self.lam * (theta - self.g))
            out.append(predict_prob(theta, c["X" + split]))
        return out


class PerFedAvg(FL):
    name = "perfedavg"

    def __init__(self, *a, **k):
        super().__init__(*a, **k); self.g = self.theta0.clone()

    def round(self, t):
        S = self.sample(); ws_, ns = [], []
        for i in S:
            c = self.fed[i]; w = self.g.clone(); n = len(c["ytr"])
            for _ in range(self.E):
                perm = torch.randperm(n, generator=self.gen)
                for j in range(0, n, 2 * self.bs):
                    b1 = perm[j:j + self.bs]; b2 = perm[j + self.bs:j + 2 * self.bs]
                    if len(b2) == 0:
                        b2 = b1
                    _, g1 = loss_grad(w, c["Xtr"][b1], c["ytr"][b1])
                    _, g2 = loss_grad(w - self.lr * g1, c["Xtr"][b2], c["ytr"][b2])   # first-order MAML
                    w = w - self.lr * g2
            ws_.append(w); ns.append(self.n[i]); self.wan += 2 * self.P * 4.0
        a = np.asarray(ns) / np.sum(ns)
        self.g = sum(float(x) * w for x, w in zip(a, ws_))

    def predict(self, split):
        out = []
        for c in self.fed:
            m = min(len(c["ytr"]), 1024)
            _, gr = loss_grad(self.g, c["Xtr"][:m], c["ytr"][:m])
            out.append(predict_prob(self.g - self.lr * gr, c["X" + split]))
        return out


class IFCA(FL):
    name = "ifca"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.K = self.cfg.get("K", 3)
        gk = torch.Generator().manual_seed(self.seed + 77)
        self.models = [self.theta0 + 0.05 * torch.randn(self.P, generator=gk) * (j > 0) for j in range(self.K)]

    def _pick(self, i):
        c = self.fed[i]; m = min(len(c["ytr"]), 1024)
        with torch.no_grad():
            ls = [bce(w, c["Xtr"][:m], c["ytr"][:m]).item() for w in self.models]
        return int(np.argmin(ls))

    def round(self, t):
        S = self.sample(); upd = {j: ([], []) for j in range(self.K)}
        for i in S:
            j = self._pick(i)
            w, _, _ = self.local(i, self.models[j])
            upd[j][0].append(w); upd[j][1].append(self.n[i]); self.wan += (self.K + 1) * self.P * 4.0
        for j, (ws_, ns) in upd.items():
            if ws_:
                a = np.asarray(ns) / np.sum(ns)
                self.models[j] = sum(float(x) * w for x, w in zip(a, ws_))

    def predict(self, split):
        return [predict_prob(self.models[self._pick(i)], c["X" + split]) for i, c in enumerate(self.fed)]


# ------------------------------------------------------------------ hierarchical methods
class HierFAVG(FL):
    """Client-edge-cloud hierarchical FedAvg (Liu et al. 2020): hospital aggregation every round,
    global aggregation every tau rounds. Global model is the prediction model."""
    name = "hierfavg"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.tau = self.cfg.get("tau", 5)
        self.phi = [self.theta0.clone() for _ in range(self.H)]
        self.g = self.theta0.clone()

    def hospital_round(self, S):
        """Local training + intra-hospital aggregation. Returns dict h -> (new_phi, risk, N_h, extra)."""
        res = {}
        for h in range(self.H):
            Sh = [i for i in S if self.hosp[i] == h]
            if not Sh:
                continue
            ds, ws, ls = [], [], []
            for i in Sh:
                w, l, _ = self.local(i, self.phi[h])
                dl, nb = self.uplink(i, w - self.phi[h])
                ds.append(dl); ws.append(self.n[i]); ls.append(l); self.lan += self.P * 4.0 + nb
                self.lan += self.P * 4.0
            denom = max(1, round(self.frac * (self.hosp == h).sum()))
            new = self.phi[h] + self.aggregate(ds, ws, denom_n=denom)
            res[h] = (new, float(np.average(ls, weights=ws)), float(np.sum(ws)))
        return res

    def global_weights(self, res):
        return {h: r[2] for h, r in res.items()}

    def round(self, t):
        S = self.sample(); res = self.hospital_round(S)
        for h, r in res.items():
            self.phi[h] = r[0]
        if t % self.tau == 0:
            self.sync(res)

    def sync(self, res):
        gw = self.global_weights(res); tot = sum(gw.values())
        self.g = sum((gw[h] / tot) * self.phi[h] for h in res)
        self.pre = [p.clone() for p in self.phi]
        for h in range(self.H):
            self.phi[h] = self.g.clone()
        self.wan += 2 * len(res) * self.P * 4.0

    def predict(self, split):
        return [predict_prob(self.g, c["X" + split]) for c in self.fed]


class HiPerC(HierFAVG):
    """HiPerC-FL: hierarchical aggregation + client personalisation (proximal to the hospital model)
    + federated V-REx invariance (causal) regulariser over hospital environments.

    lam_c : V-REx coefficient (0 disables)       lam_p : proximal strength of personal models
    tau   : hospital->global sync period (1 = flat)   mode : 'personal' | 'hospital' | 'global'
    """
    name = "hiperc"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.lam_c = self.cfg.get("lam_c", 1.0); self.lam_p = self.cfg.get("lam_p", 0.1)
        self.mode = self.cfg.get("mode", "personal")
        self.v = [self.theta0.clone() for _ in range(self.N)]
        self.pre = [self.theta0.clone() for _ in range(self.H)]
        self.last_w = None

    def global_weights(self, res):
        p = {h: r[2] for h, r in res.items()}; tot = sum(p.values())
        p = {h: x / tot for h, x in p.items()}
        if self.lam_c <= 0 or len(res) < 2:
            self.last_w = p; return p
        rbar = sum(p[h] * res[h][1] for h in res)
        om = {h: max(0.0, 1.0 + 2.0 * self.lam_c * (res[h][1] - rbar)) for h in res}
        gw = {h: p[h] * om[h] for h in res}
        if sum(gw.values()) <= 0:
            gw = p
        self.last_w = {h: x / sum(gw.values()) for h, x in gw.items()}
        return gw

    def round(self, t):
        S = self.sample()
        old = [p.clone() for p in self.phi]                # hospital models at start of round
        res = self.hospital_round(S)
        if self.mode == "personal":
            for i in S:
                self.v[i], _, _ = self.local(i, self.v[i], shared=False, mu=self.lam_p, center=old[self.hosp[i]])
        for h, r in res.items():
            self.phi[h] = r[0]
        if t % self.tau == 0:
            self.sync(res)

    def predict(self, split):
        out = []
        for i, c in enumerate(self.fed):
            if self.mode == "personal":
                w = self.v[i]
            elif self.mode == "hospital":
                w = self.pre[self.hosp[i]]
            else:
                w = self.g
            out.append(predict_prob(w, c["X" + split]))
        return out


# ------------------------------------------------------------------ non-federated references
def run_centralized(fed, d, cfg, seed, epochs=60, ext_X=None):
    """Pooled MLP (upper reference). Model chosen by validation AUROC each epoch."""
    set_model(cfg.get("model", "mlp")); set_init(cfg.get("init", "he"))
    X = torch.cat([c["Xtr"] for c in fed]); y = torch.cat([c["ytr"] for c in fed])
    gen = torch.Generator().manual_seed(seed)
    w = init_vec(d, seed); best_v, best = -1, None
    for ep in range(1, epochs + 1):
        w, _, _ = sgd(w, X, y, 1, cfg.get("lr", 0.03), cfg.get("bs", 32), gen)
        if ep % 2 == 0 or ep == epochs:
            pv = [predict_prob(w, c["Xva"]) for c in fed]; va = pooled_auc(fed, pv, "va")
            if va > best_v:
                best_v = va
                best = dict(val=pv, test=[predict_prob(w, c["Xte"]) for c in fed], round=ep)
                if ext_X is not None:
                    best["ext"] = predict_prob(w, ext_X)
    best["val_auc"] = best_v; best["curve"] = []; best["total_wan"] = 0; best["total_lan"] = 0
    return best


def run_local(fed, d, cfg, seed, epochs=60):
    """Each hospital trains only on its own data (no collaboration)."""
    hosp = np.array([c["hosp"] for c in fed]); gen = torch.Generator().manual_seed(seed)
    val = [None] * len(fed); test = [None] * len(fed)
    for h in sorted(set(hosp)):
        ids = [i for i in range(len(fed)) if hosp[i] == h]
        X = torch.cat([fed[i]["Xtr"] for i in ids]); y = torch.cat([fed[i]["ytr"] for i in ids])
        w = init_vec(d, seed); best_v, bw = 1e9, w
        Xv = torch.cat([fed[i]["Xva"] for i in ids]); yv = torch.cat([fed[i]["yva"] for i in ids])
        for ep in range(1, epochs + 1):
            w, _, _ = sgd(w, X, y, 1, cfg.get("lr", 0.03), cfg.get("bs", 32), gen)
            with torch.no_grad():
                lv = bce(w, Xv, yv).item()
            if lv < best_v:
                best_v, bw = lv, w.clone()
        for i in ids:
            val[i] = predict_prob(bw, fed[i]["Xva"]); test[i] = predict_prob(bw, fed[i]["Xte"])
    return dict(val=val, test=test, round=epochs, val_auc=pooled_auc(fed, val, "va"), curve=[],
                total_wan=0, total_lan=0)


def run_logreg(fed, X_cols=None, C_grid=(0.01, 0.1, 1.0, 10.0), ext_X=None):
    """Pooled L2 logistic regression - the conventional clinical prediction model (upper reference)."""
    from sklearn.linear_model import LogisticRegression
    X = torch.cat([c["Xtr"] for c in fed]).numpy(); y = torch.cat([c["ytr"] for c in fed]).numpy()
    if X_cols is not None:
        X = X[:, X_cols]
    Xv = torch.cat([c["Xva"] for c in fed]).numpy(); yv = torch.cat([c["yva"] for c in fed]).numpy()
    if X_cols is not None:
        Xv = Xv[:, X_cols]
    best_v, bm = -1, None
    for C in C_grid:
        m = LogisticRegression(C=C, max_iter=2000).fit(X, y)
        a = roc_auc_score(yv, m.predict_proba(Xv)[:, 1])
        if a > best_v:
            best_v, bm = a, m
    sel = (lambda a: a[:, X_cols]) if X_cols is not None else (lambda a: a)
    val = [bm.predict_proba(sel(c["Xva"].numpy()))[:, 1] for c in fed]
    test = [bm.predict_proba(sel(c["Xte"].numpy()))[:, 1] for c in fed]
    out = dict(val=val, test=test, round=0, val_auc=best_v, curve=[], total_wan=0, total_lan=0)
    if ext_X is not None:
        out["ext"] = bm.predict_proba(sel(ext_X.numpy()))[:, 1]
    return out


METHODS = dict(fedavg=FedAvg, fedavg_ft=FedAvgFT, fedprox=FedProx, scaffold=Scaffold, ditto=Ditto, pfedme=PFedMe,
               perfedavg=PerFedAvg, ifca=IFCA, hierfavg=HierFAVG, hiperc=HiPerC)


def run_method(name, fed, d, cfg, seed, rounds, eval_every=5, ext_X=None):
    if name == "centralized":
        return run_centralized(fed, d, cfg, seed, ext_X=ext_X)
    if name == "local":
        return run_local(fed, d, cfg, seed)
    if name == "logreg":
        return run_logreg(fed, ext_X=ext_X)
    return METHODS[name](fed, d, cfg, seed).run(rounds, eval_every, ext_X=ext_X)
