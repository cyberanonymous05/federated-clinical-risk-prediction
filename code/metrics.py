"""Evaluation metrics: discrimination, calibration, thresholded performance, decision curves, subgroup."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from scipy import stats


def auroc(y, p):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, p))


def auprc(y, p):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(average_precision_score(y, p))


def youden_threshold(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def ece(y, p, bins=10):
    y = np.asarray(y); p = np.asarray(p)
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -1e-9, 1 + 1e-9
    tot = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p > a) & (p <= b)
        if m.sum():
            tot += m.sum() * abs(y[m].mean() - p[m].mean())
    return float(tot / len(y))


def calib_slope_intercept(y, p):
    """Logistic recalibration: logit(P(y=1)) = a + b*logit(p). Ideal a=0, b=1."""
    import statsmodels.api as sm
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    try:
        m = sm.GLM(y, sm.add_constant(z), family=sm.families.Binomial()).fit()
        return float(m.params[1]), float(m.params[0])
    except Exception:
        return np.nan, np.nan


def full_metrics(y, p, thr):
    y = np.asarray(y).astype(int); p = np.asarray(p)
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * sens / max(prec + sens, 1e-12)
    slope, icpt = calib_slope_intercept(y, p)
    return dict(auroc=auroc(y, p), auprc=auprc(y, p), acc=(tp + tn) / len(y), sensitivity=sens, specificity=spec,
                prec=prec, f1=f1, brier=float(np.mean((p - y) ** 2)), ece=ece(y, p),
                cal_slope=slope, cal_int=icpt, thr=thr, prev=float(y.mean()), n=len(y))


def net_benefit(y, p, thresholds):
    y = np.asarray(y); n = len(y); out = []
    for t in thresholds:
        pred = p >= t
        tp = ((pred) & (y == 1)).sum(); fp = ((pred) & (y == 0)).sum()
        out.append(tp / n - fp / n * t / (1 - t))
    return np.array(out)


def treat_all_nb(y, thresholds):
    prev = np.mean(y)
    return np.array([prev - (1 - prev) * t / (1 - t) for t in thresholds])


def t_ci(vals, level=0.95):
    v = np.asarray([x for x in vals if not np.isnan(x)], dtype=float)
    if len(v) < 2:
        return (np.nan, np.nan)
    m, s = v.mean(), v.std(ddof=1)
    h = stats.t.ppf(0.5 + level / 2, len(v) - 1) * s / np.sqrt(len(v))
    return (m - h, m + h)


def holm(pvals):
    p = np.asarray(pvals, dtype=float); m = len(p)
    order = np.argsort(p); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i]); adj[i] = min(1.0, run)
    return adj


def delong_pvalue(y, p1, p2):
    """Two-sided DeLong test for two correlated AUCs on the same cases."""
    y = np.asarray(y).astype(int)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    def comps(p):
        P = p[pos][:, None]; N = p[neg][None, :]
        psi = (P > N) + 0.5 * (P == N)
        return psi.mean(), psi.mean(1), psi.mean(0)
    a1, v10_1, v01_1 = comps(np.asarray(p1)); a2, v10_2, v01_2 = comps(np.asarray(p2))
    m, n = len(pos), len(neg)
    s10 = np.cov(np.vstack([v10_1, v10_2])); s01 = np.cov(np.vstack([v01_1, v01_2]))
    S = s10 / m + s01 / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var <= 0:
        return np.nan
    z = (a1 - a2) / np.sqrt(var)
    return float(2 * (1 - stats.norm.cdf(abs(z))))
