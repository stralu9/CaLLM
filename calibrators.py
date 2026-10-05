"""Baseline calibrators.

  post-hoc     temperature, Platt, isotonic, histogram binning, beta
  CalibraEval  label-free order-preserving map (Li et al., 2025)
  grouped MC   GCUR linear / logistic, IGLB (fit on clustering groups,
               Detommaso et al., 2024: UMAP(20) + GMM chosen by BIC)
  LenControl   length-controlled logistic model (Dubois et al., 2024)

Every post-hoc calibrator is ``Cal(**kw).fit(p, y).predict(p_new)``; the
grouped ones also take ``{name: bool mask}`` dicts of group memberships.
"""
from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, LogisticRegression

from data import EPS

_EPS_POSTHOC = 1e-6


def _clip(p, eps=_EPS_POSTHOC):
    return np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)


# --------------------------------------------------------------------------- #
# Post-hoc (no group information)
# --------------------------------------------------------------------------- #

class TemperatureScaling:
    """``sigmoid(logit(p) / T)``, fit as a logistic regression through the
    origin."""

    def __init__(self):
        self.coef_ = 1.0

    def fit(self, p, y):
        y = np.asarray(y).astype(int)
        if len(np.unique(y)) > 1:
            lr = LogisticRegression(fit_intercept=False, C=1e10, solver="lbfgs",
                                    max_iter=1000)
            lr.fit(logit(_clip(p)).reshape(-1, 1), y)
            self.coef_ = float(lr.coef_[0, 0])
        return self

    def predict(self, p):
        return expit(self.coef_ * logit(_clip(p)))


class PlattScaling:
    """``sigmoid(a * logit(p) + b)``."""

    def __init__(self):
        self.a_, self.b_ = 1.0, 0.0

    def fit(self, p, y):
        y = np.asarray(y).astype(int)
        if len(np.unique(y)) > 1:
            lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
            lr.fit(logit(_clip(p)).reshape(-1, 1), y)
            self.a_, self.b_ = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        return self

    def predict(self, p):
        return expit(self.a_ * logit(_clip(p)) + self.b_)


class IsotonicCal:
    def fit(self, p, y):
        self.ir_ = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self.ir_.fit(np.asarray(p, dtype=np.float64), np.asarray(y, dtype=np.float64))
        return self

    def predict(self, p):
        return self.ir_.transform(np.asarray(p, dtype=np.float64))


class HistogramBinning:
    """Equal-width bins, each mapped to its calibration positive rate."""

    def __init__(self, n_bins: int = 15):
        self.n_bins = n_bins
        self.edges_ = np.linspace(0.0, 1.0, n_bins + 1)
        self.bin_val_ = np.full(n_bins, 0.5)

    def _bin(self, p):
        return np.clip(np.digitize(np.asarray(p, dtype=np.float64),
                                   self.edges_[1:-1]), 0, self.n_bins - 1)

    def fit(self, p, y):
        idx, y = self._bin(p), np.asarray(y, dtype=np.float64)
        for b in range(self.n_bins):
            if (idx == b).any():
                self.bin_val_[b] = float(y[idx == b].mean())
        return self

    def predict(self, p):
        return self.bin_val_[self._bin(p)]


class BetaCalibration:
    """Logistic regression on ``[log p, -log(1 - p)]`` (Kull et al., 2017)."""

    def __init__(self):
        self.lr_ = None

    def fit(self, p, y):
        p, y = _clip(p), np.asarray(y).astype(int)
        if len(np.unique(y)) > 1:
            self.lr_ = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
            self.lr_.fit(np.column_stack([np.log(p), -np.log(1.0 - p)]), y)
        return self

    def predict(self, p):
        p = _clip(p)
        if self.lr_ is None:
            return p
        return self.lr_.predict_proba(
            np.column_stack([np.log(p), -np.log(1.0 - p)]))[:, 1]


POSTHOC = {"temperature": TemperatureScaling, "platt": PlattScaling,
           "isotonic": IsotonicCal, "histogram": HistogramBinning,
           "beta": BetaCalibration}


# --------------------------------------------------------------------------- #
# CalibraEval
# --------------------------------------------------------------------------- #

@dataclass
class CalibraEvalConfig:
    lam: float = 0.1       # weight of the (negative) anti-collapse term, in [0, 1)
    lr: float = 0.05       # Adam step size
    max_iter: int = 500
    tol: float = 1e-6      # stop when max |delta g| on the grid falls below

    def __post_init__(self):
        if not 0.0 <= self.lam < 1.0:
            raise ValueError(f"lam must satisfy 0 <= lam < 1, got {self.lam}")


class CalibraEval:
    """CalibraEval's order-preserving map g (Li et al., 2025), L1 objective.

    ``fit(s0, s2)`` takes P(judge emits "A") in the arrangements
    X0 = [(A, o1); (B, o2)] and X2 = [(A, o2); (B, o1)] (ID-token swap).
    g = cumsum(softmax(d)) on the grid of all observed scores minimises

        mean (g0 + g2 - 1)^2 - lam * mean (g0 - g2)^2

    by full-batch Adam from the identity map. 
    A fit that does no better than the trivial constant map is
    replaced by the identity. Unseen scores go through an isotonic
    interpolation of the grid."""

    def __init__(self, config: Optional[CalibraEvalConfig] = None):
        self.cfg = config or CalibraEvalConfig()

    def fit(self, s0, s2) -> "CalibraEval":
        cfg = self.cfg
        arrays = [np.clip(np.asarray(s, dtype=np.float64).reshape(-1), 0.0, 1.0)
                  for s in (s0, s2)]
        K = arrays[0].size
        pool = np.concatenate(arrays)
        order = np.argsort(pool, kind="stable")
        ranks = np.empty_like(order)
        ranks[order] = np.arange(pool.size)
        idx = ranks + 1                                  # grid offset for z_0
        z = np.concatenate(([0.0], pool[order], [1.0]))
        i0, i2 = idx[:K], idx[-K:]
        d = np.log(np.maximum(np.diff(z, prepend=0.0), 1e-9))  # identity init
        d -= d.mean()
        m, v = np.zeros_like(d), np.zeros_like(d)
        b1, b2, eps = 0.9, 0.999, 1e-8
        g_prev = None
        for t in range(1, cfg.max_iter + 1):
            p = np.exp(_log_softmax(d))
            g = np.cumsum(p)
            if g_prev is not None and np.max(np.abs(g - g_prev)) < cfg.tol:
                break
            g_prev = g
            g0, g2 = g[i0], g[i2]
            u, spread = g0 + g2 - 1.0, g0 - g2
            c0, c2 = 2.0 * u, 2.0 * u
            c0 = c0 - 2.0 * cfg.lam * spread
            c2 = c2 + 2.0 * cfg.lam * spread
            grad = _grad(d, [(i0, c0 / K), (i2, c2 / K)], p, g)
            m = b1 * m + (1 - b1) * grad
            v = b2 * v + (1 - b2) * grad * grad
            d = d - cfg.lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
            d -= d.mean()
            if not np.all(np.isfinite(d)):
                raise ValueError(f"CalibraEval diverged at step {t} ({cfg})")
        g = np.cumsum(np.exp(_log_softmax(d)))
        g0, g2 = g[i0], g[i2]
        loss = np.mean((g0 + g2 - 1.0) ** 2) - cfg.lam * np.mean((g0 - g2) ** 2)
        if not loss < -1e-4:       # no better than the constant-0.5 map
            g = z.copy()
        self._iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self._iso.fit(z, g)
        return self

    def calibrate(self, p) -> np.ndarray:
        p = np.atleast_1d(np.asarray(p, dtype=np.float64))
        return self._iso.transform(np.clip(p, 0.0, 1.0))


def _log_softmax(d):
    z = d - d.max()
    return z - np.log(np.exp(z).sum())


def _grad(d, queries, p, cum_p):
    """dL/dd for (grid index, dL/dg) pairs, via the cumulative-softmax
    derivative dg_j/dd_k = p_k (1[k <= j] - g_j), in O(n + M)."""
    idx = np.concatenate([np.asarray(i) for i, _ in queries])
    coeff = np.concatenate([np.asarray(c, dtype=np.float64) for _, c in queries])
    w = np.bincount(idx, weights=coeff, minlength=d.size)
    tail_sum = np.cumsum(w[::-1])[::-1]
    return p * (tail_sum - float(np.dot(coeff, cum_p[idx])))


# --------------------------------------------------------------------------- #
# Grouped multicalibration: GCUR (linear / logistic) and IGLB
# --------------------------------------------------------------------------- #

def _group_features(p, group_masks, keys):
    cols = [p.reshape(-1, 1)]
    for k in keys:
        cols.append(group_masks[k].astype(float).reshape(-1, 1) if k in group_masks
                    else np.zeros((len(p), 1)))
    return np.hstack(cols)


class GCURLinear:
    """Least squares ``y ~ p + sum_g lambda_g 1[x in g]``, clipped to [0, 1]."""

    def fit(self, p, y, group_masks):
        self.keys = sorted(group_masks)
        self.model = LinearRegression().fit(
            _group_features(p, group_masks, self.keys), y)
        return self

    def predict(self, p, group_masks):
        return np.clip(self.model.predict(
            _group_features(p, group_masks, self.keys)), 0.0, 1.0)


class GCURLogistic:
    """Logistic regression on ``[p, group indicators]``."""

    def fit(self, p, y, group_masks):
        self.keys = sorted(group_masks)
        self.model = LogisticRegression(solver="lbfgs", max_iter=1000, C=1e10)
        self.model.fit(_group_features(p, group_masks, self.keys), y.astype(int))
        return self

    def predict(self, p, group_masks):
        return self.model.predict_proba(
            _group_features(p, group_masks, self.keys))[:, 1]


def _round_to_grid(p, M):
    grid = np.arange(1, M + 1) / M
    return grid[np.argmin(np.abs(p[:, None] - grid[None, :]), axis=1)]


class IGLB:
    """Iterative Grouped Linear Binning: repeatedly patch the (group,
    overlapping bin) cell with the largest mass-weighted squared residual by a
    linear scaling sigmoid(alpha + beta logit p), until that cell's mass drops
    below ``epsilon``."""

    def __init__(self, M: int = 20, max_iter: int = 500, epsilon: float = 0.01):
        self.M, self.max_iter, self.epsilon = M, max_iter, epsilon

    def _bins(self, p):
        out = {}
        for m in range(1, self.M + 1):
            out[(m, "<=")] = p <= m / self.M
            out[(m, ">=")] = p >= (m - 1) / self.M
        return out

    @staticmethod
    def _ls(p, alpha, beta):
        return expit(alpha + beta * logit(np.clip(p, 1e-7, 1 - 1e-7)))

    def fit(self, p, y, group_masks):
        self.patches = []
        p = _round_to_grid(p.copy(), self.M)
        n = len(y)
        for _ in range(self.max_iter):
            best = (-1, None, None, 0)             # score, group, bin, mass
            bins = self._bins(p)
            for g, gm in group_masks.items():
                for b, bm in bins.items():
                    cell = gm & bm
                    n_cell = cell.sum()
                    if n_cell < 5:
                        continue
                    score = (n_cell / n) * (y[cell] - p[cell]).mean() ** 2
                    if score > best[0]:
                        best = (score, g, b, n_cell / n)
            score, g, b, mass = best
            if g is None or score < 1e-10 or mass < self.epsilon:
                break
            cell = group_masks[g] & bins[b]
            alpha, beta = 0.0, 1.0
            if len(np.unique(y[cell])) > 1:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        lr = LogisticRegression(solver="lbfgs", max_iter=500, C=1e10)
                        lr.fit(logit(np.clip(p[cell], 1e-7, 1 - 1e-7)).reshape(-1, 1),
                               y[cell].astype(int))
                        alpha, beta = float(lr.intercept_[0]), float(lr.coef_[0, 0])
                    except Exception:
                        alpha, beta = 0.0, 1.0
            h = p.copy()
            h[cell] = self._ls(p[cell], alpha, beta)
            p = _round_to_grid(np.clip(h, 0, 1), self.M)
            self.patches.append((g, b, alpha, beta))
        return self

    def predict(self, p, group_masks):
        p = _round_to_grid(p.copy(), self.M)
        for g, b, alpha, beta in self.patches:
            if g not in group_masks:
                continue
            cell = group_masks[g] & self._bins(p)[b]
            if cell.sum() > 0:
                p[cell] = self._ls(p[cell], alpha, beta)
                p = _round_to_grid(np.clip(p, 0, 1), self.M)
        return np.clip(p, 0.0, 1.0)


GROUPED = {"gcur_linear": GCURLinear, "gcur_logistic": GCURLogistic,
           "iglb": IGLB}


class ClusterGroups:
    """Data-driven groups for the grouped-MC baselines (Detommaso et al.,
    2024): UMAP(20) of the joint embedding of the shown presentation, then a
    Gaussian mixture on [UMAP coordinates, score] with K in 2..20 chosen by
    BIC; one group per component with at least ``min_n`` calibration rows.
    Fit on the calibration rows of each split (at most ``max_fit`` of them)
    and memoised per calibration set. The reversed presentation is assigned
    with its own embedding and score.

    ``groups(cal, te) -> (masks_cal, masks_te, masks_te_reversed)``."""

    def __init__(self, shown_emb, reversed_emb, s0, s1, umap_dim: int = 20,
                 k_grid: Sequence[int] = tuple(range(2, 21)),
                 max_fit: int = 20000, min_n: int = 30, seed: int = 0):
        self.shown_emb = np.asarray(shown_emb, dtype=np.float32)
        self.reversed_emb = np.asarray(reversed_emb, dtype=np.float32)
        self.s0, self.s1 = np.asarray(s0, dtype=float), np.asarray(s1, dtype=float)
        self.umap_dim, self.k_grid = umap_dim, tuple(k_grid)
        self.max_fit, self.min_n, self.seed = max_fit, min_n, seed
        self._fits: Dict[str, tuple] = {}

    def _fit(self, cal):
        key = hashlib.md5(np.ascontiguousarray(cal).tobytes()).hexdigest()
        if key not in self._fits:
            import umap
            from sklearn.mixture import GaussianMixture
            rng = np.random.default_rng(self.seed)
            rows = (np.sort(rng.choice(cal, self.max_fit, replace=False))
                    if len(cal) > self.max_fit else cal)
            reducer = umap.UMAP(n_components=self.umap_dim, random_state=self.seed)
            x = np.column_stack([reducer.fit_transform(self.shown_emb[rows]),
                                 self.s0[rows]])
            best, best_bic = None, np.inf
            for k in self.k_grid:
                gmm = GaussianMixture(n_components=k, covariance_type="full",
                                      random_state=self.seed, reg_covar=1e-5).fit(x)
                bic = gmm.bic(x)
                if bic < best_bic:
                    best, best_bic = gmm, bic
            lab_cal = self._assign(reducer, best, self.shown_emb[cal], self.s0[cal])
            keep = [k for k in range(best.n_components)
                    if int((lab_cal == k).sum()) >= self.min_n]
            print(f"    [cluster groups] BIC picked K={best.n_components}, "
                  f"{len(keep)} groups with >= {self.min_n} calibration rows",
                  flush=True)
            self._fits[key] = (reducer, best, keep, lab_cal)
        return self._fits[key]

    @staticmethod
    def _assign(reducer, gmm, emb, score):
        return gmm.predict(np.column_stack([reducer.transform(emb), score]))

    def __call__(self, cal, te):
        cal, te = np.asarray(cal), np.asarray(te)
        reducer, gmm, keep, lab_cal = self._fit(cal)
        lab_te = self._assign(reducer, gmm, self.shown_emb[te], self.s0[te])
        lab_rev = self._assign(reducer, gmm, self.reversed_emb[te], self.s1[te])
        masks = lambda lab: {f"cluster:{k}": lab == k for k in keep}
        return masks(lab_cal), masks(lab_te), masks(lab_rev)


# --------------------------------------------------------------------------- #
# Length control
# --------------------------------------------------------------------------- #

def length_control(dlen, s0, s1, y, model_a, model_b, cal, te,
                   C: float = 1.0, min_freq: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Length-controlled judge (Dubois et al., 2024). On the calibration rows
    fit

        P(y = 1) = sigmoid(logit(f0) + b0 + theta[model_A] - theta[model_B]
                           + b2 * z(len_A - len_B))

    with logit(f0) a fixed offset and an L2 penalty 1/(2C) on theta and b2
    (models seen fewer than ``min_freq`` times share one level), then predict
    with the length term at 0. Returns the predictions from ``s0`` and from
    ``s1`` (the reversed pass, with its own slots)."""
    from scipy.optimize import minimize
    from sklearn.preprocessing import OneHotEncoder

    def _logit(p):
        p = np.clip(p, EPS, 1 - EPS)
        return np.log(p / (1.0 - p))

    yc = y[cal].astype(int)
    if len(np.unique(yc)) < 2:
        return np.clip(s0[te], EPS, 1 - EPS), np.clip(s1[te], EPS, 1 - EPS)
    sd = float(dlen[cal].std())
    zc = ((dlen - float(dlen[cal].mean())) / (sd if sd > 0 else 1.0))[cal]
    offset_c = _logit(s0[cal])
    use_model = len(set(model_a[cal]) | set(model_b[cal])) > 2
    if use_model:
        enc = OneHotEncoder(handle_unknown="ignore", min_frequency=min_freq,
                            dtype=np.float64)
        enc.fit(np.concatenate([model_a[cal], model_b[cal]]).reshape(-1, 1))
        diff = lambda rows: (enc.transform(model_a[rows].reshape(-1, 1))
                             - enc.transform(model_b[rows].reshape(-1, 1))).toarray()
        Xm_c = diff(cal)
    else:
        Xm_c = np.zeros((len(zc), 0))
    k = Xm_c.shape[1]

    def _nll(w):                     # w = [b0, theta_1..theta_k, b2]
        theta, b2 = w[1:1 + k], w[-1]
        p = np.clip(1.0 / (1.0 + np.exp(-(offset_c + w[0] + Xm_c @ theta + b2 * zc))),
                    EPS, 1 - EPS)
        return (-np.mean(yc * np.log(p) + (1 - yc) * np.log(1 - p))
                + (np.sum(theta ** 2) + b2 ** 2) / (2.0 * C))

    w = minimize(_nll, x0=np.zeros(k + 2), method="L-BFGS-B").x
    model_term = diff(te) @ w[1:1 + k] if use_model else 0.0

    def _pred(base):
        return np.clip(1.0 / (1.0 + np.exp(-(_logit(base[te]) + w[0] + model_term))),
                       EPS, 1 - EPS)

    return _pred(s0), _pred(s1)
