"""Scores every method on the question-grouped out-of-fold predictions.

    python evaluate.py bias --dataset mtbench --judge muse --embedder muse-internal
        per-group MCE% / MCEsigma on position, verbosity and family, the joint
        MCE over all three, and the MCE on unspecified embedding segments
        -> results/Q1_mce.csv
    python evaluate.py perf --dataset mtbench --judge muse --embedder muse-internal
        accuracy / Brier / log loss / ECCE of the order-symmetrised scores
        -> results/Q2_perf.csv
    python evaluate.py family_lowfloor --datasets rewardbench,mtbench \
        --judge muse --embedder muse-internal --floor 20
        the family group alone at a lower per-fold floor, for the datasets
        whose own-family sides are too thin for the default one
        -> results/Q1_family_lowfloor.csv
    python evaluate.py summary --embedder muse-internal
        means across datasets -> results/summary_{bias,perf}[__embedder].csv
    python evaluate.py decodability --judge muse --embedder muse-internal
        can each group be decoded from CaLLM's features? (App. E)
        -> results/diagnostics/group_decodability.csv

Each metric is computed inside every held-out fold and averaged; ``*_se`` /
``*_lo`` / ``*_hi`` give the SE and 95% t-interval over the folds. A group is
scored in a fold only when both of its directional sides have at least
``--min_axis_n`` (default 30) pairs there; the neutral third level (same length,
both or neither from the judge family) joins when it clears the same floor.
On RewardBench and MTBench the own-family sides never both reach 30 in a fold,
so their family cells in ``Q1_mce.csv`` are empty; ``family_lowfloor``
re-scores that one group at floor 20 into a side-car CSV, which fills exactly
those cells. The joint MCE and every other cell keep the default floor.
"""
from __future__ import annotations

import argparse
import fcntl
import os
import warnings
from collections import Counter

import numpy as np
import pandas as pd

import callm as C
import data
import embed

warnings.filterwarnings("ignore", category=FutureWarning)
Q1_CSV = os.path.join(C.RESULTS_DIR, "Q1_mce.csv")
Q2_CSV = os.path.join(C.RESULTS_DIR, "Q2_perf.csv")
DECODABILITY_CSV = os.path.join(C.RESULTS_DIR, "diagnostics",
                                "group_decodability.csv")


# --------------------------------------------------------------------------- #
# Metric primitives
# --------------------------------------------------------------------------- #

def ecce_mce(f, y, seg_df=None, categorical_cols=None, max_values: int = 64) -> dict:
    """ECCE (whole sample) and MCE (worst segment of ``categorical_cols``)
    of ``mcgrad.metrics.MulticalibrationError``, relative (``*_perc``) and in
    SDs under the calibrated null (``*_sigma``); NaN when undefined."""
    from mcgrad import metrics
    nan = {k: float("nan") for k in ("ecce_perc", "ecce_sigma", "mce_perc",
                                     "mce_sigma")}
    f = np.clip(np.asarray(f, dtype=np.float64), 1e-7, 1 - 1e-7)
    y = np.asarray(y, dtype=np.float64)
    if len(np.unique(y)) < 2:
        return nan
    df = (seg_df.copy().reset_index(drop=True) if seg_df is not None
          else pd.DataFrame(index=range(len(f))))
    df["__mcg_y"], df["__mcg_p"] = y, f
    try:
        m = metrics.MulticalibrationError(
            df=df, label_column="__mcg_y", score_column="__mcg_p",
            categorical_segment_columns=list(categorical_cols or []),
            numerical_segment_columns=[],
            max_values_per_segment_feature=max_values)
        return dict(ecce_perc=float(m.global_ecce_relative),
                    ecce_sigma=float(m.global_ecce_sigma),
                    mce_perc=float(m.mce_relative), mce_sigma=float(m.mce_sigma))
    except Exception:
        return nan


def segment_ecce(f, y, lab, max_values: int):
    """One ``MulticalibrationError`` over the label column: the global ECCE and
    the ECCE of every cell, on one normalisation (so MCE = max of them)."""
    from mcgrad import metrics
    ff = np.clip(np.asarray(f, dtype=np.float64), 1e-7, 1 - 1e-7)
    yy = np.asarray(y, dtype=np.float64)
    if len(np.unique(yy)) < 2:
        return None
    try:
        m = metrics.MulticalibrationError(
            df=pd.DataFrame({"category": lab, "__mcg_y": yy, "__mcg_p": ff}),
            label_column="__mcg_y", score_column="__mcg_p",
            categorical_segment_columns=["category"],
            max_values_per_segment_feature=max_values)
        rel = np.asarray(m.segments_ecce_relative, dtype=float)
        sig = np.asarray(m.segments_ecce_sigma, dtype=float)
        cells = {str(r.value): (float(rel[int(r.idx_segment)]),
                                float(sig[int(r.idx_segment)]))
                 for r in m._segments[1].itertuples(index=False)
                 if int(r.idx_segment) < len(rel)}
        return dict(cells=cells, global_perc=float(m.global_ecce_relative),
                    global_sigma=float(m.global_ecce_sigma))
    except Exception:
        return None


def bias_gap(f, y, v, vp) -> float:
    """E[f - y | v] - E[f - y | vp]."""
    if v.sum() == 0 or vp.sum() == 0:
        return float("nan")
    r = np.asarray(f, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    return float(r[v].mean() - r[vp].mean())


def fold_ci(vals, conf: float = 0.95) -> dict:
    """Mean, SE and two-sided t-interval over the finite per-fold values."""
    from scipy import stats
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    nan = float("nan")
    if v.size == 0:
        return dict(mean=nan, se=nan, lo=nan, hi=nan, n=0)
    if v.size < 2:
        return dict(mean=float(v.mean()), se=nan, lo=nan, hi=nan, n=1)
    mean, se = float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size))
    half = float(stats.t.ppf(0.5 + conf / 2.0, v.size - 1)) * se
    return dict(mean=mean, se=se, lo=mean - half, hi=mean + half, n=int(v.size))


def _ci_cols(prefix, vals, mean=True) -> dict:
    ci = fold_ci(vals)
    out = {prefix: ci["mean"]} if mean else {}
    out.update({f"{prefix}_se": ci["se"], f"{prefix}_lo": ci["lo"],
                f"{prefix}_hi": ci["hi"]})
    return out


# --------------------------------------------------------------------------- #
# Known groups
# --------------------------------------------------------------------------- #

def _usable(ax, mask, floor) -> bool:
    """Both directional groups reach the per-fold support floor."""
    return (int((ax["v"] & mask).sum()) >= floor
            and int((ax["vp"] & mask).sum()) >= floor)


def _levels(ax, mask, floor):
    """(selected rows, level label per row, number of levels) in ``mask``:
    the two directional groups, plus the neutral one if it clears the floor."""
    lv = [(ax["v_name"], ax["v"] & mask), (ax["vp_name"], ax["vp"] & mask)]
    if ax.get("w") is not None and int((ax["w"] & mask).sum()) >= floor:
        lv.append((ax["w_name"], ax["w"] & mask))
    sel = np.zeros(len(mask), dtype=bool)
    lab = np.empty(len(mask), dtype=object)
    for name, m in lv:
        sel |= m
        lab[m] = name
    return sel, lab, len(lv)


def axis_mce(f, y, ax, name, valid, fold, floor) -> dict:
    """Per-group MCE, |bias gap| and ECCE of one axis, averaged over the folds
    where the axis is usable."""
    n_w = int((ax["w"] & valid).sum()) if ax.get("w") is not None else 0
    support = ax["v"] | ax["vp"] | (ax["w"] if ax.get("w") is not None else False)
    per_fold, levels = [], []
    ecce = {k: [] for k in ("axis_ecce_perc", "axis_ecce_sigma", "ecce_v_perc",
                            "ecce_v_sigma", "ecce_vp_perc", "ecce_vp_sigma",
                            "ecce_w_perc", "ecce_w_sigma")}
    for k in np.unique(fold):
        fmask = valid & (fold == k)
        if not _usable(ax, fmask, floor):
            continue
        sel, lab, n_lv = _levels(ax, fmask, floor)
        d = ecce_mce(f[sel], y[sel], seg_df=pd.DataFrame({name: lab[sel]}),
                     categorical_cols=[name], max_values=8)
        bg = bias_gap(f, y, ax["v"] & fmask, ax["vp"] & fmask)
        per_fold.append(dict(mce_perc=d["mce_perc"], mce_sigma=d["mce_sigma"],
                             abs_bias_gap=abs(bg) if np.isfinite(bg) else np.nan))
        levels.append(n_lv)
        s = segment_ecce(f[sel], y[sel], lab[sel], 8)
        if s is None:
            continue
        ecce["axis_ecce_perc"].append(s["global_perc"])
        ecce["axis_ecce_sigma"].append(s["global_sigma"])
        for tag in ("v", "vp", "w"):
            cell = s["cells"].get(str(ax.get(f"{tag}_name")))
            if cell is not None:
                ecce[f"ecce_{tag}_perc"].append(cell[0])
                ecce[f"ecce_{tag}_sigma"].append(cell[1])
    dfp = pd.DataFrame(per_fold, columns=["mce_perc", "mce_sigma", "abs_bias_gap"])
    out = dict(axis=name, v_name=ax["v_name"], vp_name=ax["vp_name"],
               mce_perc=float(dfp["mce_perc"].mean(skipna=True)) if len(dfp) else np.nan,
               mce_sigma=float(dfp["mce_sigma"].mean(skipna=True)) if len(dfp) else np.nan,
               abs_bias_gap=float(dfp["abs_bias_gap"].mean(skipna=True)) if len(dfp) else np.nan,
               n=int((support & valid).sum()))
    out.update(_ci_cols("mce_perc", dfp["mce_perc"], mean=False))
    out.update(_ci_cols("mce_sigma", dfp["mce_sigma"], mean=False))
    out.update(mce_n_folds=fold_ci(dfp["mce_perc"])["n"],
               w_name=ax.get("w_name", ""), n_w=n_w,
               n_levels=float(np.mean(levels)) if levels else float("nan"))
    for key, vals in ecce.items():
        if key.startswith("axis_ecce"):
            out.update(_ci_cols(key, vals))
        else:
            out[key] = fold_ci(vals)["mean"]
    out["axis_ecce_n_folds"] = fold_ci(ecce["axis_ecce_perc"])["n"]
    return out


def joint_mce(f, y, axes, valid, fold, floor) -> dict:
    """MCE over the three axes and their intersections, per fold."""
    per_fold = []
    for k in np.unique(fold):
        fmask = valid & (fold == k)
        cols = {}
        for name in C.AXES:
            if _usable(axes[name], fmask, floor):
                _, lab, _ = _levels(axes[name], fmask, floor)
                cols[name] = np.where(pd.isna(lab), "na", lab)[fmask]
        if cols:
            per_fold.append(ecce_mce(f[fmask], y[fmask], seg_df=pd.DataFrame(cols),
                                     categorical_cols=list(cols), max_values=8))
    out = {}
    for unit in ("perc", "sigma"):
        out.update(_ci_cols(f"joint_mce_{unit}", [d[f"mce_{unit}"] for d in per_fold]))
    out["joint_mce_n_folds"] = fold_ci([d["mce_perc"] for d in per_fold])["n"]
    return out


# --------------------------------------------------------------------------- #
# Unspecified groups: segments of an evaluation-only embedding PCA
# --------------------------------------------------------------------------- #

def eval_coords(parts, pos, k: int = 8, seed: int = 0) -> pd.DataFrame:
    """PCA of [e(Q), e(R_A), e(R_B)] over all rows (no labels, no scores)."""
    from sklearn.decomposition import PCA
    e_lo = np.asarray(parts["lo"], dtype=np.float32)
    e_hi = np.asarray(parts["hi"], dtype=np.float32)
    posc = pos[:, None]
    X = np.concatenate([np.asarray(parts["prompt"], dtype=np.float32),
                        np.where(posc, e_lo, e_hi), np.where(posc, e_hi, e_lo)], axis=1)
    k = max(1, min(int(k), X.shape[0] - 1, X.shape[1]))
    return pd.DataFrame(PCA(n_components=k, random_state=seed).fit_transform(X),
                        columns=[f"eval_pc{i}" for i in range(k)])


EMB_SEG = dict(max_depth=3, max_values=3, min_samples=10, max_n_segments=1000)
_EMB_KEYS = ("emb_mce_perc", "emb_mce_sigma", "emb_mce_pvalue",
             "emb_n_segments", "ecce_perc", "ecce_sigma")


def _emb_mce(f, y, coords) -> dict:
    from mcgrad.metrics import MulticalibrationError
    df = coords.reset_index(drop=True).copy()
    df["__y"] = np.asarray(y, dtype=np.float64)
    df["__p"] = np.clip(np.asarray(f, dtype=np.float64), 1e-7, 1 - 1e-7)
    m = MulticalibrationError(
        df=df, label_column="__y", score_column="__p",
        numerical_segment_columns=list(coords.columns),
        max_depth=EMB_SEG["max_depth"],
        max_values_per_segment_feature=EMB_SEG["max_values"],
        min_samples_per_segment=EMB_SEG["min_samples"],
        max_n_segments=EMB_SEG["max_n_segments"])
    return dict(emb_mce_perc=float(m.mce_relative), emb_mce_sigma=float(m.mce_sigma),
                emb_mce_pvalue=float(m.mce_pvalue),
                emb_n_segments=int(m.total_number_segments),
                ecce_perc=float(m.global_ecce_relative),
                ecce_sigma=float(m.global_ecce_sigma))


def embedding_segment_mce(f, y, coords, fold, valid) -> dict:
    """MCE over the embedding segments, per fold (``*_pooled``: all folds)."""
    pooled = _emb_mce(f[valid], y[valid], coords.loc[valid])
    out = {f"{k}_pooled": v for k, v in pooled.items()}
    floor = max(2 * EMB_SEG["min_samples"], 50)
    per_fold = [_emb_mce(f[m], y[m], coords.loc[m])
                for k in np.unique(fold)
                for m in [valid & (fold == k)] if int(m.sum()) >= floor]
    if not per_fold:
        nan = float("nan")
        return {**pooled, **out,
                **{f"emb_mce_{u}_{s}": nan for u in ("perc", "sigma")
                   for s in ("se", "lo", "hi")}, "emb_n_folds": 0}
    for key in _EMB_KEYS:
        vals = [d[key] for d in per_fold]
        out.update(_ci_cols(key, vals) if key in ("emb_mce_perc", "emb_mce_sigma")
                   else {key: fold_ci(vals)["mean"]})
    out["emb_n_segments"] = float(out["emb_n_segments"])
    out["emb_n_folds"] = len(per_fold)
    return out


# --------------------------------------------------------------------------- #
# Result tables
# --------------------------------------------------------------------------- #

def _locked_update(path: str, update) -> None:
    """Read-modify-write ``path`` under an exclusive lock, atomically."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            out = update(pd.read_csv(path, float_precision="round_trip")
                         if os.path.exists(path) else None)
            out.to_csv(f"{path}.tmp.{os.getpid()}", index=False)
            os.replace(f"{path}.tmp.{os.getpid()}", path)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def write_rows(path: str, rows: pd.DataFrame) -> None:
    """Replace this block's (dataset, judge, embedder) rows in ``path``."""
    keys = ["dataset", "judge", "embedder"]

    def _update(prev):
        if prev is not None:
            old = pd.MultiIndex.from_frame(prev[keys].astype(str))
            new = pd.MultiIndex.from_frame(rows[keys].astype(str))
            out = pd.concat([prev[~old.isin(new)], rows], ignore_index=True)
        else:
            out = rows
        order = {m: i for i, m in enumerate(C.METHODS)}
        return (out.assign(__o=out["method"].map(order))
                .sort_values(keys + ["__o"], kind="stable").drop(columns="__o"))
    _locked_update(path, _update)


def run_bias(a) -> None:
    d = C.build_oof(a.dataset, a.judge, a.embedder)
    panel, single, fold = d["panel"], d["single"], d["fold"]
    y = C.shown_y(panel).astype(float)
    axes = C.paper_axes(panel, a.judge)
    coords = eval_coords(d["parts"], panel["pos_a_is_lo"].values.astype(bool))
    for name, ax in axes.items():
        print(f"    {name:<9} " + "  ".join(
            f"{ax[t + '_name']}={int(ax[t].sum()):,}" for t in ("v", "vp", "w")
            if ax.get(t) is not None))
    rows = []
    for m in C.present_methods(single):
        f = C.method_score(panel, single, m)
        valid = np.isfinite(f)
        emb = embedding_segment_mce(f, y, coords, fold, valid)
        joint = joint_mce(f, y, axes, valid, fold, a.min_axis_n)
        base = dict(dataset=a.dataset, judge=a.judge, embedder=a.embedder,
                    method=m, family=C.FAMILY[m])
        for name in C.AXES:
            rows.append({**base, **axis_mce(f, y, axes[name], name, valid, fold,
                                            a.min_axis_n), **joint, **emb})
        print(f"    [{m:<22}] " + " ".join(
            f"{r['axis']}={r['mce_sigma']:.2f}" for r in rows[-3:])
            + f" joint={joint['joint_mce_sigma']:.2f} "
              f"emb-seg={emb['emb_mce_sigma']:.2f} (MCEsigma)", flush=True)
    write_rows(Q1_CSV, pd.DataFrame(rows))
    print(f"updated {Q1_CSV}")


def _perf(f, y) -> dict:
    p = np.clip(f, data.EPS, 1 - data.EPS)
    out = dict(acc=float(np.mean((f >= 0.5) == (y >= 0.5))),
               brier=float(np.mean((f - y) ** 2)),
               log_loss=float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
               if p.size else float("nan"))
    e = ecce_mce(f, y)
    out["ecce_perc"], out["ecce_sigma"] = e["ecce_perc"], e["ecce_sigma"]
    return out


def bootstrap_ci(stat, n: int, n_boot: int = 1000, seed: int = 0):
    """Point estimate and percentile 95% interval over item resamples."""
    rng = np.random.default_rng(seed)
    boots = np.array([stat(rng.integers(0, n, size=n)) for _ in range(n_boot)])
    boots = boots[np.isfinite(boots)]
    point = float(stat(np.arange(n)))
    if boots.size == 0:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def run_perf(a) -> None:
    d = C.build_oof(a.dataset, a.judge, a.embedder)
    panel, sym, fold = d["panel"], d["sym"], d["fold"]
    y = C.shown_y(panel).astype(float)
    f0 = C.method_score(panel, sym, "verbalized")
    rows = []
    for m in C.present_methods(sym):
        f = C.method_score(panel, sym, m)
        valid = np.isfinite(f)
        row = dict(dataset=a.dataset, judge=a.judge, embedder=a.embedder,
                   method=m, family=C.FAMILY[m], n=int(valid.sum()),
                   **_perf(f[valid], y[valid]))
        per_fold = [_perf(f[s], y[s]) for k in np.unique(fold)
                    for s in [valid & (fold == k)]
                    if int(s.sum()) >= 20 and len(np.unique(y[s])) > 1]
        for key in ("acc", "brier", "log_loss", "ecce_perc", "ecce_sigma"):
            ci = fold_ci([p[key] for p in per_fold])
            row.update({f"{key}_fold_mean": ci["mean"], f"{key}_se": ci["se"],
                        f"{key}_lo": ci["lo"], f"{key}_hi": ci["hi"]})
        row["n_folds_scored"] = len(per_fold)
        rows.append(row)
        if m in C.MCGRAD_METHODS:
            ok = valid & np.isfinite(f0)
            diff = (((f[ok] >= 0.5) == (y[ok] >= 0.5)).astype(float)
                    - ((f0[ok] >= 0.5) == (y[ok] >= 0.5)).astype(float))
            pt, lo, hi = bootstrap_ci(lambda i: float(diff[i].mean()), len(diff))
            print(f"    {m}: accuracy gain over f0 {pt * 100:+.2f} pp "
                  f"[{lo * 100:+.2f}, {hi * 100:+.2f}]")
    perf = pd.DataFrame(rows)
    ref = perf.set_index("method").loc["verbalized"]
    for col in ("acc", "brier", "log_loss", "ecce_perc", "ecce_sigma"):
        perf[f"delta_{col}"] = perf[col] - ref[col]
    print(perf[["method", "acc", "brier", "log_loss", "ecce_perc"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    write_rows(Q2_CSV, perf)
    print(f"updated {Q2_CSV}")


LOWFLOOR_CSV = os.path.join(C.RESULTS_DIR, "Q1_family_lowfloor.csv")
_LOWFLOOR_COLS = ("mce_perc", "mce_sigma", "abs_bias_gap", "n", "mce_perc_se",
                  "mce_perc_lo", "mce_perc_hi", "mce_sigma_se", "mce_sigma_lo",
                  "mce_sigma_hi", "mce_n_folds", "w_name", "n_w", "n_levels")


def run_family_lowfloor(a) -> None:
    """The family group alone at floor ``--floor``, off the cached OOF fit."""
    for ds in [s for s in a.datasets.split(",") if s]:
        d = C.build_oof(ds, a.judge, a.embedder)
        panel, single, fold = d["panel"], d["single"], d["fold"]
        y = C.shown_y(panel).astype(float)
        ax = C.paper_axes(panel, a.judge)["family"]
        print(f"=== {ds} / {a.judge} / {a.embedder}: family sides per fold "
              + str([(int((ax["v"] & (fold == k)).sum()),
                      int((ax["vp"] & (fold == k)).sum()))
                     for k in np.unique(fold)]))
        rows = []
        for m in C.present_methods(single):
            f = C.method_score(panel, single, m)
            valid = np.isfinite(f)
            r = axis_mce(f, y, ax, "family", valid, fold, a.floor)
            used = sum(_usable(ax, valid & (fold == k), a.floor)
                       for k in np.unique(fold))
            rows.append(dict(dataset=ds, judge=a.judge, embedder=a.embedder,
                             method=m, family=C.FAMILY[m], axis="family",
                             v_name=ax["v_name"], vp_name=ax["vp_name"],
                             min_axis_n=a.floor, n_folds_used=int(used),
                             **{k: r[k] for k in _LOWFLOOR_COLS}))
            print(f"    [{m:<22}] family MCE% {r['mce_perc']:.2f} / MCEsigma "
                  f"{r['mce_sigma']:.2f} ({used} folds)")
        write_rows(LOWFLOOR_CSV, pd.DataFrame(rows))
    print(f"updated {LOWFLOOR_CSV}")


def run_summary(a) -> None:
    """Means across datasets, one embedder at a time."""
    suffix = ("" if a.embedder == C.DEFAULT_EMBEDDER
              else f"__{embed.safe_model_name(a.embedder)}")
    for path, name in ((Q1_CSV, "bias"), (Q2_CSV, "perf")):
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df = df[df["embedder"] == a.embedder]
        if df.empty:
            continue
        if name == "bias":
            per = ["joint_mce_perc", "joint_mce_sigma", "emb_mce_sigma",
                   "emb_mce_perc", "ecce_perc", "ecce_sigma"]
            per_m = df.groupby(["dataset", "judge", "embedder", "method"])[per] \
                .first().reset_index()
            out = df.groupby(["method", "axis"])["mce_sigma"].mean().unstack()
            out = out.join(df.groupby(["method", "axis"])["axis_ecce_perc"].mean()
                           .unstack().add_suffix("_ecce%"))
            out = out.join(per_m.groupby("method")[per].mean())
        else:
            out = df.groupby("method")[["acc", "delta_acc", "brier", "log_loss",
                                        "ecce_perc", "delta_ecce_perc",
                                        "ecce_sigma"]].mean()
            out["n_datasets"] = df.groupby("method")["dataset"].nunique()
        out = out.reindex([m for m in C.METHODS if m in out.index])
        dest = os.path.join(C.RESULTS_DIR, f"summary_{name}{suffix}.csv")
        out.to_csv(dest)
        print(f"=== {name}, mean across datasets ({a.embedder}) -> {dest}")
        print(out.to_string(float_format=lambda v: f"{v:.3f}"))


# --------------------------------------------------------------------------- #
# Decodability diagnostic
# --------------------------------------------------------------------------- #

def run_probe(X, label, seed=0, val_frac=0.2, test_frac=0.2):
    """Six LightGBM probes fit on a train split, the best by validation AUC
    scored on a held-out test split. Returns None when a side is too thin."""
    import lightgbm as lgb
    from scipy.stats import rankdata
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    X, label = np.asarray(X, dtype=np.float64), np.asarray(label, dtype=int)
    n, n_pos = len(label), int(label.sum())
    if n < 60 or n_pos < 15 or n - n_pos < 15:
        return None
    X_tr, X_rest, y_tr, y_rest = train_test_split(
        X, label, test_size=val_frac + test_frac, random_state=seed, stratify=label)
    X_va, X_te, y_va, y_te = train_test_split(
        X_rest, y_rest, test_size=test_frac / (val_frac + test_frac),
        random_state=seed, stratify=y_rest)
    best = None
    for name, leaves, depth, min_child, lr in (
            ("lgbm_shallow", 15, 4, 20, 0.05), ("lgbm_medium", 31, 6, 20, 0.05),
            ("lgbm_deep", 63, -1, 20, 0.05), ("lgbm_shallow_reg", 15, 4, 40, 0.02),
            ("lgbm_medium_reg", 31, 6, 40, 0.02), ("lgbm_deep_fast", 63, -1, 10, 0.1)):
        clf = lgb.LGBMClassifier(num_leaves=leaves, max_depth=depth,
                                 min_child_samples=min_child, n_estimators=300,
                                 learning_rate=lr, class_weight="balanced",
                                 verbosity=-1, random_state=0).fit(X_tr, y_tr)
        auc = (roc_auc_score(y_va, clf.predict_proba(X_va)[:, 1])
               if len(np.unique(y_va)) > 1 else float("nan"))
        if np.isfinite(auc) and (best is None or auc > best[2]):
            best = (name, clf, auc)
    if best is None:
        return None
    p = best[1].predict_proba(X_te)[:, 1]
    two = len(np.unique(y_te)) > 1
    auc_se = float("nan")
    pos, neg = p[y_te == 1], p[y_te == 0]
    if two and len(pos) >= 2 and len(neg) >= 2:          # DeLong SE
        r = rankdata(np.concatenate([pos, neg]))
        v10 = (r[:len(pos)] - rankdata(pos)) / len(neg)
        v01 = 1.0 - (r[len(pos):] - rankdata(neg)) / len(pos)
        auc_se = float(np.sqrt(v10.var(ddof=1) / len(pos) + v01.var(ddof=1) / len(neg)))
    acc = float(np.mean((p >= 0.5) == y_te))
    return dict(n=n, pos_rate=float(label.mean()), best_config=best[0],
                test_auc=float(roc_auc_score(y_te, p)) if two else float("nan"),
                test_auc_se=auc_se, test_acc=acc,
                test_acc_se=float(np.sqrt(acc * (1 - acc) / len(y_te))),
                majority_acc=float(max(y_te.mean(), 1 - y_te.mean())))


def _decodability_groups(panel, dataset, judge) -> dict:
    out = {f"{k} (LO frame)": v for k, v in C.lo_axes(panel, judge).items()}
    pos = panel["pos_a_is_lo"].fillna(False).values.astype(bool)
    slots = lambda lo, hi: (np.where(pos, lo, hi), np.where(pos, hi, lo))
    jf = data.judge_family(judge)
    fa, fb = slots(panel["family_lo"].astype(str).str.lower().values == jf,
                   panel["family_hi"].astype(str).str.lower().values == jf)
    out["family (A/B frame)"] = dict(v=fa & ~fb, vp=fb & ~fa)
    la, lb = slots(panel["len_lo"].values.astype(float),
                   panel["len_hi"].values.astype(float))
    out["length (A/B frame)"] = dict(v=la > lb, vp=lb > la)
    if dataset == "pku_saferlhf":     # which response is unsafe
        try:
            raw = data._hf_dataset("PKU-Alignment/PKU-SafeRLHF", "train")
            raw = raw.assign(qid=[data.pku_qid(p, a, b) for p, a, b in zip(
                raw.prompt, raw.response_0, raw.response_1)])
            m = raw.drop_duplicates("qid").set_index("qid").reindex(
                panel["qid"].astype(str).values)
            sa, sb = slots(m["is_response_0_safe"].fillna(True).values.astype(bool),
                           m["is_response_1_safe"].fillna(True).values.astype(bool))
            out["safety (A/B frame)"] = dict(v=~sa & sb, vp=~sb & sa)
        except Exception as e:
            print(f"[pku] raw dataset unavailable ({e}); safety group skipped")
    return out


def run_decodability(a) -> None:
    """For each group, probe test AUC from CaLLM's features PCA_k(e_A - e_B)
    at the adopted width (the most common one across the outer folds), the full difference, and (+f0) with the judge's
    logit f0; 0.5 = not decodable."""
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split
    key = ["dataset", "judge", "embedder", "group", "representation"]
    for ds in [s for s in a.datasets.split(",") if s]:
        print(f"\n=== {ds} ===")
        rows = []
        panel = data.build_panel(ds, a.judge)
        parts = embed.encode_parts(ds, panel, a.embedder)
        pos = panel["pos_a_is_lo"].fillna(False).values.astype(bool)
        ea = np.where(pos[:, None], parts["lo"], parts["hi"])
        eb = np.where(pos[:, None], parts["hi"], parts["lo"])
        widths = [int(((C.load_tuned(ds, a.judge, a.embedder, f)[0] or {})
                       .get(C.CALLM) or {}).get("n_pca") or C.CONTRAST_DEFAULT_WIDTH)
                  for f in range(C.N_FOLDS)]
        k = Counter(widths).most_common(1)[0][0]     # the folds' modal width
        print(f"  CaLLM width per fold: {widths} -> k={k}")
        f0 = np.clip(C.method_score(panel, {}, "verbalized"), 1e-6, 1 - 1e-6)
        lf0 = np.log(f0 / (1 - f0))[:, None]
        for name, ax in _decodability_groups(panel, ds, a.judge).items():
            sel = np.where(np.asarray(ax["v"], bool) | np.asarray(ax["vp"], bool))[0]
            if len(sel) > 20000:
                sel = np.sort(np.random.default_rng(0).choice(sel, 20000,
                                                              replace=False))
            lab = np.asarray(ax["v"], bool)[sel].astype(int)
            if len(sel) < 60 or lab.sum() < 15 or len(lab) - lab.sum() < 15:
                print(f"  {name}: too thin ({len(sel)} rows) -- skipped")
                continue
            tr, _ = train_test_split(np.arange(len(sel)), test_size=0.4,
                                     random_state=0, stratify=lab)
            d = ea[sel][tr] - eb[sel][tr]              # PCA fit on train only
            kk = max(1, min(k, 2 * len(tr) - 1, d.shape[1]))
            pca = PCA(n_components=kk, random_state=0).fit(np.concatenate([d, -d]))
            reps = {f"pca_{k}": pca.transform(ea[sel] - eb[sel]),
                    "full_diff": ea[sel] - eb[sel]}
            reps = {**reps, **{f"{r}+f0": np.hstack([X, lf0[sel]])
                               for r, X in reps.items()}, "f0": lf0[sel]}
            for rep, X in reps.items():
                r = run_probe(X, lab)
                if r is None:
                    continue
                rows.append(dict(dataset=ds, judge=a.judge, embedder=a.embedder,
                                 group=name, representation=rep,
                                 n_features=X.shape[1], n=r["n"],
                                 pos_rate=r["pos_rate"], test_auc=r["test_auc"],
                                 test_auc_se=r["test_auc_se"],
                                 test_acc=r["test_acc"],
                                 test_acc_se=r["test_acc_se"],
                                 majority_acc=r["majority_acc"],
                                 best_config=r["best_config"]))
                print(f"  {name:24s} {rep:14s} AUC={r['test_auc']:.3f}"
                      f"±{r['test_auc_se']:.3f} n={r['n']:,}")
        if rows:      # merged per dataset, so a long sweep keeps what it finished
            new = pd.DataFrame(rows)
            _locked_update(DECODABILITY_CSV, lambda prev: new if prev is None
                           else pd.concat([prev, new], ignore_index=True)
                           .drop_duplicates(key, keep="last"))
            print(f"  -> {DECODABILITY_CSV}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=["bias", "perf", "family_lowfloor", "summary",
                                    "decodability"])
    p.add_argument("--dataset", choices=data.DATASETS)
    p.add_argument("--datasets", default=None,
                   help="decodability (default: all) / family_lowfloor "
                        "(default: rewardbench,mtbench)")
    p.add_argument("--judge", default=data.DEFAULT_JUDGE)
    p.add_argument("--embedder", default=C.DEFAULT_EMBEDDER)
    p.add_argument("--min_axis_n", type=int, default=30,
                   help="per-fold support floor of a group")
    p.add_argument("--floor", type=int, default=20,
                   help="family_lowfloor's per-fold support floor")
    a = p.parse_args()
    if a.what in ("bias", "perf") and not a.dataset:
        p.error(f"{a.what} needs --dataset")
    if a.datasets is None:
        a.datasets = ("rewardbench,mtbench" if a.what == "family_lowfloor"
                      else ",".join(data.DATASETS))
    dict(bias=run_bias, perf=run_perf, family_lowfloor=run_family_lowfloor,
         summary=run_summary, decodability=run_decodability)[a.what](a)


if __name__ == "__main__":
    main()
