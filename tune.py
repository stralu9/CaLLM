"""Hyperparameter search, nested inside the evaluation's outer folds.

Protocol (all methods), for every outer fold k of the 5 question-grouped folds
``build_oof`` scores on:
  * the other four folds are split at random, whole questions at a time, into
    80% fit / 20% validation (``--val_frac``); fold k itself is never read;
  * every config is fit on the 80% and scored by its log loss on the 20%;
  * the lowest validation log loss is adopted as fold k's config, written to
    ``results/tuned_lgb.json`` / ``results/tuned_baselines.json`` under
    ``[dataset[__judge][__embedder]][method]["fold<k>"]``.
``build_oof`` then refits fold k's config on all four training folds and
scores it on fold k, so every fold has its own config and no test fold ever
influences the config it is scored with.

    python tune.py mcgrad --dataset mtbench --judge muse \
        --embedder muse-internal --method mcgrad_gen_cdiff_nc
        CaLLM / CaLLM-C: for each PCA width in {4, 8, 16, 32}, one
        ``mcgrad.tuning`` (Ax/BoTorch) search of --n_trials trials, each trial
        fit on the 80% and scored on the 20%; the library default and the four
        width winners are then refit on the 80% and compared on the 20%
        (ties -> the default).
    python tune.py baselines --dataset mtbench --judge muse --embedder muse-internal
        the block's tunable baselines, Optuna TPE, --n_configs trials, the
        first one the default: histogram, CalibraEval, LenControl and IGLB on
        the main block (muse, muse-internal); CalibraEval and LenControl on
        the ablation blocks, which fit no other baseline.
    python tune.py calibraeval --dataset mtbench --judge muse
        CalibraEval never reads an embedding: per fold, re-rank the configs
        adopted by the different embedder blocks of one judge on that fold's
        validation split and adopt the best one in all of them.
    --folds 0 2   tunes only those outer folds (default: all).

Search logs: ``results/tuning/<key>__<method>__fold<k>.csv`` (every compared
config) and ``results/tuning/ax_trials/<key>__<method>__fold<k>.csv``.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os

import numpy as np
import pandas as pd

import callm as C
import data
from data import EPS

VAL_SEED = 1000                 # the 80/20 split of outer fold k uses VAL_SEED + k


def build_context(dataset, judge, embedder, seed=0):
    panel = data.build_panel(dataset, judge, seed)
    feat = C.Features(dataset, panel, judge, embedder)
    groups = C.question_groups(panel)
    fold = C.outer_folds(panel, seed)
    print(f"panel rows: {len(panel):,} | questions: {groups.max() + 1:,} | "
          f"outer fold sizes: {np.bincount(fold).tolist()}")
    return dict(panel=panel, feat=feat, groups=groups, fold=fold)


def grouped_holdout(groups: np.ndarray, frac: float, seed: int):
    """Random question-grouped split: whole groups, in random order, go to
    validation until it holds ``frac`` of the rows. (fit, val) row indices."""
    codes = pd.factorize(np.asarray(groups))[0]
    sizes = np.bincount(codes)
    order = np.random.default_rng(seed).permutation(len(sizes))
    n_val = int(np.searchsorted(np.cumsum(sizes[order]), frac * len(codes))) + 1
    is_val = np.zeros(len(sizes), dtype=bool)
    is_val[order[:n_val]] = True
    return np.where(~is_val[codes])[0], np.where(is_val[codes])[0]


def fold_split(ctx, k: int, val_frac: float):
    """(fit, val) rows of outer fold k: an 80/20 question-grouped split of the
    rows outside fold k."""
    train = np.where(ctx["fold"] != k)[0]
    fit, val = grouped_holdout(ctx["groups"][train], val_frac, VAL_SEED + k)
    fit, val = train[fit], train[val]
    assert not np.isin(ctx["fold"][np.r_[fit, val]], [k]).any()
    assert not np.intersect1d(ctx["groups"][fit], ctx["groups"][val]).size
    print(f"\n--- outer fold {k}: fit {len(fit):,} | validation {len(val):,} "
          f"| test (untouched) {int((ctx['fold'] == k).sum()):,} rows ---",
          flush=True)
    return fit, val


def val_scores(feat, val, f_lo) -> dict:
    """Validation accuracy / Brier / log loss of P(LO better) ``f_lo``, and
    whether the fit returned f0 unchanged (``identity``)."""
    pos, y = feat.pos[val], feat.y[val].astype(float)
    f = np.clip(np.where(pos, f_lo, 1.0 - f_lo), EPS, 1 - EPS)
    return dict(acc=float(np.mean((f >= 0.5) == (y >= 0.5))),
                brier=float(np.mean((f - y) ** 2)),
                log_loss=float(-np.mean(y * np.log(f) + (1 - y) * np.log(1 - f))),
                identity=float(np.allclose(f, feat.s0[val], atol=1e-9)))


def _row(cfg, agg, is_default, trial, fold, fit, val):
    return dict(outer_fold=fold, config=json.dumps(cfg, sort_keys=True), **cfg,
                acc=agg["acc"], brier=agg["brier"], log_loss=agg["log_loss"],
                identity=agg["identity"], is_default=is_default, trial=trial,
                n_fit=len(fit), n_val=len(val))


def _fmt(agg):
    return (f"val log_loss={agg['log_loss']:.4f} brier={agg['brier']:.4f} "
            f"acc={agg['acc']:.4f} identity={bool(agg['identity'])}")


def adopt(path, key, fold, method, cfg) -> None:
    """Write ``cfg`` as ``[key][method][fold<k>]`` (locked read-modify-write,
    so tune jobs of different folds / methods can run concurrently)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(f"{path}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        adopted = json.load(open(path)) if os.path.exists(path) else {}
        entry = adopted.setdefault(key, {})
        if not C.is_per_fold(entry.get(method)):
            entry[method] = {}                  # drop an old single config
        entry[method][C.fold_key(fold)] = cfg
        with open(f"{path}.tmp", "w") as fh:
            json.dump(adopted, fh, indent=2, sort_keys=True)
        os.replace(f"{path}.tmp", path)


def finish(dataset, judge, embedder, method, fold, rows) -> None:
    """Write the ranked search log of one outer fold and adopt its winner
    (lowest validation log loss)."""
    key = C.out_key(dataset, judge, embedder)
    df = pd.DataFrame(rows)
    df = (df.assign(_k=df["log_loss"].astype(float).fillna(np.inf),
                    _d=~df["is_default"].astype(bool))    # ties -> the default
          .sort_values(["_k", "_d"], kind="stable").drop(columns=["_k", "_d"])
          .reset_index(drop=True))
    os.makedirs(C.TUNING_DIR, exist_ok=True)
    df.to_csv(os.path.join(C.TUNING_DIR, f"{key}__{method}__{C.fold_key(fold)}.csv"),
              index=False)
    print(df[["config", "log_loss", "acc", "brier", "identity",
              "is_default"]].head(10).to_string(index=False))
    if not np.isfinite(float(df["log_loss"].iloc[0])):
        raise RuntimeError(f"{key} / {method} / fold {fold}: every config failed")
    best = json.loads(df["config"].iloc[0])
    path = (C.TUNED_BASELINES_JSON if method in C.BASELINE_DEFAULTS
            else C.TUNED_LGB_JSON)
    adopt(path, key, fold, method, best)
    print(f"{key} / {method} / fold {fold}: adopted {best} "
          f"(val log_loss={df['log_loss'].iloc[0]:.4f})")


def tune_mcgrad(a) -> None:
    ctx = build_context(a.dataset, a.judge, a.embedder)
    key, feat = C.out_key(a.dataset, a.judge, a.embedder), ctx["feat"]
    default = C.library_default_lgb()
    trial_dir = os.path.join(C.TUNING_DIR, "ax_trials")
    os.makedirs(trial_dir, exist_ok=True)
    for k in a.folds:
        C.seed_everything(a.seed)          # each fold reproducible on its own
        fit, val = fold_split(ctx, k, a.val_frac)
        tables, winners = [], []
        for w in C.CONTRAST_WIDTH_GRID:
            record = []
            C.predict_split(feat, fit, val, [a.method], want_sym=False,
                            mcgrad_cfg={a.method: {"n_pca": w}},
                            tune=dict(n_trials=a.n_trials, record=record))
            tbl = record[0]["trials"].copy()
            tbl.insert(0, "n_pca", w)
            tbl.insert(0, "outer_fold", k)
            tables.append(tbl)
            winners.append((record[0]["params"], w))
            print(f"  [n_pca={w}] n_features={record[0]['n_features']} -> "
                  f"{record[0]['params']}", flush=True)
        pd.concat(tables, ignore_index=True).to_csv(os.path.join(
            trial_dir, f"{key}__{a.method}__{C.fold_key(k)}.csv"), index=False)

        # the library default and the width winners, refit on the fit rows
        todo = [(default, C.CONTRAST_DEFAULT_WIDTH, True)] + [
            (cfg, w, False) for cfg, w in winners]
        rows = []
        for i, (cfg, w, is_ref) in enumerate(todo):
            preds = C.predict_split(feat, fit, val, [a.method], want_sym=False,
                                    mcgrad_cfg={a.method: dict(cfg, n_pca=w)})
            agg = val_scores(feat, val, preds[a.method][0])
            rows.append(_row(dict(cfg, n_pca=w), agg, is_ref, i, k, fit, val))
            print(f"  [{'default' if is_ref else 'winner'} n_pca={w}] "
                  f"{_fmt(agg)}", flush=True)
        finish(a.dataset, a.judge, a.embedder, a.method, k, rows)


def _suggest(method, trial) -> dict:
    """Optuna search space of each tunable baseline (brackets its default)."""
    if method == "histogram":
        return dict(n_bins=trial.suggest_int("n_bins", 2, 60, log=True))
    if method == "calibraeval":
        return dict(lam=trial.suggest_float("lam", 1e-2, 9e-1, log=True),
                    lr=trial.suggest_float("lr", 1e-3, 3e-1, log=True),
                    max_iter=trial.suggest_int("max_iter", 50, 2000, log=True))
    if method == "length_control":
        return dict(C=trial.suggest_float("C", 1e-3, 1e3, log=True),
                    min_freq=trial.suggest_int("min_freq", 1, 60, log=True))
    return dict(M=trial.suggest_int("M", 3, 100, log=True),              # iglb
                max_iter=trial.suggest_int("max_iter", 100, 2000, log=True),
                epsilon=trial.suggest_float("epsilon", 1e-4, 2e-1, log=True))


def tune_baselines(a) -> None:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    ctx = build_context(a.dataset, a.judge, a.embedder)
    feat = ctx["feat"]
    for k in a.folds:
        fit, val = fold_split(ctx, k, a.val_frac)
        for method in C.tunable_baselines(a.judge, a.embedder):
            default, rows = dict(C.BASELINE_DEFAULTS[method]), []
            print(f"\n=== fold {k} / {method}: {a.n_configs} Optuna trials ===")

            def objective(trial):
                cfg = _suggest(method, trial)
                try:
                    preds = C.predict_split(feat, fit, val, [method], want_sym=False,
                                            baseline_params={method: cfg})
                    agg = val_scores(feat, val, preds[method][0])
                except Exception as e:              # diverged / degenerate fit
                    print(f"      (config failed: {type(e).__name__}: {e})")
                    agg = None
                if agg is None or not np.isfinite(agg["log_loss"]):
                    agg = dict(acc=np.nan, brier=np.nan, log_loss=np.inf,
                               identity=np.nan)
                rows.append(_row(cfg, agg, cfg == default, trial.number, k, fit, val))
                print(f"  [{trial.number + 1:>3}/{a.n_configs}] {cfg} {_fmt(agg)}",
                      flush=True)
                return agg["log_loss"]

            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=a.seed))
            study.enqueue_trial(default)
            study.optimize(objective, n_trials=a.n_configs)
            finish(a.dataset, a.judge, a.embedder, method, k, rows)


def calibraeval_consistency(a) -> None:
    """One CalibraEval config per (dataset, judge, fold), shared by its blocks.
    The fold's validation split does not depend on the embedder, so the
    re-ranking still reads only that fold's training folds."""
    method, default = "calibraeval", dict(C.BASELINE_DEFAULTS["calibraeval"])
    adopted = (json.load(open(C.TUNED_BASELINES_JSON))
               if os.path.exists(C.TUNED_BASELINES_JSON) else {})
    embedders = [C.DEFAULT_EMBEDDER, "muse-internal"]
    ctx = None
    for k in a.folds:
        per_block = {emb: ((adopted.get(C.out_key(a.dataset, a.judge, emb)) or {})
                           .get(method) or {}).get(C.fold_key(k))
                     for emb in embedders}
        cands, seen = [default], {json.dumps(default, sort_keys=True)}
        for cfg in per_block.values():
            if cfg is not None and json.dumps(cfg, sort_keys=True) not in seen:
                seen.add(json.dumps(cfg, sort_keys=True))
                cands.append(cfg)
        if len(cands) < 2:
            print(f"{a.dataset} / {a.judge} / fold {k}: one CalibraEval config "
                  f"already covers every block")
            continue
        ctx = ctx or build_context(a.dataset, a.judge, None)
        fit, val = fold_split(ctx, k, a.val_frac)
        rows = []
        for i, cfg in enumerate(cands):
            preds = C.predict_split(ctx["feat"], fit, val, [method], want_sym=False,
                                    baseline_params={method: cfg})
            agg = val_scores(ctx["feat"], val, preds[method][0])
            rows.append(_row(cfg, agg, cfg == default, i, k, fit, val))
            print(f"  {cfg} {_fmt(agg)}")
        for emb, cfg in per_block.items():
            if cfg is not None:
                finish(a.dataset, a.judge, emb, method, k, rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=["mcgrad", "baselines", "calibraeval"])
    p.add_argument("--dataset", required=True, choices=data.DATASETS)
    p.add_argument("--judge", default=data.DEFAULT_JUDGE)
    p.add_argument("--embedder", default=C.DEFAULT_EMBEDDER)
    p.add_argument("--method", choices=C.MCGRAD_METHODS, help="mcgrad only")
    p.add_argument("--n_trials", type=int, default=40,
                   help="Ax trials per PCA width (mcgrad)")
    p.add_argument("--n_configs", type=int, default=40,
                   help="Optuna trials per baseline")
    p.add_argument("--folds", type=int, nargs="+", default=list(range(C.N_FOLDS)),
                   help="outer folds to tune (default: all)")
    p.add_argument("--val_frac", type=float, default=0.2,
                   help="validation share of each fold's training folds")
    p.add_argument("--seed", type=int, default=0,
                   help="global RNG seed of the Ax search / Optuna sampler seed")
    a = p.parse_args()
    if a.what == "mcgrad":
        if not a.method:
            p.error("mcgrad needs --method")
        tune_mcgrad(a)
    elif a.what == "baselines":
        tune_baselines(a)
    else:
        calibraeval_consistency(a)


if __name__ == "__main__":
    main()
