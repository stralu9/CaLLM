"""CaLLM core: bias groups, folds, features and the out-of-fold fits.

Frame. Everything is fit and scored in the shown order: for a pair shown as
(R_A, R_B), f0 = P(R_A better) from the judge's verbalized confidence and
Y = 1[R_A human-preferred]. The out-of-fold cache stores P(LO better) and is
converted back with ``method_score``.

Methods.
  CaLLM    (``mcgrad_gen_cdiff_nc``)   MCGrad on PCA_k(e(R_A) - e(R_B)); the
           PCA is fit on the calibration fold's differences in both orders,
           so the features encode what distinguishes the two answers, never
           which slot holds which. k is tuned in {4, 8, 16, 32}.
  CaLLM-C  (``mcgrad_cdiff_bias_nc``)  the same features plus the declared bias
           groups (position, verbosity, family) as categoricals.
  Baselines: verbalized f0, BPC (swap average), PORTIA, CalibraEval, LenControl,
           temperature / Platt / isotonic / histogram / beta, GCUR / IGLB.
           The calibration / multicalibration baselines (temperature ... IGLB)
           are compared on the main block (Muse judge, its own hidden states)
           only; the ablation blocks refit just CalibraEval and LenControl.

Evaluation groups (``paper_axes``), from the inputs and presentation only:
  position  LO shown first / LO shown second (LO = alphabetically first model,
            or the first of two random per-pair names when responses carry no
            model identity, as on PKU)
  length    A longer / B longer / same length
  family    A from the judge family / B from it / both or neither
Folds are grouped by question (shared qid or identical prompt text).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import data
import embed
from calibrators import (GROUPED, POSTHOC, CalibraEval, CalibraEvalConfig,
                         ClusterGroups, length_control)
from data import EPS

logging.getLogger("mcgrad").setLevel(logging.ERROR)   # cosmetic unshrink warnings

RESULTS_DIR = os.path.join(data.HERE, "results")
TUNING_DIR = os.path.join(RESULTS_DIR, "tuning")
TUNED_LGB_JSON = os.path.join(RESULTS_DIR, "tuned_lgb.json")
TUNED_BASELINES_JSON = os.path.join(RESULTS_DIR, "tuned_baselines.json")
DEFAULT_EMBEDDER = "Qwen/Qwen3-Embedding-0.6B"

CALLM = "mcgrad_gen_cdiff_nc"
CALLM_C = "mcgrad_cdiff_bias_nc"
MCGRAD_METHODS = [CALLM, CALLM_C]
REFIT_BASELINES = ["calibraeval", "length_control"] + list(POSTHOC) + list(GROUPED)
# The (judge, embedder) block of the main results, the only one on which the
# calibration / multicalibration baselines are fit (App. C). The ablation
# blocks (Qwen judge, Qwen3 embedder) refit only the mitigation baselines.
MAIN_BLOCK = ("muse", "muse-internal")
ABLATION_BASELINES = ["calibraeval", "length_control"]
METHODS = (["verbalized", "bpe", "calibraeval", "length_control", "portia"]
           + list(POSTHOC) + list(GROUPED) + [CALLM_C, CALLM])
FAMILY = {"verbalized": "verbalized", CALLM: "generic_mcgrad",
          CALLM_C: "reference",
          **{m: "bias_mitigation" for m in ("bpe", "calibraeval",
                                            "length_control", "portia")},
          **{m: "posthoc" for m in POSTHOC},
          **{m: "grouped_mc" for m in GROUPED}}

# baseline hyperparameters (the untuned defaults; tuned values override them)
BASELINE_DEFAULTS = {"histogram": dict(n_bins=15),
                     "calibraeval": dict(lam=0.1, lr=0.05, max_iter=500),
                     "length_control": dict(C=1.0, min_freq=10),
                     "iglb": dict(M=20, max_iter=500, epsilon=0.01)}
# LightGBM params of every MCGrad fit, under the tuned ones
MCGRAD_LGB_PARAMS = {"min_child_samples": 20, "lambda_l2": 1.0}
CONTRAST_DEFAULT_WIDTH = 8
CONTRAST_WIDTH_GRID = [4, 8, 16, 32]
AXES = ("position", "length", "family")
N_FOLDS = 5                     # outer question-grouped folds


def refit_baselines(judge: str, embedder: str) -> List[str]:
    """The baselines ``build_oof`` refits on this block."""
    if (judge, embedder) == MAIN_BLOCK:
        return list(REFIT_BASELINES)
    return list(ABLATION_BASELINES)


def tunable_baselines(judge: str, embedder: str) -> List[str]:
    """The baselines with free hyperparameters that this block fits."""
    return [m for m in BASELINE_DEFAULTS if m in refit_baselines(judge, embedder)]


def out_key(dataset: str, judge: str, embedder: str) -> str:
    """``dataset[__judge][__embedder]``; default judge / embedder omitted."""
    key = dataset
    if judge != data.DEFAULT_JUDGE:
        key += f"__{judge}"
    if embedder != DEFAULT_EMBEDDER:
        key += f"__{embed.safe_model_name(embedder)}"
    return key


def fold_key(k: int) -> str:
    return f"fold{int(k)}"


def is_per_fold(cfg) -> bool:
    return (isinstance(cfg, dict) and bool(cfg)
            and all(str(k).startswith("fold") for k in cfg))


def load_tuned(dataset: str, judge: str, embedder: str, fold: int):
    """(MCGrad configs, baseline configs) ``{method: cfg}`` adopted for this
    block's outer fold ``fold``, or Nones. The JSONs hold
    ``{block: {method: {"fold<k>": cfg}}}``; fold k's configs were selected on
    a validation split of fold k's training folds only (``tune.py``)."""
    out = []
    for path in (TUNED_LGB_JSON, TUNED_BASELINES_JSON):
        cfg = None
        if os.path.exists(path):
            with open(path) as fh:
                entry = json.load(fh).get(out_key(dataset, judge, embedder)) or {}
            stale = sorted(m for m, v in entry.items() if not is_per_fold(v))
            if stale:
                raise ValueError(
                    f"{path}: block {out_key(dataset, judge, embedder)!r} holds "
                    f"one config for all folds for {stale}; it was tuned on "
                    f"the test folds. Re-run tune.py (one config per fold).")
            cfg = {m: v[fold_key(fold)] for m, v in entry.items()
                   if v.get(fold_key(fold))} or None
        out.append(cfg)
    return tuple(out)


# --------------------------------------------------------------------------- #
# Groups
# --------------------------------------------------------------------------- #

def anchor_flip(panel: pd.DataFrame) -> np.ndarray:
    """True where the evaluation anchor LO is the listed HI response. Only on
    datasets without model identity (every pair is ``resp0`` vs ``resp1``),
    where the listing slot is correlated with the label: there each pair gets
    two random names hashed from its qid and LO is the first of them."""
    if not (panel["model_lo"].astype(str).nunique() == 1
            and panel["model_hi"].astype(str).nunique() == 1):
        return np.zeros(len(panel), dtype=bool)
    name = lambda q, slot: hashlib.md5(
        f"lo-anchor-names-v1:{q}:{slot}".encode()).hexdigest()[:8]
    return np.array([name(q, 1) < name(q, 0)
                     for q in panel["qid"].astype(str).values], dtype=bool)


def lo_axes(panel: pd.DataFrame, judge: str) -> dict:
    """The groups in the LO frame (LO / HI longer, LO / HI own-family); used by
    the decodability diagnostic only."""
    flip = anchor_flip(panel)
    pos = panel["pos_a_is_lo"].fillna(False).values.astype(bool) ^ flip
    len_lo = panel["len_lo"].values.astype(float)
    len_hi = panel["len_hi"].values.astype(float)
    len_lo, len_hi = np.where(flip, len_hi, len_lo), np.where(flip, len_lo, len_hi)
    jf = data.judge_family(judge)
    fam_lo = panel["family_lo"].astype(str).str.lower().values == jf
    fam_hi = panel["family_hi"].astype(str).str.lower().values == jf
    fam_lo, fam_hi = np.where(flip, fam_hi, fam_lo), np.where(flip, fam_lo, fam_hi)
    return {
        "position": dict(v=pos, vp=~pos,
                         v_name="lo_shown_first", vp_name="lo_shown_second"),
        "length": dict(v=len_lo > len_hi, vp=len_hi > len_lo,
                       v_name="lo_longer", vp_name="hi_longer"),
        "family": dict(v=fam_lo & ~fam_hi, vp=fam_hi & ~fam_lo,
                       v_name="lo_judge_family", vp_name="hi_judge_family"),
    }


def paper_axes(panel: pd.DataFrame, judge: str) -> dict:
    """The paper's groups: ``{axis: dict(v, vp, v_name, vp_name[, w, w_name])}``
    with ``v`` / ``vp`` the two directional groups and ``w`` the third
    (neutral) one. Length and family are anchored on the shown slots A / B."""
    pal = panel["pos_a_is_lo"].fillna(False).values.astype(bool)
    pos = pal ^ anchor_flip(panel)
    len_lo = panel["len_lo"].values.astype(float)
    len_hi = panel["len_hi"].values.astype(float)
    len_a, len_b = np.where(pal, len_lo, len_hi), np.where(pal, len_hi, len_lo)
    jf = data.judge_family(judge)
    fam_lo = panel["family_lo"].astype(str).str.lower().values == jf
    fam_hi = panel["family_hi"].astype(str).str.lower().values == jf
    fam_a, fam_b = np.where(pal, fam_lo, fam_hi), np.where(pal, fam_hi, fam_lo)
    ax = {"position": dict(v=pos, vp=~pos, v_name="lo_shown_first",
                           vp_name="lo_shown_second"),
          "length": dict(v=len_a > len_b, vp=len_b > len_a,
                         v_name="A_longer", vp_name="B_longer",
                         w_name="same_length"),
          "family": dict(v=fam_a & ~fam_b, vp=fam_b & ~fam_a,
                         v_name="A_judge_family", vp_name="B_judge_family",
                         w_name="both_or_neither_family")}
    for name in ("length", "family"):
        ax[name]["w"] = ~(ax[name]["v"] | ax[name]["vp"])
    return ax


def group_categoricals(panel: pd.DataFrame, judge: str):
    """CaLLM-C's declared groups as categoricals, for the shown and the
    reversed presentation: ``(shown_df, reversed_df, columns)``."""
    ax = paper_axes(panel, judge)
    pos = ax["position"]["v"]
    three = lambda a: np.where(a["v"], "A", np.where(a["vp"], "B", "none"))
    swap = lambda c: np.where(c == "A", "B", np.where(c == "B", "A", "none"))
    length, family = three(ax["length"]), three(ax["family"])
    shown = pd.DataFrame({"cat_shown_position": np.where(pos, "first", "second"),
                          "cat_shown_length": length,
                          "cat_shown_family": family})
    rev = pd.DataFrame({"cat_shown_position": np.where(pos, "second", "first"),
                        "cat_shown_length": swap(length),
                        "cat_shown_family": swap(family)})
    return shown, rev, list(shown.columns)


# --------------------------------------------------------------------------- #
# Question-grouped folds
# --------------------------------------------------------------------------- #

def question_groups(panel: pd.DataFrame) -> np.ndarray:
    """Connected components of "same qid" and "same prompt text"."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(panel)
    q_codes = pd.factorize(panel["qid"].astype(str))[0]
    prompts = panel["prompt"].fillna("").astype(str).str.strip()
    p_codes = np.where(prompts.values == "", -1, pd.factorize(prompts)[0])
    n_q, has_p = q_codes.max() + 1, p_codes >= 0
    rows = np.concatenate([np.arange(n), np.arange(n)[has_p]])
    cols = np.concatenate([n + q_codes, n + n_q + p_codes[has_p]])
    size = n + n_q + (p_codes.max() + 1 if has_p.any() else 0)
    g = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(size, size))
    return pd.factorize(connected_components(g, directed=False)[1][:n])[0]


def outer_folds(panel: pd.DataFrame, seed: int = 0,
                n_splits: int = N_FOLDS) -> np.ndarray:
    """The evaluation's question-grouped fold id per row (shared by
    ``build_oof`` and ``tune.py``)."""
    return balanced_group_folds(question_groups(panel), n_splits, seed)


def balanced_group_folds(groups: np.ndarray, n_splits: int,
                         seed: int) -> np.ndarray:
    """Fold id per row; groups go, largest first (random order among equal
    sizes), to the currently smallest fold."""
    groups = pd.factorize(np.asarray(groups))[0]
    sizes = np.bincount(groups)
    perm = np.random.default_rng(seed).permutation(len(sizes))
    fold_of_group = np.empty(len(sizes), dtype=np.int64)
    load = np.zeros(n_splits, dtype=np.int64)
    for g in perm[np.argsort(-sizes[perm], kind="stable")]:
        k = int(np.argmin(load))
        fold_of_group[g] = k
        load[k] += sizes[g]
    return fold_of_group[groups]


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #

class Features:
    """Per-row inputs of every fit, in the shown frame. ``s0`` = f0, ``s1`` =
    the reverse pass in its own frame (its first-shown answer is R_B, so its
    label is 1 - y). Without an embedder only the baselines that need no
    embedding (post-hoc, CalibraEval, LenControl) can be fit."""

    def __init__(self, dataset: str, panel: pd.DataFrame, judge: str,
                 embedder: Optional[str] = None):
        clip = lambda c: np.clip(panel[c].values.astype(float), EPS, 1 - EPS)
        pos = panel["pos_a_is_lo"].values.astype(bool)
        self.pos = pos
        self.s0 = np.where(pos, clip("p_lo_orig"), 1.0 - clip("p_lo_orig"))
        self.s1 = np.where(pos, 1.0 - clip("p_lo_flip"), clip("p_lo_flip"))
        y_lo = panel["y_lo"].values.astype(int)
        self.y = np.where(pos, y_lo, 1 - y_lo).astype(int)
        # CalibraEval's arrangements stay in the judge's token frame
        self.ce_s0, self.ce_s2 = clip("ce_s0"), clip("ce_s2")
        dlen = panel["len_lo"].values.astype(float) - panel["len_hi"].values.astype(float)
        self.dlen = np.where(pos, dlen, -dlen)                 # len(A) - len(B)
        m_lo = panel["model_lo"].astype(str).values
        m_hi = panel["model_hi"].astype(str).values
        self.model_a = np.where(pos, m_lo, m_hi)
        self.model_b = np.where(pos, m_hi, m_lo)
        self.cat, self.cat_rev, self.cat_cols = group_categoricals(panel, judge)
        self.groups = None
        if embedder is None:
            return
        posc = pos[:, None]
        emb = embed.encode_texts(dataset, embed.joint_texts(panel), embedder)
        emb_swap = embed.encode_texts(dataset, embed.joint_texts(panel, True),
                                      embedder)
        parts = embed.encode_parts(dataset, panel, embedder)
        e_lo, e_hi = parts["lo"], parts["hi"]
        self.slot_a = np.where(posc, e_lo, e_hi)
        self.slot_b = np.where(posc, e_hi, e_lo)
        self.groups = ClusterGroups(np.where(posc, emb, emb_swap),
                                    np.where(posc, emb_swap, emb), self.s0, self.s1)


# --------------------------------------------------------------------------- #
# MCGrad fits
# --------------------------------------------------------------------------- #

FORBIDDEN_TOKENS = {
    "pos", "position", "order", "shown", "first", "second", "swap", "longer",
    "shorter", "verbosity", "sign", "tie", "family", "fam", "own", "self",
    "model", "judge", "pick", "picked", "chosen", "choice", "winner", "gold",
    "label", "human", "y", "lo", "hi", "cat", "group", "axis", "bias"}
FORBIDDEN_SUBSTRINGS = ("longer", "shorter", "family", "model", "pick",
                        "chosen", "gold", "label", "position", "winner")


def check_group_agnostic(df: pd.DataFrame, num_cols, cat_cols) -> None:
    """Raise if CaLLM is handed anything that names or encodes a group: no
    categorical at all, no column named after a group concept, no numeric
    column with two or fewer values (an indicator under a harmless name)."""
    bad = [f"{c!r} (categorical)" for c in (cat_cols or [])]
    sample = df.iloc[:5000]
    for c in num_cols or []:
        toks = set(t for t in re.split(r"[^a-z0-9]+", c.lower()) if t)
        hit = (toks & FORBIDDEN_TOKENS) or {s for s in FORBIDDEN_SUBSTRINGS
                                            if s in c.lower()}
        if hit:
            bad.append(f"{c!r} (name: {sorted(hit)})")
        elif not pd.api.types.is_numeric_dtype(sample[c]):
            bad.append(f"{c!r} (non-numeric)")
        elif sample[c].dropna().nunique() <= 2:
            bad.append(f"{c!r} (indicator)")
    if bad:
        raise ValueError("forbidden calibrator feature(s): " + "; ".join(bad))


def make_mcgrad(overrides: Optional[dict] = None):
    from mcgrad import methods
    return methods.MCGrad(random_state=0, early_stopping=True,
                          early_stopping_use_crossvalidation=True, n_folds=5,
                          save_training_performance=True,
                          lightgbm_params={**MCGRAD_LGB_PARAMS, **(overrides or {})})


def library_default_lgb() -> dict:
    """The config ``mcgrad.tuning`` evaluates as its first trial: MCGrad's
    default for every searched parameter (LightGBM's where MCGrad has none)."""
    from mcgrad import methods, tuning
    full = methods.MCGrad.DEFAULT_HYPERPARAMS["lightgbm_params"]
    return {c.name: full[c.name] if c.name in full
            else tuning.ORIGINAL_LIGHTGBM_PARAMS[c.name]
            for c in tuning.default_parameter_configurations}


def contrast_frames(slot_a, slot_b, cal, te, k: int, want_sym: bool):
    """PCA_k(e_A - e_B), the PCA fit on the calibration rows' differences in
    both orders (so z(B, A) = -z(A, B)). Rows outside cal / te are NaN."""
    from sklearn.decomposition import PCA
    idx, n = np.union1d(cal, te), len(slot_a)
    d_cal = slot_a[cal] - slot_b[cal]
    k = max(1, min(k, 2 * len(cal) - 1, d_cal.shape[1]))
    pca = PCA(n_components=k, random_state=0).fit(
        np.concatenate([d_cal, -d_cal], axis=0))
    cols = [f"c_diff_{i}" for i in range(pca.n_components_)]

    def _frame(z):
        full = np.full((n, len(cols)), np.nan, dtype=np.float32)
        full[idx] = z
        return pd.DataFrame(full, columns=cols)

    shown = _frame(pca.transform(slot_a[idx] - slot_b[idx]))
    rev = _frame(pca.transform(slot_b[idx] - slot_a[idx])) if want_sym else shown
    return shown, rev, cols


def _mcgrad_preds(feat: Features, method: str, cal, te, want_sym: bool,
                  cfg: Optional[dict], tune: Optional[dict]):
    """One CaLLM / CaLLM-C fit on ``cal``; predictions for ``te`` in the shown
    order and (``want_sym``) averaged with the reversed pass. ``cfg`` = the
    adopted LightGBM params plus the PCA width ``n_pca``. With ``tune``
    (``dict(n_trials, record)``) the fit is a ``mcgrad.tuning`` search: every
    trial is fit on ``cal`` and scored (log loss) on ``te``, the validation
    rows; the winner, refit on ``cal``, predicts ``te`` and is appended to
    ``record``."""
    cfg = dict(cfg or {})
    width = int(cfg.pop("n_pca", 0) or 0) or CONTRAST_DEFAULT_WIDTH
    shown, rev, num_cols = contrast_frames(feat.slot_a, feat.slot_b, cal, te,
                                           width, want_sym)
    cat_cols = None
    if method == CALLM_C:
        cat_cols = feat.cat_cols
        shown = shown.copy()
        for c in cat_cols:
            shown[c] = feat.cat[c].values
        if want_sym:
            rev = rev.copy()
            for c in cat_cols:
                rev[c] = feat.cat_rev[c].values
        else:
            rev = shown
    s0, s1 = feat.s0, feat.s1
    df_cal = shown.iloc[cal].copy()
    df_cal["prediction"] = np.clip(s0[cal], EPS, 1 - EPS)
    df_cal["label"] = feat.y[cal].astype(int)
    cols = dict(numerical_feature_column_names=num_cols,
                categorical_feature_column_names=cat_cols)
    if method == CALLM:
        check_group_agnostic(df_cal, num_cols, cat_cols)
    mcg = make_mcgrad(cfg or None)
    if tune is None:
        mcg.fit(df_train=df_cal, prediction_column_name="prediction",
                label_column_name="label", **cols)
    else:
        from mcgrad import tuning
        # our question-grouped validation rows, not the library's row-level split
        df_val = shown.iloc[te].copy()
        df_val["prediction"] = np.clip(s0[te], EPS, 1 - EPS)
        df_val["label"] = feat.y[te].astype(int)
        mcg, trials = tuning.tune_mcgrad_params(
            model=mcg, df_train=df_cal, df_val=df_val,
            prediction_column_name="prediction", label_column_name="label",
            n_trials=int(tune["n_trials"]), **cols)
        searched = [c.name for c in tuning.default_parameter_configurations]
        params = dict(getattr(mcg, "lightgbm_params", {}) or {})
        tune["record"].append(dict(
            params={k: params[k] for k in searched if k in params},
            trials=trials, n_rows=len(df_cal),
            n_features=len(num_cols) + len(cat_cols or [])))

    def _pred(df_src, base):
        df = df_src.iloc[te].copy()
        df["prediction"] = np.clip(base[te], EPS, 1 - EPS)
        return np.clip(np.asarray(mcg.predict(
            df=df, prediction_column_name="prediction", **cols)), EPS, 1 - EPS)

    single = _pred(shown, s0)
    sym = (np.clip(0.5 * (single + (1.0 - _pred(rev, s1))), EPS, 1 - EPS)
           if want_sym else None)
    return single, sym


def _baseline_preds(feat: Features, cal, te, want_sym: bool, methods,
                    baseline_params: Optional[dict]) -> dict:
    """The refit baselines, shown frame. The order-symmetrised score averages
    in the reversed pass, whose prediction is complemented first."""
    s0, s1, y = feat.s0, feat.s1, feat.y
    params = lambda m: {**BASELINE_DEFAULTS.get(m, {}),
                        **((baseline_params or {}).get(m) or {})}
    clip = lambda p: np.clip(p, EPS, 1 - EPS)
    sym = lambda single, rev: clip(0.5 * (single + (1.0 - rev))) if want_sym else None
    preds = {}
    for m, cls in POSTHOC.items():
        if m in methods:
            g = cls(**params(m)).fit(s0[cal], y[cal].astype(int))
            single = clip(g.predict(s0[te]))
            preds[m] = (single, sym(single, clip(g.predict(s1[te]))))
    if "calibraeval" in methods:
        # g maps P(judge emits "A"); g(X0) = debiased P(LO wins)
        ce = CalibraEval(CalibraEvalConfig(**params("calibraeval")))
        ce.fit(feat.ce_s0[cal], feat.ce_s2[cal])
        pos = feat.pos[te]
        g0, g2 = ce.calibrate(feat.ce_s0[te]), ce.calibrate(feat.ce_s2[te])
        single = clip(np.where(pos, g0, 1.0 - g0))
        preds["calibraeval"] = (single, sym(single, clip(np.where(pos, g2, 1.0 - g2))))
    if "length_control" in methods:
        single, rev = length_control(feat.dlen, s0, s1, y, feat.model_a,
                                     feat.model_b, cal, te,
                                     **params("length_control"))
        preds["length_control"] = (single, sym(single, rev))
    grouped = [m for m in GROUPED if m in methods]
    if grouped:
        gm_cal, gm_te, gm_rev = feat.groups(cal, te)
        for m in grouped:
            if not gm_cal:
                single = clip(s0[te])
                preds[m] = (single, sym(single, clip(s1[te])))
                continue
            model = GROUPED[m](**params(m))
            model.fit(clip(s0[cal]), y[cal].astype(int), gm_cal)
            single = clip(model.predict(clip(s0[te]), gm_te))
            preds[m] = (single, sym(single, clip(model.predict(clip(s1[te]), gm_rev))))
    return preds


def predict_split(feat: Features, cal, te, methods, want_sym: bool,
                  mcgrad_cfg: Optional[dict] = None,
                  baseline_params: Optional[dict] = None,
                  tune: Optional[dict] = None) -> dict:
    """Fit ``methods`` on ``cal`` and predict ``te``:
    ``{method: (P(LO better), order-symmetrised or None)}``."""
    preds = _baseline_preds(feat, cal, te, want_sym, methods, baseline_params)
    for m in MCGRAD_METHODS:
        if m in methods:
            preds[m] = _mcgrad_preds(feat, m, cal, te, want_sym,
                                     (mcgrad_cfg or {}).get(m), tune)
    pos = feat.pos[te]
    to_lo = lambda p: None if p is None else np.where(pos, p, 1.0 - p)
    return {m: (to_lo(a), to_lo(b)) for m, (a, b) in preds.items()}


# --------------------------------------------------------------------------- #
# Out-of-fold predictions (question-grouped k-fold), cached
# --------------------------------------------------------------------------- #

def build_oof(dataset: str, judge: str, embedder: str, seed: int = 0,
              n_splits: int = N_FOLDS, refit: bool = False) -> dict:
    """Every method's out-of-fold P(LO better): ``single`` (shown order) and
    ``sym`` (order-symmetrised). Fold k is fit on the other folds with fold k's
    own tuned configs. The block's baselines are ``refit_baselines``. Cached
    in ``results/oof_cache`` per (block, methods, configs)."""
    panel = data.build_panel(dataset, judge, seed)
    fold = outer_folds(panel, seed, n_splits)
    baselines = refit_baselines(judge, embedder)
    fitted = MCGRAD_METHODS + baselines
    tuned = [load_tuned(dataset, judge, embedder, k) for k in range(n_splits)]
    untuned = [f"fold{k}:{m}" for k, (lgb, bl) in enumerate(tuned)
               for m, got in [*[(m, lgb) for m in MCGRAD_METHODS],
                              *[(m, bl) for m in tunable_baselines(judge, embedder)]]
               if not (got or {}).get(m)]
    if untuned:
        print(f"[oof] WARNING {out_key(dataset, judge, embedder)}: no tuned "
              f"config, defaults used for {', '.join(untuned)}")
    sig = hashlib.md5(json.dumps(dict(
        dataset=dataset, judge=judge, embedder=embedder, seed=seed,
        n_splits=n_splits, methods=fitted, tuned=tuned),
        sort_keys=True).encode()).hexdigest()[:16]
    path = os.path.join(RESULTS_DIR, "oof_cache",
                        f"{out_key(dataset, judge, embedder)}__{sig}.npz")
    single: Dict[str, np.ndarray] = {}
    sym: Dict[str, np.ndarray] = {}
    if os.path.exists(path) and not refit:
        z = np.load(path)
        if np.array_equal(z["fold"], fold):
            print(f"[oof] reusing {path}")
            for m in fitted:
                single[m], sym[m] = z[f"single__{m}"], z[f"sym__{m}"]
    if not single:
        print(f"[oof] fitting {len(panel):,} pairs, {n_splits} question-grouped "
              f"folds -> {path}")
        feat = Features(dataset, panel, judge, embedder)
        for m in fitted:
            single[m], sym[m] = np.full(len(panel), np.nan), np.full(len(panel), np.nan)
        for k in range(n_splits):
            te, cal = np.where(fold == k)[0], np.where(fold != k)[0]
            lgb, bl = tuned[k]
            for batch in (baselines, [CALLM], [CALLM_C]):
                preds = predict_split(feat, cal, te, batch, want_sym=True,
                                      mcgrad_cfg=lgb, baseline_params=bl)
                for m, (a, b) in preds.items():
                    single[m][te], sym[m][te] = a, b
            print(f"    [fold {k + 1}/{n_splits}] done", flush=True)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(f"{path}.tmp.npz", fold=fold,
                            **{f"single__{m}": v for m, v in single.items()},
                            **{f"sym__{m}": v for m, v in sym.items()})
        os.replace(f"{path}.tmp.npz", path)
    if panel["p_portia_lo"].notna().any():       # fresh-inference, order-invariant
        single["portia"] = sym["portia"] = panel["p_portia_lo"].values.astype(float)
    return dict(panel=panel, fold=fold, single=single, sym=sym,
                parts=embed.encode_parts(dataset, panel, embedder))


def present_methods(scores: Dict[str, np.ndarray]) -> List[str]:
    return [m for m in METHODS if m in ("verbalized", "bpe") or m in scores]


def shown_y(panel: pd.DataFrame) -> np.ndarray:
    """Y = 1[R_A human-preferred]."""
    pos = panel["pos_a_is_lo"].fillna(False).values.astype(bool)
    y_lo = panel["y_lo"].values.astype(int)
    return np.where(pos, y_lo, 1 - y_lo).astype(int)


def method_score(panel: pd.DataFrame, scores: Dict[str, np.ndarray],
                 method: str) -> np.ndarray:
    """A method's P(R_A better)."""
    if method == "verbalized":
        f = np.clip(panel["p_lo_orig"].values.astype(float), EPS, 1 - EPS)
    elif method == "bpe":
        f = np.clip(panel["p_bpe_lo"].values.astype(float), EPS, 1 - EPS)
    else:
        f = scores[method]
    pos = panel["pos_a_is_lo"].fillna(False).values.astype(bool)
    return np.where(pos, f, 1.0 - f)


def seed_everything(seed: int) -> None:
    """Seed the global RNGs: ``mcgrad.tuning`` builds its Ax search unseeded."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
