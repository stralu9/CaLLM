"""Judges, datasets and the per-pair panel every experiment starts from.

Each loader returns one row per (question, unordered response pair) in the
canonical orientation: ``model_lo`` is the alphabetically first model and
``human_label`` is +1 when it was preferred, -1 when the other one was, 0 on a
tie (ties are dropped from the panel).

``build_panel`` joins a dataset with its judge predictions
(``cache/{dataset}_{judge}_predictions.csv``, written by ``judge.py``):
one row per decisive pair with the judge's P(LO better) in the shown order
(``p_lo_orig``) and in the swapped order (``p_lo_flip``), the swap-average
(``p_bpe_lo``), CalibraEval's two arrangements in the judge's own token frame
(``ce_s0`` / ``ce_s2``) and the PORTIA verdict (``p_portia_lo``).
"""
from __future__ import annotations

import glob
import hashlib
import os
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "cache")
EPS = 1e-2

# judge tag -> Hugging Face checkpoint
JUDGES = {"muse": "meta-models/Muse-Glimmer-30B",
          "qwen": "Qwen/Qwen2.5-7B-Instruct"}
DEFAULT_JUDGE = "muse"
# The loader family a judge belongs to (self-preference group): every loader
# labels Llama responders "llama", and Muse is a Llama-family model.
JUDGE_FAMILY = {"muse": "llama", "qwen": "qwen"}
DATASETS = ("arena_100k", "ppe_human", "pku_saferlhf", "rewardbench", "mtbench")


def judge_family(judge: str) -> str:
    return JUDGE_FAMILY.get(judge, judge.lower())


def pred_csv(dataset: str, judge: str) -> str:
    return os.path.join(CACHE_DIR, f"{dataset}_{judge}_predictions.csv")


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #

def _family(model: str, prefixes) -> str:
    """Model family: the first matching ``(prefix, family)`` wins, else the
    part of the name before the first '-'."""
    m = (model or "").lower()
    for prefix, fam in prefixes:
        if m.startswith(prefix):
            return fam
    return m.split("-")[0]


_MTBENCH_FAMILIES = [("gpt-", "gpt"), ("claude", "claude"), ("vicuna", "vicuna"),
                     ("llama", "llama"), ("alpaca", "alpaca")]
_REWARDBENCH_FAMILIES = [(p, p) for p in ("gpt", "claude", "vicuna")] + [
    ("llama", "llama"), ("oasst-llama", "llama")] + [
    (p, p) for p in ("alpaca", "mistral", "zephyr", "tulu", "starchat",
                     "starling", "openchat", "dolly", "mpt", "rwkv",
                     "guanaco", "palm")] + [
    ("oasst", "oasst"), ("openassistant", "oasst")]
_ARENA_FAMILIES = [("gpt", "gpt"), ("claude", "claude"), ("gemini", "gemini"),
                   ("gemma", "gemini"), ("llama", "llama"),
                   ("oasst-llama", "llama"), ("mistral", "mistral"),
                   ("mixtral", "mixtral"), ("qwen", "qwen"),
                   ("deepseek", "deepseek"), ("yi-", "yi"), ("vicuna", "vicuna"),
                   ("alpaca", "alpaca"), ("chatglm", "chatglm"),
                   ("koala", "koala"), ("dolly", "dolly"), ("mpt", "mpt"),
                   ("rwkv", "rwkv"), ("phi-", "phi"), ("command", "cohere")]
_PPE_FAMILIES = [("gpt", "gpt"), ("o1", "gpt"), ("o3", "gpt"),
                 ("chatgpt", "gpt"), ("claude", "claude"), ("gemini", "gemini"),
                 ("gemma", "gemini"), ("llama", "llama"),
                 ("meta-llama", "llama"), ("mistral", "mistral"),
                 ("mixtral", "mistral"), ("qwen", "qwen"),
                 ("deepseek", "deepseek"), ("yi-", "yi"), ("command", "cohere"),
                 ("cohere", "cohere"), ("phi", "phi")]


def _row(qid, lo, hi, text_lo, text_hi, prompt, label, families, **extra):
    return dict(qid=str(qid), model_lo=lo, model_hi=hi,
                human_label=int(label),
                family_lo=_family(lo, families), family_hi=_family(hi, families),
                len_lo=len(text_lo or ""), len_hi=len(text_hi or ""),
                prompt=prompt, text_lo=text_lo, text_hi=text_hi, **extra)


def _messages(conv, role):
    try:
        return [m.get("content", "") or "" for m in conv
                if isinstance(m, dict) and m.get("role") == role]
    except TypeError:
        return []


def _hf_dataset(name: str, split: str) -> pd.DataFrame:
    from datasets import load_dataset
    return load_dataset(name, split=split).to_pandas()


def load_arena_100k() -> pd.DataFrame:
    """``lmarena-ai/arena-human-preference-100k``, ties dropped."""
    df = _hf_dataset("lmarena-ai/arena-human-preference-100k", "train")
    df = df[df["winner"].isin(["model_a", "model_b"])]
    rows = []
    for r in df.itertuples(index=False):
        a, b = r.model_a, r.model_b
        lo, hi = sorted([a, b])
        ta = (_messages(r.conversation_a, "assistant") or [""])[-1]
        tb = (_messages(r.conversation_b, "assistant") or [""])[-1]
        winner = a if r.winner == "model_a" else b
        rows.append(_row(r.question_id, lo, hi, *((ta, tb) if a == lo else (tb, ta)),
                         (_messages(r.conversation_a, "user") or [""])[0],
                         1 if winner == lo else -1, _ARENA_FAMILIES,
                         turn=int(r.turn)))
    return pd.DataFrame(rows)


def load_ppe_human() -> pd.DataFrame:
    """``lmarena-ai/PPE-Human-Preference-V1`` (test split), ties dropped."""
    df = _hf_dataset("lmarena-ai/PPE-Human-Preference-V1", "test")
    rows = []
    for r in df.itertuples(index=False):
        a, b = str(r.model_a), str(r.model_b)
        lo, hi = sorted([a, b])
        w = str(r.winner).strip().lower()
        if w not in ("model_a", "model_b"):
            continue
        winner = a if w == "model_a" else b
        ta, tb = r.response_1 or "", r.response_2 or ""
        rows.append(_row(r.question_id, lo, hi, *((ta, tb) if a == lo else (tb, ta)),
                         str(r.prompt), 1 if winner == lo else -1,
                         _PPE_FAMILIES))
    return pd.DataFrame(rows)


def pku_qid(prompt: str, r0: str, r1: str) -> str:
    return hashlib.md5(f"{prompt}\x00{r0}\x00{r1}".encode("utf-8")).hexdigest()[:16]


def load_pku_saferlhf() -> pd.DataFrame:
    """``PKU-Alignment/PKU-SafeRLHF``: label = the BETTER (helpfulness)
    response. There is no model identity, so the two responses are the
    placeholders ``resp0`` / ``resp1`` and the family group is empty."""
    df = _hf_dataset("PKU-Alignment/PKU-SafeRLHF", "train")
    rows = []
    for r in df.itertuples(index=False):
        if r.better_response_id not in (0, 1):
            continue
        prompt, r0, r1 = (str(r.prompt or ""), str(r.response_0 or ""),
                          str(r.response_1 or ""))
        rows.append(dict(qid=pku_qid(prompt, r0, r1), model_lo="resp0",
                         model_hi="resp1",
                         human_label=1 if int(r.better_response_id) == 0 else -1,
                         family_lo="resp0", family_hi="resp1",
                         len_lo=len(r0), len_hi=len(r1),
                         prompt=prompt, text_lo=r0, text_hi=r1))
    return pd.DataFrame(rows)


def load_rewardbench() -> pd.DataFrame:
    """``allenai/reward-bench`` (filtered split); ``chosen`` is the winner."""
    df = _hf_dataset("allenai/reward-bench", "filtered")
    rows = []
    for r in df.itertuples(index=False):
        chosen, rejected = str(r.chosen_model), str(r.rejected_model)
        lo, hi = sorted([chosen, rejected])
        tc, tr = r.chosen or "", r.rejected or ""
        rows.append(_row(r.id, lo, hi, *((tc, tr) if chosen == lo else (tr, tc)),
                         r.prompt or "", 1 if chosen == lo else -1,
                         _REWARDBENCH_FAMILIES))
    return pd.DataFrame(rows)


def mtbench_parquet() -> str:
    """The ``lmsys/mt_bench_human_judgments`` parquet, fetched on first use
    into ``cache/.hfcache``."""
    pattern = os.path.join(CACHE_DIR, ".hfcache",
                           "datasets--lmsys--mt_bench_human_judgments",
                           "snapshots", "*", "data", "human-*.parquet")
    if not glob.glob(pattern):
        try:
            from huggingface_hub import snapshot_download
            snapshot_download("lmsys/mt_bench_human_judgments",
                              repo_type="dataset",
                              cache_dir=os.path.join(CACHE_DIR, ".hfcache"))
        except Exception as e:
            print(f"MT-Bench download failed: {type(e).__name__}: {e}")
    found = sorted(glob.glob(pattern))
    if not found:
        raise SystemExit(f"MT-Bench parquet not found under {CACHE_DIR}/.hfcache")
    return found[0]


def load_mtbench() -> pd.DataFrame:
    """MT-Bench human votes on turn 1, one row per (question, pair) with the
    majority vote (0 = tie); ``nb_lo`` / ``nb_hi`` count the votes for each
    side."""
    df = pd.read_parquet(mtbench_parquet())
    votes = defaultdict(lambda: [0, 0])
    meta = {}
    for r in df.itertuples(index=False):
        if int(r.turn) != 1:
            continue
        a, b = str(r.model_a), str(r.model_b)
        lo, hi = sorted([a, b])
        key = (int(r.question_id), lo, hi)
        w = str(r.winner)
        raw = 1 if w == "model_a" else (-1 if w == "model_b" else 0)
        canon = raw if a == lo else -raw               # +1 == lo wins
        if canon == 1:
            votes[key][0] += 1
        elif canon == -1:
            votes[key][1] += 1
        else:
            _ = votes[key]
        if key not in meta:
            conv_lo, conv_hi = ((r.conversation_a, r.conversation_b) if a == lo
                                else (r.conversation_b, r.conversation_a))
            first = lambda conv, role: (_messages(conv, role) or [""])[0]
            meta[key] = (first(conv_lo, "user"), first(conv_lo, "assistant"),
                         first(conv_hi, "assistant"))
    rows = []
    for (qid, lo, hi), (nb_lo, nb_hi) in votes.items():
        prompt, t_lo, t_hi = meta[(qid, lo, hi)]
        rows.append(_row(qid, lo, hi, t_lo, t_hi, prompt,
                         np.sign(nb_lo - nb_hi), _MTBENCH_FAMILIES,
                         nb_lo=nb_lo, nb_hi=nb_hi))
    return pd.DataFrame(rows)


LOADERS = {"arena_100k": load_arena_100k, "ppe_human": load_ppe_human,
           "pku_saferlhf": load_pku_saferlhf, "rewardbench": load_rewardbench,
           "mtbench": load_mtbench}




# --------------------------------------------------------------------------- #
# Panel
# --------------------------------------------------------------------------- #

def _judge_signals(preds: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Per pair: P(LO better) in the two presentation orders, CalibraEval's
    arrangements and the seeded coin that picks which order was 'shown'.

    'verbalized' shows LO as A, 'reverse' shows it as B (letter 'B' picks LO).
    CalibraEval works in the judge's token frame s = P(judge emits "A"):
    s0 = P_lo(verbalized), s2 = 1 - P_lo(reverse)."""
    key = ["qid", "modelA", "modelB"]

    def _plo(mode, lo_letter):
        sub = preds[preds["mode"] == mode].drop_duplicates(key).set_index(key)
        picks_lo = sub["letter"].str.upper() == lo_letter
        return pd.Series(np.where(picks_lo, sub["confidence"],
                                  1.0 - sub["confidence"]), index=sub.index)

    df = pd.concat([_plo("verbalized", "A").rename("p_lo_verb"),
                    _plo("reverse", "B").rename("p_lo_rev")],
                   axis=1, join="inner")
    df = df.reset_index()
    pv = np.clip(df["p_lo_verb"].values, EPS, 1 - EPS)
    pr = np.clip(df["p_lo_rev"].values, EPS, 1 - EPS)
    orig_is_verb = np.random.default_rng(seed).random(len(df)) < 0.5
    out = pd.DataFrame(dict(
        qid=df["qid"].values, model_lo=df["modelA"].values,
        model_hi=df["modelB"].values,
        p_lo_orig=np.where(orig_is_verb, pv, pr),   # belief in the shown order
        p_lo_flip=np.where(orig_is_verb, pr, pv),   # belief in the swap
        pos_a_is_lo=orig_is_verb,                   # LO shown as A
        ce_s0=pv, ce_s2=np.clip(1.0 - df["p_lo_rev"].values, EPS, 1 - EPS)))
    out["p_bpe_lo"] = 0.5 * (out["p_lo_orig"] + out["p_lo_flip"])
    return out


def _portia(preds: pd.DataFrame) -> pd.DataFrame:
    """PORTIA's order-invariant P(LO), parsed from the ``raw`` column of the
    ``portia`` rows (fallback: the stored letter / confidence)."""
    df = preds[preds["mode"] == "portia"]
    exact = df["raw"].astype(str).str.extract(r"p_portia_lo=([-+0-9.eE]+)",
                                              expand=False).astype(float)
    verdict = np.where(df["letter"].str.upper() == "A", df["confidence"],
                       1.0 - df["confidence"])
    out = df[["qid", "modelA", "modelB"]].assign(
        p_portia_lo=exact.fillna(pd.Series(verdict, index=df.index)))
    return (out.drop_duplicates(["qid", "modelA", "modelB"], keep="last")
            .rename(columns={"modelA": "model_lo", "modelB": "model_hi"}))


def build_panel(dataset: str, judge: str = DEFAULT_JUDGE,
                seed: int = 0) -> pd.DataFrame:
    """One row per decisive pair: judge signals + texts, lengths, families,
    ``y_lo`` = 1[LO human-preferred] and the joint text ``[Q] [A] LO [B] HI``."""
    path = pred_csv(dataset, judge)
    if not os.path.exists(path):
        raise SystemExit(f"missing judge predictions {path}; run judge.py first")
    preds = pd.read_csv(path)
    for c in ("qid", "modelA", "modelB", "mode", "letter"):
        preds[c] = preds[c].astype(str)
    raw_conf = preds["confidence"].astype(float)
    preds["confidence"] = raw_conf.clip(50, 100) / 100.0
    panel = _judge_signals(preds, seed)
    pairs = LOADERS[dataset]()
    if "turn" in pairs.columns:          # ArenaExpl: first-turn battles only
        pairs = pairs[pairs["turn"] == 1]
    pairs = pairs.drop(columns=["turn", "nb_lo", "nb_hi"], errors="ignore")
    panel = panel.merge(pairs, on=["qid", "model_lo", "model_hi"], how="inner")
    panel = panel[panel["human_label"].isin([1, -1])].reset_index(drop=True)
    panel["y_lo"] = (panel["human_label"] == 1).astype(int)
    panel["text"] = (panel["prompt"].fillna("").astype(str)
                     + "\n[A] " + panel["text_lo"].fillna("").astype(str)
                     + "\n[B] " + panel["text_hi"].fillna("").astype(str))
    portia = _portia(preds.assign(confidence=raw_conf / 100.0))
    panel = panel.merge(portia, on=["qid", "model_lo", "model_hi"], how="left")
    return panel
