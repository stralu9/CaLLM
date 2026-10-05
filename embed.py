"""Text embeddings, cached per (dataset, embedder).

Two encoders: any sentence-transformers model (``Qwen/Qwen3-Embedding-0.6B``)
and ``muse-internal``, the mean-pooled last hidden state of the Muse judge
itself. Every vector is cached by the SHA1 of its text in
``$CALLM_EMB_CACHE/{dataset}_embeddings_{model}.npz`` (default
``cache/embeddings``), so each text is encoded once.

Run as a script to fill the cache ahead of tuning / scoring:

    python embed.py --embedder muse-internal --judge muse --datasets mtbench
"""
from __future__ import annotations

import argparse
import hashlib
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import data

EMB_CACHE_ROOT = os.environ.get("CALLM_EMB_CACHE",
                                os.path.join(data.HERE, "cache", "embeddings"))
MUSE_INTERNAL = "muse-internal"


def safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "_")


def cache_path(dataset: str, model_name: str) -> str:
    return os.path.join(EMB_CACHE_ROOT, f"{dataset}_embeddings_"
                                        f"{safe_model_name(model_name)}.npz")


def _hash_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _load_cache(path: str, model_name: str) -> Dict[str, np.ndarray]:
    if not os.path.exists(path):
        return {}
    with np.load(path, allow_pickle=False) as z:
        if "model" in z.files and str(z["model"]) != model_name:
            print(f"  emb cache {path} was produced with {str(z['model'])!r}; "
                  "ignoring it")
            return {}
        keys, arr = z["keys"], np.asarray(z["embeddings"], dtype=np.float32)
    # rows whose fp16 token sum overflowed (older muse caches) are re-encoded
    ok = np.ones(len(arr), dtype=bool)
    for s in range(0, len(arr), 8192):
        ok[s:s + 8192] = np.isfinite(arr[s:s + 8192]).all(axis=1)
    if not ok.all():
        print(f"  emb cache {path}: dropping {int((~ok).sum())} non-finite rows")
    return {str(keys[k]): arr[k] for k in np.flatnonzero(ok)}


def _save_cache(path: str, mapping: Dict[str, np.ndarray],
                model_name: str) -> None:
    if not mapping:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    keys = np.array(list(mapping.keys()))
    embs = np.stack([mapping[k] for k in keys]).astype(np.float32)
    with open(f"{path}.tmp", "wb") as f:
        np.savez(f, keys=keys, embeddings=embs, model=np.array(model_name))
    os.replace(f"{path}.tmp", path)


# --------------------------------------------------------------------------- #
# Encoders
# --------------------------------------------------------------------------- #

class MuseInternalEncoder:
    """Mean-pooled last-layer hidden states of the Muse judge, with the
    ``encode`` / ``get_sentence_embedding_dimension`` surface of a
    ``SentenceTransformer``. Loaded 8-bit by default
    (``MUSE_EMBED_LOAD_IN_8BIT=0`` for full precision,
    ``MUSE_EMBED_LOAD_IN_4BIT=1`` for 4-bit), sharded with device_map="auto"."""

    def __init__(self, device: str):
        import torch
        from judge import configure_hf, load_judge
        configure_hf()
        load_4bit = os.environ.get("MUSE_EMBED_LOAD_IN_4BIT") == "1"
        load_8bit = (not load_4bit
                     and os.environ.get("MUSE_EMBED_LOAD_IN_8BIT", "1") != "0")
        load_device = "auto" if device == "cuda" else device
        cpu_dtype = None
        if load_device == "cpu":            # bitsandbytes is CUDA-only
            load_8bit = load_4bit = False
            cpu_dtype = getattr(torch, os.environ.get("MUSE_EMBED_CPU_DTYPE",
                                                      "bfloat16"))
        self.model, self.tok = load_judge(
            data.JUDGES["muse"], load_device, load_in_8bit=load_8bit,
            load_in_4bit=load_4bit, dtype=cpu_dtype)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        # the backbone only: the full-vocab lm_head is not needed
        self._backbone = getattr(self.model, "model", self.model)
        self.max_length = int(os.environ.get("MUSE_EMBED_MAX_LEN", "4096"))
        # cap on padded tokens per forward: bitsandbytes' int8 kernel overflows
        # past 2**31 elements of (tokens x widest layer)
        text_cfg = getattr(self.model.config, "text_config", self.model.config)
        widest = max(int(getattr(text_cfg, "intermediate_size", 0) or 0),
                     int(getattr(text_cfg, "hidden_size", 0) or 0), 1)
        hard_cap = (2**31 - 1) // widest - 2048
        self.max_tokens = max(self.max_length, min(
            hard_cap, int(os.environ.get("MUSE_EMBED_MAX_TOKENS", "16384"))))
        self._dim: Optional[int] = None

    def get_sentence_embedding_dimension(self) -> int:
        if self._dim is None:
            self._dim = int(self.encode(["x"], batch_size=1).shape[1])
        return self._dim

    def encode(self, texts: List[str], batch_size: int = 32,
               **_) -> np.ndarray:
        import torch
        ids = self.tok(list(texts), truncation=True,
                       max_length=self.max_length)["input_ids"]
        order = sorted(range(len(texts)), key=lambda k: len(ids[k]),
                       reverse=True)
        sub_batches: List[List[int]] = []
        for k in order:                  # longest first: cur[0] sets the padding
            cur = sub_batches[-1] if sub_batches else None
            if (cur is not None and len(cur) < max(1, batch_size)
                    and (len(cur) + 1) * max(1, len(ids[cur[0]]))
                    <= self.max_tokens):
                cur.append(k)
            else:
                sub_batches.append([k])
        out, done_order = [], []
        for sub in sub_batches:
            enc = self.tok.pad({"input_ids": [ids[k] for k in sub]},
                               return_tensors="pt").to(self.model.device)
            done_order.extend(sub)
            enc.pop("token_type_ids", None)
            with torch.inference_mode():
                out_bb = self._backbone(**enc, use_cache=False)
                hidden = getattr(out_bb, "last_hidden_state", None)
                if hidden is None:
                    hidden = out_bb[0]
            hidden = hidden.float()          # pool in fp32 (fp16 overflows)
            mask = enc["attention_mask"].unsqueeze(-1).to(device=hidden.device,
                                                           dtype=hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            if not torch.isfinite(pooled).all():
                raise FloatingPointError(
                    "muse-internal: non-finite pooled embedding; "
                    "try MUSE_EMBED_LOAD_IN_8BIT=0")
            out.append(pooled.cpu().numpy())
            del enc, out_bb, hidden, mask, pooled
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        embs = np.concatenate(out, axis=0).astype(np.float32)
        restored = np.empty_like(embs)
        restored[np.asarray(done_order)] = embs
        return restored


_ENCODERS: Dict[Tuple[str, str], object] = {}


def _encoder(model_name: str, device: str, cache_folder: str):
    if (model_name, device) not in _ENCODERS:
        if model_name == MUSE_INTERNAL:
            enc = MuseInternalEncoder(device)
        else:
            from sentence_transformers import SentenceTransformer
            enc = SentenceTransformer(
                model_name, device=device, cache_folder=cache_folder,
                model_kwargs={"torch_dtype": "bfloat16"} if device == "cuda"
                else None)
        _ENCODERS[(model_name, device)] = enc
    return _ENCODERS[(model_name, device)]


def _length_aware_batches(texts: List[str], batch_size: int,
                          budget_elems: int = 1_610_612_736,
                          chars_per_token: int = 3) -> List[List[int]]:
    """Longest-first chunks with ``chunk * seq_len**2 <= budget_elems``, so a
    few very long texts never share a batch."""
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
    batches, i = [], 0
    while i < len(order):
        seq_len = max(1, len(texts[order[i]]) // chars_per_token)
        b = max(1, min(batch_size, budget_elems // (seq_len * seq_len)))
        batches.append(order[i:i + b])
        i += b
    return batches


def _encode_with_backoff(enc, texts: List[str]) -> np.ndarray:
    """Encode as one batch; on CUDA OOM retry as two halves. The retry runs
    outside the ``except`` block so the failed attempt's tensors are freed."""
    import torch
    try:
        return np.asarray(enc.encode(texts, batch_size=len(texts),
                                     show_progress_bar=False,
                                     convert_to_numpy=True), dtype=np.float32)
    except torch.cuda.OutOfMemoryError:
        pass
    torch.cuda.empty_cache()
    if len(texts) == 1:
        raise torch.cuda.OutOfMemoryError(
            f"CUDA OOM encoding a single text ({len(texts[0])} chars)")
    mid = len(texts) // 2
    return np.concatenate([_encode_with_backoff(enc, texts[:mid]),
                           _encode_with_backoff(enc, texts[mid:])], axis=0)


def encode_texts(dataset: str, texts: List[str], model_name: str,
                 batch_size: int = 32) -> np.ndarray:
    """``(len(texts), dim)`` float32 embeddings, from the cache where
    possible; missing texts are encoded and the cache is checkpointed every
    ``EMB_CACHE_FLUSH_EVERY`` chunks."""
    path = cache_path(dataset, model_name)
    cached = _load_cache(path, model_name)
    keys = [_hash_text(t) for t in texts]
    missing = [i for i, k in enumerate(keys) if k not in cached]
    if missing:
        import torch
        from tqdm import tqdm
        cache_folder = os.environ.setdefault(
            "SENTENCE_TRANSFORMERS_HOME",
            os.path.join(EMB_CACHE_ROOT, "sentence_transformers"))
        device = os.environ.get("EMB_DEVICE") or (
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"  encoding {len(missing)}/{len(texts)} texts with {model_name} "
              f"on {device} -> {os.path.basename(path)}")
        enc = _encoder(model_name, device, cache_folder)
        todo = [texts[i] for i in missing]
        budget = (max(1, int(torch.cuda.mem_get_info()[0] * 0.3) // 4)
                  if torch.cuda.is_available() else None)
        chunks = _length_aware_batches(
            todo, batch_size, **({"budget_elems": budget} if budget else {}))
        new = np.empty((len(todo), enc.get_sentence_embedding_dimension()),
                       dtype=np.float32)
        flush_every = int(os.environ.get("EMB_CACHE_FLUSH_EVERY", "300"))
        for c, chunk in enumerate(tqdm(chunks, desc="  encoding"), start=1):
            new[chunk] = _encode_with_backoff(enc, [todo[k] for k in chunk])
            if flush_every and c % flush_every == 0:
                done = [k for ch in chunks[:c] for k in ch]
                cached.update({keys[missing[k]]: new[k] for k in done})
                _save_cache(path, cached, model_name)
        cached.update({keys[i]: new[j] for j, i in enumerate(missing)})
        _save_cache(path, cached, model_name)
        print(f"  wrote {path} ({len(cached)} entries)")
    return np.stack([cached[k] for k in keys]).astype(np.float32)


def joint_texts(panel: pd.DataFrame, swapped: bool = False) -> List[str]:
    """``prompt [A] LO [B] HI`` (the panel's ``text``) or its swap."""
    if not swapped:
        return panel["text"].astype(str).tolist()
    col = lambda c: panel[c].fillna("").astype(str)
    return (col("prompt") + "\n[A] " + col("text_hi")
            + "\n[B] " + col("text_lo")).tolist()


def encode_parts(dataset: str, panel: pd.DataFrame,
                 model_name: str) -> Dict[str, np.ndarray]:
    """Separate embeddings of the prompt and of the LO / HI answers."""
    return {key: encode_texts(dataset, panel[col].fillna("").astype(str).tolist(),
                              model_name)
            for key, col in (("prompt", "prompt"), ("lo", "text_lo"),
                             ("hi", "text_hi"))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=list(data.DATASETS))
    p.add_argument("--embedder", required=True)
    p.add_argument("--judge", default=data.DEFAULT_JUDGE)
    a = p.parse_args()
    for ds in a.datasets:
        panel = data.build_panel(ds, a.judge)
        print(f"=== {ds}: {len(panel)} pairs, embedder {a.embedder} ===")
        encode_texts(ds, joint_texts(panel), a.embedder)
        encode_texts(ds, joint_texts(panel, swapped=True), a.embedder)
        encode_parts(ds, panel, a.embedder)


if __name__ == "__main__":
    main()
