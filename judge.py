"""Judge inference -> ``cache/{dataset}_{judge}_predictions.csv``.

Every unordered pair (``modelA`` = the alphabetically first model) is judged in
the requested presentations; in the paper's (ID token, answer) notation:

    verbalized   X0 = [(A, ans_a); (B, ans_b)]   canonical
    reverse      X2 = [(A, ans_b); (B, ans_a)]   answers swapped (ID-token swap)

The judge writes a one-sentence reason, then ``answer: A|B`` and
``confidence: 50-100``. Rows already in the file are never regenerated.

``--portia`` runs the PORTIA baseline (Li et al., 2024) instead: both answers
are split into k aligned segments (length alignment, then semantic alignment for
the pairs whose two orders still disagree), interleaved, and judged in both
orders; the two P(LO) are averaged when the orders agree, else 0.5. The result
is stored as ``mode=portia`` rows (``raw`` carries ``p_portia_lo``).

    python judge.py --dataset mtbench --judge muse --device cuda
    python judge.py --dataset mtbench --judge muse --portia
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import re
from typing import List, Optional, Tuple

import data

MODES = ("verbalized", "reverse")
FIELDS = ("seed", "qid", "modelA", "modelB", "nbannotatorsA", "nbannotatorsb",
          "mode", "letter", "confidence", "raw")

INSTRUCTIONS = (
    "You are comparing two assistant responses, A and B, to a user prompt. "
    "Decide which response is better overall, weighing helpfulness, "
    "correctness, clarity, and how well it follows the user's instruction.\n"
    "Respond in EXACTLY this format, one item per line:\n"
    "reason: <one short sentence justifying your choice>\n"
    "answer: <A or B>\n"
    "confidence: <integer 50-100>\n"
    "The confidence is the percent chance that your chosen response is truly "
    "the better one: 50 means a pure coin-flip, 100 means certain. Calibrate "
    "honestly and use the FULL range -- pick the specific two-digit value "
    "that matches your actual certainty (e.g. 57, 63, 78, 94). Do NOT default "
    "to round numbers such as 80, 85, or 90."
)


def build_prompt(prompt: str, option_a: str, option_b: str) -> str:
    """Judge prompt with response A printed before response B."""
    blocks = [f"Response A:\n{(option_a or '').strip()}\n",
              f"Response B:\n{(option_b or '').strip()}\n"]
    return (f"{INSTRUCTIONS}\n\nUser prompt:\n{(prompt or '').strip()}\n\n"
            + "\n".join(blocks))


_ANSWER_RE = re.compile(r"answer\s*[:\-]?\s*\(?\s*([AB])\b", re.IGNORECASE)
_CONF_RE = re.compile(r"confidence\s*[:\-]?\s*(\d{1,3})", re.IGNORECASE)
_FALLBACK_LETTER_RE = re.compile(r"\b(?:response\s+)?([AB])\b")


def parse_verdict(text: str) -> Tuple[str, int]:
    """(letter, confidence in [50, 100]); ('A', 50) where a field is missing."""
    text = text or ""
    m = _ANSWER_RE.search(text) or _FALLBACK_LETTER_RE.search(text)
    mc = _CONF_RE.search(text)
    return ((m.group(1).upper() if m else "A"),
            max(50, min(100, int(mc.group(1)) if mc else 50)))


# --------------------------------------------------------------------------- #
# PORTIA prompt construction
# --------------------------------------------------------------------------- #

# sentence ends (optionally followed by a closing quote / bracket) and newlines
_SENT_BOUNDARY = re.compile(r'(?<=[.!?])["\')\]]*\s+|\n+')


def _boundaries(text: str) -> List[int]:
    return [p for p in (m.end() for m in _SENT_BOUNDARY.finditer(text or ""))
            if 0 < p < len(text)]


def _cut(text: str, cuts) -> List[str]:
    b = [0] + sorted(cuts) + [len(text)]
    return [text[b[i]:b[i + 1]] for i in range(len(b) - 1)]


def length_align(text: str, k: int) -> List[str]:
    """k segments of about equal length, cut at the sentence boundary nearest
    each equidistant target; the whole text when it cannot be split."""
    cands = _boundaries(text)
    if k <= 1 or len(cands) < k - 1:
        return [text]
    chosen: List[int] = []
    for t in [round(i * len(text) / k) for i in range(1, k)]:
        chosen.append(min([c for c in cands if c not in chosen],
                          key=lambda c: abs(c - t)))
    chosen = sorted(set(chosen))
    return [text] if len(chosen) < k - 1 else _cut(text, chosen)


def _overlap(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    return len(sa & sb) / max(len(sa), len(sb)) if sa and sb else 0.0


def semantic_align(t1: str, t2: str, k: int, max_candidates: int = 6):
    """Split points (among the ``max_candidates`` boundaries nearest the
    equidistant targets) maximising the summed token overlap of matching
    segments."""
    def _cands(text):
        c = _boundaries(text)
        if len(c) <= max_candidates:
            return sorted(c)
        targets = [i * len(text) / k for i in range(1, k)]
        return sorted(sorted(c, key=lambda x: min(abs(x - t) for t in targets))
                      [:max_candidates])
    c1, c2 = _cands(t1), _cands(t2)
    if len(c1) < k - 1 or len(c2) < k - 1:
        return length_align(t1, k), length_align(t2, k)
    best_s, best = -1.0, None
    for cut1 in itertools.combinations(c1, k - 1):
        seg1 = _cut(t1, list(cut1))
        for cut2 in itertools.combinations(c2, k - 1):
            seg2 = _cut(t2, list(cut2))
            s = sum(_overlap(seg1[i], seg2[i]) for i in range(k))
            if s > best_s:
                best_s, best = s, (seg1, seg2)
    return best if best is not None else (length_align(t1, k), length_align(t2, k))


def portia_prompt(prompt: str, segs_a: List[str], segs_b: List[str]) -> str:
    lines = [INSTRUCTIONS, "", "User prompt:", (prompt or "").strip(), ""]
    for i in range(min(len(segs_a), len(segs_b))):
        n = i + 1
        lines += [f"[The Start of Assistant A's response part {n}]",
                  (segs_a[i] or "").strip(),
                  f"[The End of Assistant A's response part {n}]",
                  f"[The Start of Assistant B's response part {n}]",
                  (segs_b[i] or "").strip(),
                  f"[The End of Assistant B's response part {n}]", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #

def configure_hf() -> None:
    """HF caches under $HF_HOME; registers the ``muse_glimmer`` architecture
    (a Qwen2-style decoder with a nested text config)."""
    hf_home = os.environ.setdefault("HF_HOME",
                                    os.path.expanduser("~/.cache/huggingface"))
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(hf_home, "datasets"))
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(hf_home, "hub"))
    try:            # symbols some transformers versions still import
        import huggingface_hub
        for attr in ("HfFolder", "Repository"):
            if not hasattr(huggingface_hub, attr):
                setattr(huggingface_hub, attr, type(attr, (), {}))
        for fn in ("create_repo", "list_repo_files", "whoami"):
            if not hasattr(huggingface_hub, fn):
                setattr(huggingface_hub, fn, lambda *a, **kw: None)
    except Exception:
        pass
    try:
        from transformers import (AutoConfig, AutoModelForCausalLM,
                                  PretrainedConfig, Qwen2Config,
                                  Qwen2ForCausalLM)
        if "muse_glimmer" in getattr(AutoConfig, "MAPPING", {}):
            return

        class MuseGlimmerConfig(Qwen2Config):
            model_type = "muse_glimmer"

            def __init__(self, text_config=None, vision_config=None, **kwargs):
                if isinstance(text_config, dict):
                    kwargs.update(text_config)
                    self.text_config = PretrainedConfig(**text_config)
                elif isinstance(text_config, PretrainedConfig):
                    self.text_config = text_config
                    kwargs.update(text_config.to_dict())
                else:
                    self.text_config = text_config
                self.vision_config = (PretrainedConfig(**vision_config)
                                      if isinstance(vision_config, dict)
                                      else vision_config)
                tc = self.text_config
                if "hidden_act" not in kwargs and tc and hasattr(tc, "hidden_activation"):
                    kwargs["hidden_act"] = tc.hidden_activation
                if "rope_theta" not in kwargs:
                    rp = getattr(tc, "rope_parameters", {}) if tc else {}
                    if isinstance(rp, dict):
                        kwargs["rope_theta"] = rp.get("rope_theta", 500000.0)
                kwargs.setdefault("use_sliding_window", False)
                kwargs.setdefault("sliding_window", 4096)
                kwargs.setdefault("max_window_layers", 28)
                super().__init__(**kwargs)
                if tc is not None:
                    for k, v in (tc.to_dict() if hasattr(tc, "to_dict")
                                 else getattr(tc, "__dict__", {})).items():
                        setattr(self, k, v)
                if not hasattr(self, "hidden_act") and tc and hasattr(tc, "hidden_activation"):
                    self.hidden_act = tc.hidden_activation

            def __getattr__(self, name):
                if (name != "text_config" and hasattr(self, "text_config")
                        and hasattr(self.text_config, name)):
                    return getattr(self.text_config, name)
                if name == "use_sliding_window":
                    return False
                raise AttributeError(f"'{type(self).__name__}' object has no "
                                     f"attribute '{name}'")

        class MuseGlimmerForCausalLM(Qwen2ForCausalLM):
            config_class = MuseGlimmerConfig

            def _init_weights(self, module):
                try:
                    if (hasattr(module, "weight") and module.weight is not None
                            and not getattr(module.weight, "is_floating_point",
                                            lambda: True)()):
                        return
                    super()._init_weights(module)
                except Exception:
                    pass

        AutoConfig.register("muse_glimmer", MuseGlimmerConfig)
        try:
            AutoModelForCausalLM.register(MuseGlimmerConfig, MuseGlimmerForCausalLM)
        except Exception:
            pass
    except Exception:
        pass


def _load_tokenizer(model_name: str):
    import transformers
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    if not hasattr(transformers, "TokenizersBackend"):
        transformers.TokenizersBackend = type("TokenizersBackend",
                                              (PreTrainedTokenizerFast,), {})
    for kw in ({}, {"extra_special_tokens": {}}):
        try:
            return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,
                                                 **kw)
        except Exception:
            pass
    return PreTrainedTokenizerFast.from_pretrained(model_name, trust_remote_code=True)


def load_judge(model_name: str, device: str, load_in_8bit: bool = False,
               load_in_4bit: bool = False, attn_implementation: Optional[str] = None,
               dtype=None, gpu_reserve_gib: float = 8.0):
    """(model, tokenizer). ``device="auto"`` (and 8/4-bit) shard with
    device_map="auto", budgeting each GPU at its free memory minus
    ``gpu_reserve_gib``."""
    import torch
    print(f"loading {model_name}...")
    tok = _load_tokenizer(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA not available")
    if dtype is None:
        dtype = (torch.bfloat16 if device != "cpu" and torch.cuda.is_available()
                 else torch.float32)
    kw = {"torch_dtype": dtype, "trust_remote_code": True, "device_map": device}
    if load_in_8bit or load_in_4bit:
        from transformers import BitsAndBytesConfig
        kw["device_map"] = "auto"
        if load_in_8bit:        # int8 matmuls run in fp16
            kw["torch_dtype"] = torch.float16
            kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=dtype)
    if kw["device_map"] == "auto":
        kw["offload_folder"] = "offload"
        if torch.cuda.is_available():
            kw["max_memory"] = {
                i: f"{max(0.0, torch.cuda.mem_get_info(i)[0] / 1024 ** 3 - gpu_reserve_gib):.1f}GiB"
                for i in range(torch.cuda.device_count())}
    if attn_implementation:
        kw["attn_implementation"] = attn_implementation
    from transformers import AutoModelForCausalLM
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **kw)
    except ValueError as e:      # vision-language checkpoints (muse_glimmer)
        if "Unrecognized configuration class" not in str(e):
            raise
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(model_name, **kw)
    model.eval()
    return model, tok


def generate(model, tok, prompts: List[str], batch_size: int,
             max_new_tokens: int, max_length: int, tag: str = "") -> List[str]:
    """Greedy generation, the assistant turn primed with ``reason:``."""
    import torch

    def _chat(text):
        try:
            return tok.apply_chat_template([{"role": "user", "content": text}],
                                           tokenize=False,
                                           add_generation_prompt=True) + "reason:"
        except Exception:
            return "User: " + text + "\nAssistant:reason:"
    out: List[str] = []
    for start in range(0, len(prompts), batch_size):
        enc = tok([_chat(p) for p in prompts[start:start + batch_size]],
                  return_tensors="pt", padding=True, truncation=True,
                  max_length=max_length).to(model.device)
        enc.pop("token_type_ids", None)
        with torch.inference_mode():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
        out += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:],
                                skip_special_tokens=True)
        del enc, gen
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if (start // batch_size) % 10 == 0:
            done = min(start + batch_size, len(prompts))
            print(f"  {tag}{done}/{len(prompts)} ({100. * done / len(prompts):.1f}%)",
                  flush=True)
    return out


# --------------------------------------------------------------------------- #
# Runs
# --------------------------------------------------------------------------- #

def _pairs(dataset: str, limit: Optional[int]):
    df = data.LOADERS[dataset]()
    if "nb_lo" not in df.columns:           # one vote per pair
        df["nb_lo"] = (df["human_label"] == 1).astype(int)
        df["nb_hi"] = (df["human_label"] == -1).astype(int)
    rows = list(df.itertuples(index=False))
    return rows[:limit] if limit is not None else rows


def _done(path: str) -> set:
    """(qid, modelA, modelB, mode) already in the predictions file."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return set()
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        h = next(reader, None) or []
        if not {"qid", "mode", "modelA", "modelB"} <= set(h):
            return set()
        qi, mi, ai, bi = (h.index(c) for c in ("qid", "mode", "modelA", "modelB"))
        return {(r[qi], r[ai], r[bi], r[mi]) for r in reader
                if len(r) > max(qi, mi, ai, bi)}


def run(a, model_name: str, path: str) -> None:
    pairs, done = _pairs(a.dataset, a.limit), _done(path)
    work = []
    for p in pairs:
        for mode in a.modes:
            if (str(p.qid), p.model_lo, p.model_hi, mode) in done:
                continue
            if mode == "reverse":
                prompt = build_prompt(p.prompt, p.text_hi, p.text_lo)
            else:
                prompt = build_prompt(p.prompt, p.text_lo, p.text_hi)
            work.append((p, mode, prompt))
    print(f"{a.dataset}: {len(pairs)} pairs, {len(work)} generations to do "
          f"({', '.join(a.modes)}) -> {path}")
    if not work:
        return
    model, tok = load_judge(model_name, a.device, a.load_in_8bit, a.load_in_4bit,
                            a.attn, gpu_reserve_gib=a.gpu_reserve_gib)
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    if not new:                                  # make sure we append a new line
        with open(path, "rb+") as fh:
            fh.seek(-1, os.SEEK_END)
            if fh.read(1) != b"\n":
                fh.write(b"\n")
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(FIELDS)
        step = a.batch_size
        for s in range(0, len(work), step):
            batch = work[s:s + step]
            raws = generate(model, tok, [pr for _, _, pr in batch], step,
                            a.max_new_tokens, a.max_length)
            for (p, mode, _), raw in zip(batch, raws):
                letter, conf = parse_verdict(raw)
                w.writerow([a.seed, p.qid, p.model_lo, p.model_hi, p.nb_lo,
                            p.nb_hi, mode, letter, conf,
                            raw.replace("\n", " | ").replace("\r", " ")])
            fh.flush()


def _p_lo(letter: str, conf: int, lo_is_a: bool) -> float:
    c = max(50, min(100, int(conf))) / 100.0
    return c if (letter or "A").upper() == ("A" if lo_is_a else "B") else 1.0 - c


def run_portia(a, model_name: str, path: str) -> None:
    done = _done(path)
    pairs = [p for p in _pairs(a.dataset, a.limit)
             if (str(p.qid), p.model_lo, p.model_hi, "portia") not in done]
    print(f"{a.dataset}: PORTIA on {len(pairs)} pairs (k={a.portia_k}) -> {path}")
    if not pairs:
        return
    model, tok = load_judge(model_name, a.device, a.load_in_8bit, a.load_in_4bit,
                            a.attn, gpu_reserve_gib=a.gpu_reserve_gib)

    def _stage(todo, split, tag):
        """Both orders (LO as A, then LO as B) of every pair ->
        [(P(LO), consistent)] via the agreement rule."""
        prompts = []
        for p in todo:
            s_lo, s_hi = split(p)
            prompts += [portia_prompt(p.prompt, s_lo, s_hi),
                        portia_prompt(p.prompt, s_hi, s_lo)]
        v = [parse_verdict(r) for r in generate(
            model, tok, prompts, a.batch_size, a.max_new_tokens, a.max_length,
            tag=f"[portia/{tag}] ")]
        out = []
        for i in range(len(todo)):
            p1, p2 = _p_lo(*v[2 * i], lo_is_a=True), _p_lo(*v[2 * i + 1], lo_is_a=False)
            out.append((0.5 * (p1 + p2), True) if (p1 >= 0.5) == (p2 >= 0.5)
                       else (0.5, False))
        return out

    res = [r + ("length",) for r in _stage(
        pairs, lambda p: (length_align(p.text_lo, a.portia_k),
                          length_align(p.text_hi, a.portia_k)), "length")]
    retry = [i for i, r in enumerate(res) if not r[1]]
    if retry and not a.portia_no_semantic:
        sem = _stage([pairs[i] for i in retry], lambda p: semantic_align(
            p.text_lo, p.text_hi, a.portia_k, a.portia_max_candidates), "semantic")
        for i, r in zip(retry, sem):
            res[i] = r + ("semantic",)
    rows = []
    for p, (val, consistent, stage) in zip(pairs, res):
        pv = max(0.0, min(1.0, float(val)))
        letter, conf = (("A", int(round(pv * 100))) if pv >= 0.5
                        else ("B", int(round((1.0 - pv) * 100))))
        rows.append([a.seed, p.qid, p.model_lo, p.model_hi, p.nb_lo, p.nb_hi,
                     "portia", letter, conf, f"p_portia_lo={val:.6f} "
                     f"stage={stage} consistent={int(consistent)}"])
    # rewrite the file, replacing any earlier portia row of these pairs
    new_keys = {(str(r[1]), str(r[2]), str(r[3])) for r in rows}
    kept = []
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, newline="") as fh:
            reader = csv.reader(fh)
            next(reader, None)
            kept = [r for r in reader if not (len(r) > 6 and r[6] == "portia"
                                              and (r[1], r[2], r[3]) in new_keys)]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(FIELDS)
        w.writerows(kept + rows)
    n_ok = sum(r[1] for r in res)
    print(f"order-consistent: {n_ok}/{len(pairs)}; the rest -> p_portia_lo=0.5")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=data.DATASETS)
    p.add_argument("--judge", default=data.DEFAULT_JUDGE, choices=list(data.JUDGES))
    p.add_argument("--modes", default="verbalized,reverse",
                   help=f"comma-separated subset of {','.join(MODES)}")
    p.add_argument("--portia", action="store_true", help="run PORTIA instead")
    p.add_argument("--portia_k", type=int, default=3)
    p.add_argument("--portia_no_semantic", action="store_true")
    p.add_argument("--portia_max_candidates", type=int, default=6)
    p.add_argument("--device", default="auto", help="cuda / cpu / auto")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=96)
    p.add_argument("--max_length", type=int, default=6144)
    p.add_argument("--load_in_8bit", action="store_true")
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--flash_attn", action="store_true")
    p.add_argument("--gpu_reserve_gib", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=0, help="recorded in the CSV")
    p.add_argument("--limit", type=int, default=None, help="first N pairs only")
    a = p.parse_args()
    a.modes = [m.strip() for m in a.modes.split(",") if m.strip()]
    if set(a.modes) - set(MODES):
        p.error(f"unknown mode(s) {set(a.modes) - set(MODES)}")
    a.attn = "flash_attention_2" if a.flash_attn else None
    configure_hf()
    path = data.pred_csv(a.dataset, a.judge)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    (run_portia if a.portia else run)(a, data.JUDGES[a.judge], path)


if __name__ == "__main__":
    main()
