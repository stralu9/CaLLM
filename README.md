# CaLLM: code for "A CaLLM Judge Is a Better Judge"

This folder contains the code for every result in the paper, from judge inference to the metric tables.

## Quick start

```bash
conda create -n callm python=3.10 && conda activate callm
pip install -r requirements.txt
export HF_HOME=/path/to/hf_cache        # model weights + datasets (optional)

./run_all.sh                            # everything (GPU needed for inference / embeddings)
USE_PROVIDED_CONFIGS=1 STAGES="tune score diag" ./run_all.sh   # the paper's tuned configs, no search
DATASETS=mtbench BLOCKS=muse:muse-internal ./run_all.sh        # one dataset, main block
DRY_RUN=1 ./run_all.sh                  # print the commands
```

`run_all.sh` runs five stages (`STAGES`): `infer` → `embed` → `tune` → `score` → `diag`. Each stage reads the previous stage's outputs from `cache/` and writes to `cache/` or `results/`. It runs them for every dataset and every `judge:embedder` block:

| block | role in the paper |
|---|---|
| `muse:muse-internal` | main results (Q1–Q3) and the comparison with other calibration / multicalibration methods (App. C) |
| `qwen:Qwen/Qwen3-Embedding-0.6B` | judge ablation (App. A) |
| `muse:Qwen/Qwen3-Embedding-0.6B` | embedder ablation (App. B) |

The calibration / multicalibration baselines of App. C (temperature, Platt, isotonic, histogram, beta, GCUR / LOGR, IGLB) are fit on the main block only. The two ablation blocks report CaLLM, CaLLM-C and the bias-mitigation baselines (f0, BPC, CalibraEval, PORTIA, LenControl).

## Outputs

| file | content |
|---|---|
| `results/Q1_mce.csv` | per-group MCE% / MCEσ on position, verbosity and family, plus the joint MCE over the three (the specified groups of Q1, Q2 and App. A–C); MCE on unspecified embedding segments in the `emb_mce_*` columns. A group is scored in a fold only when both of its directional sides have at least 30 pairs there |
| `results/Q1_family_lowfloor.csv` | the family group alone, re-scored at a floor of 20 pairs on RewardBench and MTBench, whose own-family sides never both reach 30 in a fold. It fills exactly the family cells that are empty in `Q1_mce.csv`; the joint MCE and every other cell keep the floor of 30 |
| `results/Q2_perf.csv` | accuracy / log loss / Brier / ECCE (Q3) |
| `results/diagnostics/group_decodability.csv` | how well each group can be recovered from CaLLM's features, for every block (App. E) |
| `results/summary_{bias,perf}[__embedder].csv` | means across datasets |
| `results/tuned_{lgb,baselines}.json` | adopted hyperparameters, one per outer fold, keyed `dataset[__judge][__embedder]` → method → `fold<k>` |

Every metric is computed inside each of the 5 question-grouped held-out folds and then averaged. The `*_se` / `*_lo` / `*_hi` columns hold the SE and 95% t-interval over the folds.

Hyperparameters are tuned inside each fold (nested). For held-out fold k, the other four folds are split at random, by question, into 80% fit and 20% validation. Every configuration is fit on the 80% and scored by log loss on the 20%, and the best one is refit on all four folds and evaluated on fold k. Each fold therefore has its own configuration, and no held-out fold influences the configuration it is scored with. This applies to CaLLM, CaLLM-C and every tuned baseline alike (histogram, CalibraEval, LenControl and IGLB on the main block; CalibraEval and LenControl on the ablation blocks).

## Files

| file | what it does |
|---|---|
| `run_all.sh` | the whole pipeline |
| `judge.py` | judge inference: verbalized confidence in both presentation orders, PORTIA (`--portia`) → `cache/{dataset}_{judge}_predictions.csv` |
| `data.py` | judge registry, the five dataset loaders, and the per-pair panel (judge signals + texts + labels) |
| `embed.py` | text embeddings (judge hidden states or a sentence embedder), cached in `cache/embeddings/` |
| `calibrators.py` | baselines: temperature / Platt / isotonic / histogram / beta, CalibraEval, GCUR / IGLB with clustering groups, length control |
| `callm.py` | CaLLM itself: bias groups, question-grouped folds, features, MCGrad fits, cached out-of-fold predictions |
| `tune.py` | nested hyperparameter search: one configuration per outer fold (lowest validation log loss) |
| `evaluate.py` | bias and performance metrics, the family low-floor re-score, summaries, decodability diagnostic |
| `configs/` | the tuned per-fold hyperparameters used in the paper (`USE_PROVIDED_CONFIGS=1`) |
| `requirements.txt` | pinned Python 3.10 environment |

## Method names in the code

| paper | code |
|---|---|
| CaLLM | `mcgrad_gen_cdiff_nc` (MCGrad on PCA_k(e_A − e_B), k tuned in {4, 8, 16, 32}) |
| CaLLM-C | `mcgrad_cdiff_bias_nc` (the same features plus the declared bias groups) |
| f0 / BPC / CalibraEval / PORTIA / LenControl | `verbalized` / `bpe` / `calibraeval` / `portia` / `length_control` |
| Temperature / Platt / LOGR / IGLB (App. C) | `temperature` / `platt` / `gcur_logistic` / `iglb` |
| ArenaExpl / PPE / PKU / RewardBench / MTBench | `arena_100k` / `ppe_human` / `pku_saferlhf` / `rewardbench` / `mtbench` |
| Muse-Glimmer-30B / Qwen2.5-7B judge | `--judge muse` / `--judge qwen` |
| judge embeddings / Qwen3-Embedding-0.6B | `--embedder muse-internal` / `--embedder Qwen/Qwen3-Embedding-0.6B` |

## Notes

- Environment variables: `HF_HOME` (Hugging Face cache, default `~/.cache/huggingface`) and `CALLM_EMB_CACHE` (embedding cache, default `cache/embeddings`).
- Judge inference loads `meta-models/Muse-Glimmer-30B` and `Qwen/Qwen2.5-7B-Instruct`. The `muse-internal` embedder also needs the Muse weights, because it pools the judge's own hidden states (8-bit by default, sharded across the visible GPUs). On a multi-GPU machine, pass `DEVICE=auto` to shard the 30B judge.
- `mcgrad.tuning` runs its Ax search unseeded, so `tune.py` seeds the global RNGs first (`TUNE_SEED`, default 0). `USE_PROVIDED_CONFIGS=1` scores with the exact configurations used in the paper.
- The `results/oof_cache/` files save the out-of-fold predictions for each block. `evaluate.py bias`, `perf` and `family_lowfloor` share them, and they are refit whenever the adopted configs change.
