#!/bin/bash
# Reproduces every result of the paper. Stages (STAGES, default all, in order):
#   infer  judge inference (GPU)             -> cache/{dataset}_{judge}_predictions.csv
#   embed  response embeddings               -> $CALLM_EMB_CACHE (cache/embeddings)
#   tune   nested hyperparameter search: one config per outer fold, chosen on
#          an 80/20 question-grouped split of the other four folds
#                                            -> results/tuned_{lgb,baselines}.json
#          (USE_PROVIDED_CONFIGS=1: copy the paper's configs/ instead)
#   score  5-fold question-grouped CV        -> results/Q1_mce.csv, results/Q2_perf.csv,
#          + the family group at floor 20 on RewardBench / MTBench
#                                            -> results/Q1_family_lowfloor.csv
#   diag   group decodability (App. E), every block
#                                            -> results/diagnostics/group_decodability.csv
#
# Blocks "judge:embedder": muse:muse-internal (main paper; also App. C, the
#   calibration / multicalibration baselines, fit on this block only),
#   qwen:Qwen/Qwen3-Embedding-0.6B (judge ablation, App. A),
#   muse:Qwen/Qwen3-Embedding-0.6B (embedder ablation, App. B).
#
# Examples:
#   ./run_all.sh
#   USE_PROVIDED_CONFIGS=1 STAGES="tune score diag" ./run_all.sh
#   DATASETS=mtbench BLOCKS=muse:muse-internal ./run_all.sh
#   DRY_RUN=1 ./run_all.sh                      # print the commands only
#
# Other overrides: PY, DEVICE (cuda), N_TRIALS (40 Ax trials per PCA width and
#   fold), N_CONFIGS (40 Optuna trials per baseline and fold), TUNE_SEED (0).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PY=${PY:-python}
STAGES=${STAGES:-"infer embed tune score diag"}
DATASETS=${DATASETS:-pku_saferlhf,arena_100k,ppe_human,rewardbench,mtbench}
BLOCKS=${BLOCKS:-muse:muse-internal,muse:Qwen/Qwen3-Embedding-0.6B,qwen:Qwen/Qwen3-Embedding-0.6B}
DEVICE=${DEVICE:-cuda}
N_TRIALS=${N_TRIALS:-40}
N_CONFIGS=${N_CONFIGS:-40}
TUNE_SEED=${TUNE_SEED:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}

run() { if [ -n "${DRY_RUN:-}" ]; then echo "  $*"; else "$@" || echo "!! FAILED: $*"; fi; }
has() { [[ " $STAGES " == *" $1 "* ]]; }
JUDGES=$(for b in ${BLOCKS//,/ }; do echo "${b%%:*}"; done | sort -u)

if has infer; then
  for J in $JUDGES; do
    for d in ${DATASETS//,/ }; do
      run $PY judge.py --dataset "$d" --judge "$J" --device "$DEVICE"
      run $PY judge.py --dataset "$d" --judge "$J" --device "$DEVICE" --portia
    done
  done
fi

if has embed; then
  for b in ${BLOCKS//,/ }; do
    run $PY embed.py --judge "${b%%:*}" --embedder "${b#*:}" --datasets ${DATASETS//,/ }
  done
fi

if has tune; then
  if [ -n "${USE_PROVIDED_CONFIGS:-}" ]; then
    run mkdir -p results
    run cp configs/tuned_lgb.json configs/tuned_baselines.json results/
  else
    for b in ${BLOCKS//,/ }; do
      J=${b%%:*}; E=${b#*:}
      for d in ${DATASETS//,/ }; do
        # the block's tunable baselines (histogram, CalibraEval, LenControl and
        # IGLB on the main block; CalibraEval and LenControl on the ablations)
        run $PY tune.py baselines --dataset "$d" --judge "$J" --embedder "$E" \
            --n_configs "$N_CONFIGS"
        for M in mcgrad_gen_cdiff_nc mcgrad_cdiff_bias_nc; do     # CaLLM, CaLLM-C
          run $PY tune.py mcgrad --dataset "$d" --judge "$J" --embedder "$E" \
              --method "$M" --n_trials "$N_TRIALS" --seed "$TUNE_SEED"
        done
      done
    done
    for J in $JUDGES; do
      for d in ${DATASETS//,/ }; do
        run $PY tune.py calibraeval --dataset "$d" --judge "$J"
      done
    done
  fi
fi

if has score; then
  for b in ${BLOCKS//,/ }; do
    J=${b%%:*}; E=${b#*:}
    for d in ${DATASETS//,/ }; do
      run $PY evaluate.py bias --dataset "$d" --judge "$J" --embedder "$E"
      run $PY evaluate.py perf --dataset "$d" --judge "$J" --embedder "$E"
    done
    # the family group at floor 20 where its sides are too thin for 30
    LF=$(for x in rewardbench mtbench; do
           [[ ",$DATASETS," == *",$x,"* ]] && echo "$x"; done | paste -sd, -)
    [ -n "$LF" ] && run $PY evaluate.py family_lowfloor --datasets "$LF" \
        --judge "$J" --embedder "$E" --floor 20
  done
  for E in $(for b in ${BLOCKS//,/ }; do echo "${b#*:}"; done | sort -u); do
    run $PY evaluate.py summary --embedder "$E"
  done
fi

if has diag; then
  for b in ${BLOCKS//,/ }; do
    run $PY evaluate.py decodability --judge "${b%%:*}" --embedder "${b#*:}" \
        --datasets "$DATASETS"
  done
fi
echo "=== done -> results/ ==="
