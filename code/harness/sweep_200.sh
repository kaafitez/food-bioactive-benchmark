#!/usr/bin/env bash
# 200-COMPOUND version of the sweep: runs the benchmark + 4-condition ablation
# on the expanded 200-compound set (50 per stratum) in ./data_200/, writing
# results into ./results_200/. Same models, same protocol — only the compound
# set is larger, to power the per-stratum tests.
#
# Usage:
#   bash sweep_200.sh                          # default model list below
#   bash sweep_200.sh qwen2.5:32b gemma2:27b   # only the models you name
#
# NOTE: 200 compounds doubles the run time vs the 100-set (200 benchmark + 800
# ablation calls per model). Comfortable models on 64 GB RAM: anything <= 32B.
# Skip llama3.1:70b here unless you run it alone, overnight (it needs ~40 GB).
set -euo pipefail

MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
  MODELS=( "qwen2.5:32b" "llama3.1:8b" "gemma2:27b" "mixtral:8x7b" )
fi

echo "200-compound run. Models: ${MODELS[*]}"
echo "Each model = 200 benchmark + 800 ablation calls. This takes a while."
echo

for M in "${MODELS[@]}"; do
  echo "=============================================================="
  echo ">>> $M  (200-compound set)"
  ollama pull "$M" || { echo "!! could not pull $M — skipping"; continue; }
  python run_benchmark.py --model "$M" --mode both --data data_200 --out results_200 \
    || { echo "!! run failed for $M — skipping"; continue; }
  echo ">>> Done $M"
  echo
done

echo "=============================================================="
echo "All done. Send back everything in ./results_200/ :"
ls -1 results_200/ 2>/dev/null || true
