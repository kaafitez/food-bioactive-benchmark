#!/usr/bin/env bash
# Run the full benchmark + 4-condition ablation across several local open-weights
# models via Ollama. Zero API cost. Each model writes its own namespaced result
# files into ./results/ (e.g. qwen2.5_32b_summary.json), so nothing clobbers.
#
# Usage:
#   bash sweep.sh                         # runs the default model list below
#   bash sweep.sh llama3.1:70b gemma2:27b # runs only the models you name
#
# Edit MODELS to match what you have pulled (`ollama list` shows installed tags).
set -euo pipefail

MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
  MODELS=( "qwen2.5:32b" "llama3.1:70b" "gemma2:27b" "mixtral:8x7b" )
fi

echo "Models to run: ${MODELS[*]}"
echo "Each model = 100 benchmark + 400 ablation calls. Grab a coffee."
echo

for M in "${MODELS[@]}"; do
  echo "=============================================================="
  echo ">>> Pulling $M (skips if already present)"
  ollama pull "$M" || { echo "!! could not pull $M — skipping"; continue; }
  echo ">>> Running benchmark + ablation on $M"
  python run_benchmark.py --model "$M" --mode both || { echo "!! run failed for $M — skipping"; continue; }
  echo ">>> Done $M"
  echo
done

echo "=============================================================="
echo "All done. Send back everything in ./results/ :"
echo "  *_summary.json  *_scores.csv  *_raw.jsonl"
ls -1 results/ 2>/dev/null || true
