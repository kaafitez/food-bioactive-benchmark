#!/usr/bin/env bash
# Overnight, memory-safe run of Llama-3.1-70B on the 200-compound set.
# Run this ALONE (close Chrome and other heavy apps first) — the 70B needs
# ~40 GB RAM and will swap/freeze if the machine is busy.
#
# It uses a lean, small-context variant (num_ctx 2048) so KV-cache overhead
# stays small. Our prompts are only a few hundred tokens, so this costs nothing
# in quality. Total: 200 benchmark + 800 ablation calls — expect a few hours
# on CPU/Metal. The raw log is written incrementally, so a crash isn't fatal.
#
# Usage:  bash run_70b_200.sh
set -euo pipefail

echo ">>> Freeing memory: stopping any loaded Ollama model"
ollama stop llama3.1:70b        2>/dev/null || true
ollama stop llama3.1-70b-lean   2>/dev/null || true

echo ">>> Building the lean, small-context 70B variant (reuses downloaded weights, no re-download)"
printf 'FROM llama3.1:70b\nPARAMETER num_ctx 2048\n' > Modelfile.lean
ollama create llama3.1-70b-lean -f Modelfile.lean

echo ">>> Restarting Ollama with memory-saving flags (flash attention + quantized KV cache)"
pkill ollama 2>/dev/null || true
sleep 3
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=q8_0
nohup ollama serve > ollama_serve.log 2>&1 &
sleep 6

echo ">>> Running 70B on the 200-compound set (this is the long part)"
python run_benchmark.py --model llama3.1-70b-lean --mode both --data data_200 --out results_200

echo ">>> Done. Results in ./results_200/  (llama3.1-70b-lean_*.json / .csv / .jsonl)"
