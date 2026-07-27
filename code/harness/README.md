# Local bioactive-food LLM benchmark + grounding ablation (Ollama)

This harness reproduces the manuscript's benchmark and grounding ablation on a
**local open-weights model** via Ollama — no API keys, no cloud, no Claude
credits. Running it on your 32B model provides the open-weights, cross-family
replication that the review needs to move past its single-vendor limitation.

## What it does
- **Benchmark** (condition A): given each of 100 culturally-stratified bioactive
  compounds (name + SMILES + formula), the model predicts four properties —
  bioactivity class (14-way), food source, cultural context, bioavailability.
- **Ablation**: the same task under four grounding conditions —
  - **A** baseline (name + SMILES + formula)
  - **B** + computed structure descriptors + Bemis–Murcko scaffold
  - **C** + a database-sourced botanical/culinary provenance note
  - **D** + both
  The key question: does provenance grounding (C) close the Western–African
  accuracy gap, as it did for the Claude family?
- Grades deterministically (exact-match for the closed-vocab tasks, normalized
  token-overlap for food source) — fully reproducible, no LLM judge.

## Requirements
- [Ollama](https://ollama.com) installed and running.
- Python 3.8+ (standard library only — nothing to `pip install`).

## Run it
```bash
# 1. pull your model (example; use whatever 32B you have)
ollama pull qwen2.5:32b

# 2. make sure Ollama is serving (usually automatic):
#    it listens on http://localhost:11434

# 3. quick smoke test on 5 compounds first:
python run_benchmark.py --model qwen2.5:32b --mode benchmark --limit 5

# 4. full run (both benchmark and 4-condition ablation, 100 compounds):
python run_benchmark.py --model qwen2.5:32b --mode both
```

Options:
- `--model`  Ollama model tag (required), e.g. `qwen2.5:32b`, `llama3.1:70b`.
- `--host`   OpenAI-compatible base URL (default `http://localhost:11434/v1`).
- `--mode`   `benchmark` | `ablation` | `both` (default `both`).
- `--limit`  cap compound count for a smoke test (0 = all 100).

## Runtime
The ablation is 100 compounds × 4 conditions = 400 generations; the benchmark
adds 100 more. On a 32B model on consumer hardware expect roughly 1–3 seconds
per generation, i.e. ~10–25 minutes for the full `both` run. The raw log is
written incrementally, so an interrupted run is not lost.

## Outputs (in ./results/)
- `<model>_summary.json`  — headline numbers (overall accuracy, per-stratum,
  Western–African gap per condition). **This is the main file to send back.**
- `<model>_benchmark_scores.csv`, `<model>_ablation_scores.csv` — per-compound grades.
- `<model>_benchmark_raw.jsonl`, `<model>_ablation_raw.jsonl` — full raw responses.

**Send the three `*_summary.json`, `*_scores.csv`, `*_raw.jsonl` files back** and
they'll be integrated into the manuscript as an open-weights replication — with
the local model's real numbers, whatever they are.

## Notes / honesty
- A 32B model will likely score **lower overall** than frontier models and may
  follow the JSON format less reliably. The harness has a tolerant parser and
  retries, and counts unparseable/blank responses as refusals in the summary.
  That's expected and fine: the paper's claim is about the *cultural gap and
  whether provenance closes it*, not absolute accuracy.
- The provenance notes (condition C) were assembled from structured, non-LLM
  sources (Wikidata / Wikipedia / curated food-source fields), with cultural
  stratum / bioactivity / bioavailability tokens stripped to prevent leakage —
  identical controls to the manuscript.

## Recommended open-weights models (no API cost)

Everything below runs locally through Ollama with **zero API cost and no hidden
fees** — you only pay in disk space and run time. Pick by what your machine can
hold in RAM/VRAM (rule of thumb for 4-bit quantized: model needs ≈ 0.6 GB per
billion parameters, so a 32B ≈ 20 GB, a 70B ≈ 40 GB). All are Apache-2.0 or
similarly permissive except where noted.

**To add vendor diversity (answers the "single-vendor" reviewer flag):** pick
models from *different model families/labs*, not just different sizes.

| Model (Ollama tag) | Params | Lab / family | Approx RAM (q4) | Notes |
|---|---|---|---|---|
| `qwen2.5:32b` | 32B | Alibaba Qwen | ~20 GB | Already run — the current replication |
| `llama3.1:70b` | 70B | Meta Llama | ~40 GB | Strong, widely used; different family from Qwen |
| `llama3.1:8b` | 8B | Meta Llama | ~5 GB | Small tier — pairs with 70b for a Llama scale trend |
| `gemma2:27b` | 27B | Google Gemma (open) | ~17 GB | Google-lineage weights, **no API cost** (unlike Gemini API) |
| `mixtral:8x7b` | 47B MoE | Mistral | ~28 GB | Mixture-of-experts; fast for its quality |
| `mistral-small:24b` | 24B | Mistral | ~15 GB | Lighter Mistral option |
| `phi3:14b` | 14B | Microsoft Phi | ~9 GB | Small but capable; another distinct family |
| `deepseek-r1:32b` | 32B | DeepSeek | ~20 GB | Reasoning-tuned; distinct lineage |

**Two things the manuscript still needs, and which models give them for free:**
1. **Cross-family breadth** — run 2–3 models from *different labs* (e.g.
   `llama3.1:70b`, `gemma2:27b`, `mixtral:8x7b`). If the provenance-closes-the-gap
   dissociation holds across all of them, the single-vendor objection is fully answered.
2. **A within-family scale trend on a second vendor** — run a small + large from
   the *same* family (e.g. `llama3.1:8b` **and** `llama3.1:70b`). If the gap
   narrows-but-doesn't-close with scale on Llama too, the scale claim is no longer
   Claude-only.

Both of these currently cap the acceptance score, and both cost nothing but run time.

## Run several models in one command

```bash
# runs the default list (edit MODELS in sweep.sh to match what you can run):
bash sweep.sh

# or name exactly the models you want:
bash sweep.sh llama3.1:70b llama3.1:8b gemma2:27b
```
`sweep.sh` pulls each model, runs the full benchmark + ablation, and writes
per-model result files into `./results/`. Send the whole `results/` folder back.

## 200-compound set (expanded, for powering per-stratum tests)

The `data_200/` folder holds an **expanded 200-compound set (50 per stratum)** —
the original 100 plus 100 additional real, PubChem-validated bioactives, with
leakage-audited provenance notes (0/200 notes contain a stratum label, class,
region word, or bioavailability tier). Use it to test whether the cultural gap
and the provenance effect are stable when the sample size doubles — and whether
the secondary per-stratum effects (underpowered at n=25) firm up at n=50.

```bash
# one model on the 200-set:
python run_benchmark.py --model qwen2.5:32b --mode both --data data_200 --out results_200

# or sweep several (skips llama3.1:70b by default — too big for 64 GB here):
bash sweep_200.sh qwen2.5:32b llama3.1:8b gemma2:27b mixtral:8x7b
```
Results go to `./results_200/`. **Send back the whole `results_200/` folder.**
Run time is ~2× the 100-set (200 benchmark + 800 ablation calls per model).

### The 70B on the 200-set — run it alone, overnight

`sweep_200.sh` deliberately **excludes** `llama3.1:70b` because it needs ~40 GB
RAM and will freeze a busy 64 GB machine (doubly so at 200 compounds). To include
it, run it on its own when you're away from the laptop, with a dedicated script
that builds the lean variant and sets the memory-saving flags for you:
```bash
# close Chrome and heavy apps first, then:
bash run_70b_200.sh
```
This writes `llama3.1-70b-lean_*` files into `./results_200/` alongside the
other four models. Together that gives all five families on the 200-set, matching
the 100-set coverage.

## Files
- `run_benchmark.py`  — entry point (CLI).
- `sweep.sh`          — run several models on the 100-set in one command.
- `sweep_200.sh`      — same, on the expanded 200-compound set.
- `data_200/`         — expanded 200-compound set + leakage-audited provenance.
- `foodllm_bench.py`  — core: prompts, OpenAI-compatible client, parser, grader.
- `data/compounds.csv`          — 100 compounds + precomputed scaffolds/descriptors.
- `data/provenance_notes.csv`   — database-sourced provenance notes.
- `data/bioactivity_vocab.json` — the 14-class controlled vocabulary.
