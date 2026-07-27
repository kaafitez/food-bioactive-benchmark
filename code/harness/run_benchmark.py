#!/usr/bin/env python3
"""
run_benchmark.py — run the bioactive-food benchmark and/or the grounding ablation
against a local Ollama model (or any OpenAI-compatible endpoint).

Examples
--------
# main benchmark (baseline condition A) on your local 32B:
python run_benchmark.py --model qwen2.5:32b --mode benchmark

# full 4-condition grounding ablation:
python run_benchmark.py --model qwen2.5:32b --mode ablation

# both, custom host:
python run_benchmark.py --model llama3.1:70b --host http://localhost:11434/v1 --mode both

Outputs (written to ./results/):
  <model>_benchmark_raw.jsonl , <model>_benchmark_scores.csv
  <model>_ablation_raw.jsonl  , <model>_ablation_scores.csv
  <model>_summary.json        (headline numbers, ready to send back)
"""
import argparse, csv, json, os, sys, time
import foodllm_bench as fb

def read_compounds(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

def read_provenance(path):
    notes = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            notes[r["compound"]] = r.get("provenance_note", "")
    return notes

def run_condition(client, compounds, vocab, condition, provenance, log_fh):
    sysp = fb.system_prompt(vocab)
    rows = []
    n = len(compounds)
    for i, row in enumerate(compounds, 1):
        note = provenance.get(row["name"], "")
        user = fb.make_prompt(row, condition=condition, provenance_note=note)
        try:
            raw = client.chat(sysp, user)
            err = None
        except Exception as e:
            raw, err = "", str(e)
        pred = fb.parse_json(raw)
        g = fb.grade_row(pred, row)
        rec = {"model": client.model, "condition": condition, "compound": row["name"],
               "stratum": row["stratum"], "pred": pred, "raw": raw, "error": err, **g}
        rows.append(rec)
        log_fh.write(json.dumps(rec) + "\n"); log_fh.flush()
        if i % 10 == 0 or i == n:
            print(f"  [{condition}] {i}/{n}", file=sys.stderr, flush=True)
    return rows

def summarize(rows):
    def acc(rs, key):
        rs = [r for r in rs if r.get(key) is not None]
        return round(sum(bool(r[key]) for r in rs) / len(rs), 4) if rs else None
    out = {}
    for cond in sorted({r["condition"] for r in rows}):
        rs = [r for r in rows if r["condition"] == cond]
        overall = []
        for r in rs:
            overall.append(sum(bool(r[k]) for k in
                ["correct_bioactivity","correct_food","correct_cultural","correct_bioavail"]) / 4)
        by_str = {}
        for s in fb.STRATA:
            srs = [o for r, o in zip(rs, overall) if r["stratum"] == s]
            by_str[s] = round(sum(srs)/len(srs), 4) if srs else None
        gap = (round(by_str["Western"] - by_str["African"], 4)
               if by_str.get("Western") is not None and by_str.get("African") is not None else None)
        out[cond] = {
            "overall": round(sum(overall)/len(overall), 4) if overall else None,
            "by_task": {t: acc(rs, "correct_"+t) for t in
                        ["bioactivity","food","cultural","bioavail"]},
            "by_stratum": by_str,
            "western_african_gap": gap,
            "n": len(rs),
            "refusals": sum(1 for r in rs if r.get("error") or not r.get("pred")),
        }
    return out

def write_scores(rows, path):
    cols = ["model","condition","compound","stratum","correct_bioactivity",
            "correct_food","correct_cultural","correct_bioavail","error"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Ollama model tag, e.g. qwen2.5:32b")
    ap.add_argument("--host", default="http://localhost:11434/v1",
                    help="OpenAI-compatible base URL (Ollama default shown)")
    ap.add_argument("--mode", choices=["benchmark","ablation","both"], default="both")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--limit", type=int, default=0, help="limit #compounds (0=all; use for a smoke test)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    vocab = fb.load_vocab(os.path.join(args.data, "bioactivity_vocab.json"))
    compounds = read_compounds(os.path.join(args.data, "compounds.csv"))
    if args.limit:
        compounds = compounds[:args.limit]
    provenance = read_provenance(os.path.join(args.data, "provenance_notes.csv"))
    client = fb.ChatClient(base_url=args.host, model=args.model)
    tag = args.model.replace(":", "_").replace("/", "_")

    print(f"Model {args.model} @ {args.host} | {len(compounds)} compounds | mode={args.mode}",
          file=sys.stderr)
    summary = {"model": args.model, "n_compounds": len(compounds), "timestamp": time.time()}

    if args.mode in ("benchmark", "both"):
        with open(os.path.join(args.out, f"{tag}_benchmark_raw.jsonl"), "w") as fh:
            rows = run_condition(client, compounds, vocab, "A", provenance, fh)
        write_scores(rows, os.path.join(args.out, f"{tag}_benchmark_scores.csv"))
        summary["benchmark"] = summarize(rows)["A"]

    if args.mode in ("ablation", "both"):
        allrows = []
        with open(os.path.join(args.out, f"{tag}_ablation_raw.jsonl"), "w") as fh:
            for cond in ["A", "B", "C", "D"]:
                allrows += run_condition(client, compounds, vocab, cond, provenance, fh)
        write_scores(allrows, os.path.join(args.out, f"{tag}_ablation_scores.csv"))
        summary["ablation"] = summarize(allrows)

    with open(os.path.join(args.out, f"{tag}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== SUMMARY ===", file=sys.stderr)
    print(json.dumps(summary, indent=2))
    print(f"\nWrote results to ./{args.out}/  — send back the *_summary.json, "
          f"*_scores.csv and *_raw.jsonl files.", file=sys.stderr)

if __name__ == "__main__":
    main()
