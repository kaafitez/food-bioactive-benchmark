# A culturally stratified benchmark of language-model reasoning on food bioactives

Data and code for the illustrative analysis reported in the review
*"Bioactive-aware food AI: why language models can taste but not yet reason
about health, and how cultural provenance can help."*

The analysis asks language models to predict health-relevant properties of
bioactive compounds stratified by cultural provenance, and tests whether
supplying explicit provenance closes the resulting accuracy gap.

## Contents

```
data/
  bioactive_benchmark_set.csv   Compound set: name, cultural stratum, validated
                                SMILES, reference bioactivity class, food source,
                                bioavailability tier, physicochemical descriptors.
  provenance_notes.csv          Short retrieved provenance note per compound
                                (used in the grounding conditions).
  bioactivity_vocab.json        Controlled vocabulary of bioactivity classes.
  benchmark_scores.csv          Per-item scores, four models x four tasks (baseline).
  ablation_scores.csv           Per-item scores across the four grounding conditions
                                (name only; +structure; +provenance; both).
  crossfamily_summary.csv       Western-African gap, baseline vs +provenance, for
                                six model families from five developers.
  structure_probe_results.csv   Supervised probe of ChemBERTa / RDKit descriptors
                                decoding cultural stratum vs. structural properties.
code/
  harness/                      Runnable local-inference harness (Ollama / OpenAI-
                                compatible). See harness/README.md to reproduce.
  make_all_figures.py           Regenerates the full figure set from data/.
  make_review_figures.py        Regenerates the review's six figures from data/.
```

## Method (summary)

- **Compounds:** 100 bioactives with computationally validated structures,
  balanced at 25 across Western, East Asian, South Asian and African food systems.
- **Tasks:** predict bioactivity class, food source, cultural context of use and
  bioavailability tier, from compound name and structure alone.
- **Models:** four LLMs spanning a capability ladder; cross-family replication
  across six open- and closed-weight families from five developers.
- **Grading:** deterministic normalized-token-overlap matching against curated
  references (no model-based judge), for full reproducibility.
- **Grounding analysis:** four input conditions isolate the contribution of
  molecular structure vs. retrieved cultural provenance.
- **Statistics:** two-sided Mann-Whitney and Wilcoxon tests as appropriate.

## Reproducing

```bash
pip install pandas numpy matplotlib rdkit
python code/make_review_figures.py     # figures -> figures_review/
# to re-run the model benchmark, see code/harness/README.md
```

## Licensing

- Code (`code/`): MIT License (see LICENSE).
- Data (`data/`): Creative Commons Attribution 4.0 (CC-BY-4.0).

## Citation

If you use these materials, please cite the review article (details to be added
on publication).
