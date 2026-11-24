# LEREDD Local Evaluation Guide

The commands here run the repository's small, leakage-checked system-disjoint demonstration benchmark. They do **not** reproduce the paper's reported results: the included legacy benchmark has 20 pairs, its annotation provenance is unavailable, and the LEREDD variants use `MockLLMClient`.

## Paper-Reported Research Question Results

The figures below summarize values attributed to the research paper in this project’s prior notes. They are reference claims, not results produced by this local benchmark.

1. **RQ1 (Model Selection)**: GPT-4 achieves superior performance compared to open-source LLMs under zero-shot prompting (F1 = 0.78 overall, 0.87 for No_dependency).
2. **RQ2 (Optimal Configuration)**:
   - Embeddings: SBERT `all-MiniLM-L6-v2`
   - Distance: Euclidean distance ($sim = \frac{1}{1 + \text{Euclidean}}$)
   - Similarity Aggregation: Average formula (Eq. 1)
   - Examples per type: $k = 6$
   - Paper-reported threshold method: confidence score $\le 4 \rightarrow \text{No\_dependency}$
   - RAG: 6 chunks of 1000 characters (200 overlap) for cross-system setting.
The threshold above records the paper configuration for comparison. The local
LEREDD implementation preserves a low-confidence predicted class and marks it
`requires_review`; local reports must not describe this as reproducing the
paper's threshold relabeling behavior.

3. **RQ3 (Prompting Strategies)**: Few-shot prompting yields substantial gains over zero-shot ($F_1 = 0.91$). RAG provides no additional benefit over few-shot in intra-system setting, but improves performance in cross-system setting.
4. **RQ4 (Intra-System Baselines Comparison)**: LEREDD outperforms TF-IDF & LSA ($F_1 = 0.66$) and fine-tuned BERT ($F_1 = 0.80$) with an overall average $F_1 = 0.88$.
5. **RQ5 (Cross-System Baselines Comparison)**: LEREDD with RAG demonstrates strong cross-dataset generalization ($F_1 = 0.86$), outperforming BERT ($F_1 = 0.57$) and TF-IDF ($F_1 = 0.69$).

---

## Local Evaluation Commands

### 1. Run the system-disjoint benchmark
```bash
python scripts/run_evaluation.py
```

The command is retained for compatibility and delegates to `scripts/run_full_evaluation.py`. Each default run writes to a new timestamped directory under `data/evaluation/runs/`; pass `--output` for a chosen new path. Existing reports and source-data directories are protected from overwrite.

To run the underlying entry point directly:
```bash
python scripts/run_full_evaluation.py --split-dir data/splits/system_disjoint --output data/evaluation/system_disjoint_report.json
```

That direct example writes the standard report and ablation report; choose a fresh output location if prior results must be preserved.

### 2. Score an existing prediction file with the CLI

`leredd-cli evaluate` expects prediction records (including `pair_id` and predicted label) in `--predictions` and the corresponding labeled pairs in `--ground-truth`. Do not point both arguments at the same ground-truth JSON file.

### 3. Use the Streamlit evaluation page
```bash
streamlit run ui/app.py
```
Navigate to **Page 4: Model Evaluation** to run the fixed system-disjoint split and view metrics, exploratory statistical results, limitations, and a confusion matrix. Each UI run writes into a unique subdirectory under `data/evaluation/ui_runs/`.
