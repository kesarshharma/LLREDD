# LEREDD Research Replication Guide

This guide describes how to reproduce the empirical evaluation results from the research paper *"Automating the Detection of Requirement Dependencies Using Large Language Models"*.

## Paper Research Questions (RQs) Summary

1. **RQ1 (Model Selection)**: GPT-4 achieves superior performance compared to open-source LLMs under zero-shot prompting (F1 = 0.78 overall, 0.87 for No_dependency).
2. **RQ2 (Optimal Configuration)**:
   - Embeddings: SBERT `all-MiniLM-L6-v2`
   - Distance: Euclidean distance ($sim = \frac{1}{1 + \text{Euclidean}}$)
   - Similarity Aggregation: Average formula (Eq. 1)
   - Examples per type: $k = 6$
   - Re-annotation Threshold: Confidence score $\le 4 \rightarrow \text{No\_dependency}$
   - RAG: 6 chunks of 1000 characters (200 overlap) for cross-system setting.
3. **RQ3 (Prompting Strategies)**: Few-shot prompting yields substantial gains over zero-shot ($F_1 = 0.91$). RAG provides no additional benefit over few-shot in intra-system setting, but improves performance in cross-system setting.
4. **RQ4 (Intra-System Baselines Comparison)**: LEREDD outperforms TF-IDF & LSA ($F_1 = 0.66$) and fine-tuned BERT ($F_1 = 0.80$) with an overall average $F_1 = 0.88$.
5. **RQ5 (Cross-System Baselines Comparison)**: LEREDD with RAG demonstrates strong cross-dataset generalization ($F_1 = 0.86$), outperforming BERT ($F_1 = 0.57$) and TF-IDF ($F_1 = 0.69$).

---

## Replication Commands

### 1. Run Benchmark Evaluation Script
```bash
python scripts/run_evaluation.py
```

### 2. Run CLI Evaluation Command
```bash
leredd-cli evaluate --predictions data/annotated/adb_pairs.json --ground-truth data/annotated/adb_pairs.json --system ADB
```

### 3. Run Streamlit UI Evaluation Page
```bash
streamlit run ui/app.py
```
Navigate to **Page 4: Model Evaluation** to view full accuracy, macro/weighted F1 score tables, McNemar's test $p$-values, and confusion matrix heatmaps.
