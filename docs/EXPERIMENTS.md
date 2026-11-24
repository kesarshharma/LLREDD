# Evaluation Runs and Evidence Limits

## Reproducing the checked-in local report

The checked-in `data/evaluation/system_disjoint_report.json` is a historical
local run over the legacy split: 10 ADB training pairs, 5 APA development pairs,
and 5 TJA test pairs. It reports BERT accuracy 0.20 (macro F1 0.125), TF-IDF
accuracy 0.40 (macro F1 0.25), heuristic accuracy 0.20 (macro F1 about 0.083),
and MockLLM LEREDD accuracy 0.40 (macro F1 0.30). The test sample has only five
pairs, no annotation provenance, and covers four of six labels. These values
are descriptive of that local run only; they are not a reliable estimate of
generalization or research-grade performance.

Recreate a report without replacing the historical artifact:

```powershell
$run = ".pytest-tmp\evaluation-run"
New-Item -ItemType Directory -Force $run | Out-Null
python scripts/run_full_evaluation.py `
  --split-dir data/splits/system_disjoint `
  --output "$run\report.json"
```

The runner writes both `report.json` and `ablation_report.json` in that output
directory. LEREDD uses `MockLLMClient`; no remote LLM is called. The SBERT
embedding model must be available locally or the configured embedding fallback
must be usable. The report records the supplied split's actual systems,
partition counts, label coverage, provenance eligibility, and limitations.

The report's `ablation_validity` metadata marks retrieval-strategy and
in-context-example comparisons as non-informative under MockLLM because the
deterministic mock does not interpret retrieved examples. It also marks
confidence-threshold comparisons as non-informative because the mock always
returns confidence 5. These ablations exercise wiring only; they are not
evidence for or against those design choices. Use a real authorized provider
or a separately validated controlled test double to study those effects.

## Interpretation limits

- The local report's labels have no provenance, so the runner accepts them only
  through the documented legacy compatibility path and reports this limitation.
- A provenance-bearing annotation is eligible only when it is verified, has
  human source and a non-placeholder self-reported reviewer identity, and its
  accepted label matches the scored label. The software does not authenticate
  that identity.
- LEREDD's results are from a deterministic mock backend, not GPT-4, OpenAI, or
  Ollama. No paper-reference score is reproduced by this local report.
- The small legacy test partition and class coverage do not support claims of
  statistical significance or publication-quality model comparison. Reported
  McNemar results are exploratory.
- Replication-package imports remain pending local human review. Their label
  direction and redistribution rights must be resolved before evaluation or
  publication.

Do not edit the checked-in JSON to represent a new run. Use a new output path,
retain the split and configuration used, and report synthetic fixture runs as
software tests rather than annotation or model-quality evidence.
