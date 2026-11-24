# LEREDD Independent Annotation Guide

## Purpose

Use this guide to collect two independent human labels on the same requirement
pairs. The workflow is for agreement measurement and adjudication preparation.
It does not promote an annotation to verified ground truth. Only a separate,
explicit human decision in the existing review/export workflow can do that.

The taxonomy starts from the project proposal and the LEREDD paper's definitions
([arXiv v2, Sections III and IV-C](https://arxiv.org/abs/2602.22456v2)). Use the
ordered pair shown by the tool: Requirement A is the first requirement and
Requirement B is the second.

## Label definitions

| Label | Use when |
|---|---|
| `Requires` | A cannot be fully satisfied unless B is satisfied or a condition established by B is available. Read this as “A requires B.” |
| `Implements` | A is a higher-level requirement and B is a lower-level requirement that fulfills or realizes A. Read this as “A is implemented by B” in the project taxonomy. |
| `Conflicts` | A and B impose incompatible expectations or constraints that cannot both be met under the same conditions. This relation is symmetric. |
| `Details` | A and B describe the same action under the same condition, and A supplies additional detail or refinement for B. Read this as “A details B.” |
| `Is similar` | A and B express substantially overlapping meaning, without evidence that one requires, implements, conflicts with, or details the other. This relation is symmetric. |
| `No_dependency` | Available SRS evidence does not establish any of the five relations above. Do not use this as a substitute for uncertainty. |
| Abstain / defer | The evidence is insufficient, ambiguous, or contradictory. Use the tool's `X` option and briefly state why. Abstention is reported separately and is not a seventh taxonomy label. |

For a pair that appears to have multiple relations, record only the single most
evident relation and explain the evidence. Do not infer a dependency solely from
shared terminology or topic similarity. Use the supplied SRS context where
available; do not fill in missing system behavior from outside assumptions.

## Independent review procedure

The assisted review queue routes non-empty requirement statements shorter than
20 characters to reviewers by default as a screening cue. `queue-create` accepts
`--minimum-requirement-characters N` to change that threshold; set it to `0` to
disable the cue. This rule does not decide that a statement is incomplete or
ambiguous, change a dependency label, or replace reviewer judgment. Empty text
continues to use the existing missing-metadata review reason. The queue manifest
records the eligibility configuration used for each queue build.

1. Prepare one fixed candidate-pair JSON file and, if relevant, one UTF-8 SRS
   context file. Use the exact same files for both reviewers. The tools record
   their SHA-256 hashes in each manifest.
2. Assign two distinct, attributable reviewer IDs. The application records the
   IDs but does not authenticate them.
3. Have each reviewer run the blind annotation command separately. Do not show
   either reviewer the model or heuristic suggestions, the other reviewer's
   answers, or previous decisions until both manifests are complete.
4. Reviewers select one of the six labels or abstain, and may add a short
   rationale. The timer measures elapsed time from display of a pair until its
   label and rationale are submitted. It can include pauses and does not measure
   setup, training, or off-screen SRS reading; report this limitation.
5. Compare the completed manifests. The report includes pairwise observed
   agreement, Cohen's kappa, a confusion matrix, abstentions, and per-reviewer
   time summaries. Kappa is `null` when the label marginals make it undefined.
6. Preserve both original manifests. A human adjudicator must resolve
   disagreements separately and record any consensus decisions through the
   verified review workflow. Never replace the independent labels with the
   consensus result.

Example commands, with a fixed prepared sample and outputs kept outside the
imported replication data and historical reports:

```powershell
python scripts/annotate_independent.py `
  --input .pytest-tmp\pilot-pairs.json `
  --context .pytest-tmp\pilot-srs.txt `
  --annotator reviewer-A `
  --output .pytest-tmp\reviewer-A.json

python scripts/annotate_independent.py `
  --input .pytest-tmp\pilot-pairs.json `
  --context .pytest-tmp\pilot-srs.txt `
  --annotator reviewer-B `
  --output .pytest-tmp\reviewer-B.json

python scripts/compare_independent_annotations.py `
  --reviewer-a .pytest-tmp\reviewer-A.json `
  --reviewer-b .pytest-tmp\reviewer-B.json `
  --output .pytest-tmp\agreement-report.json
```

The prepared pair file should be a fixed sample chosen before either reviewer
starts. Do not construct a “research sample” from synthetic test pairs. Use
synthetic pairs only to exercise the software. The annotation manifests remain
independent and unverified; the comparison command does not adjudicate labels.
Synthetic/demo origin is retained on extracted requirements, candidate pairs,
review decisions, and CSV/JSON outputs. Marked records cannot be used as
evaluation ground truth or released datasets and are excluded from example
retrieval. Do not remove those markers to make a demo dataset pass validation.

## Candidate positive-recall pilot

The hybrid top-5 pilot in `.pytest-tmp/task-p50-blind-pilot/` contains a fixed
50/50 sample from the strategy-selected and strategy-omitted strata. Give both
reviewers the exact `blind-pilot-pairs.json` and the same NASA Trick SRS context.
Keep `analysis-only-strata-map.json` private until both independent annotation
manifests are complete. Resolve disagreements through the explicit human
adjudication/review workflow and export the exact 100 pair IDs with coherent
`VERIFIED` human provenance. Independent reviewer vectors alone are not eligible
as ground truth.

Only after that adjudication, run the sampling analysis with a fresh output path:

```powershell
python scripts/analyze_candidate_coverage.py `
  --pilot-manifest .pytest-tmp\task-p50-blind-pilot\pilot-manifest.json `
  --strata-map .pytest-tmp\task-p50-blind-pilot\analysis-only-strata-map.json `
  --verified-pairs .pytest-tmp\task-p50-blind-pilot\verified-adjudications.json `
  --output .pytest-tmp\task-p50-blind-pilot\candidate-recall-report.json
```

The command verifies hashes for the SRS, blinded sample, candidate output,
complete pair universe, analysis map, and adjudicated labels. It requires exact
sample-ID coverage and verified human labels; missing, provisional, or
contradictory records stop the analysis. Any of the five dependency classes is
counted positive; `No_dependency` is negative. The report uses the recorded
stratum inclusion weights and labels its normal-approximation interval as
exploratory. This 100-pair pilot is not powered and cannot establish general
performance. No report exists until real adjudications are supplied.

## Optional multi-model provisional labels

The CLI can collect votes from distinct configured model IDs with
`leredd-cli label --mode multi_llm --models model-a,model-b`. Only unanimous
valid votes yield a label; any disagreement or incomplete vote leaves the pair
unlabeled and records each model's vote/rationale for review. In `hybrid` mode,
heuristic/model disagreement is also left unresolved. These predictions remain
provisional and are not human agreement or ground truth. The `mock` provider is
for software testing only; real OpenAI/Ollama runs require the user's explicit
provider configuration and may make external or paid requests.

## Manual-versus-assisted effort study

This separate paired study estimates descriptive annotation time with and
without provisional machine suggestions. It requires two distinct reviewers and
a fixed pair file whose records all have provisional LLM, heuristic, or ensemble
provenance with an available suggestion. Imported, human-labeled, verified, and
unprovenanced records are rejected. Each pair is assigned once to each
condition, by different reviewers; the two conditions are counterbalanced
across reviewers. Do not discuss pairs or answers until both reviewers finish.

Select a rights-cleared real sample before either reviewer begins. Synthetic
records are for testing the software only. The tools preserve candidate inputs,
write new schedule/manifests/reports outside the project's `data/` tree, and
never promote study labels to verified ground truth. Reviewer IDs are
self-reported, not authenticated. The timer starts when the pair is displayed
and stops after the label and optional rationale; it can include pauses and
does not include setup or context-reading time. The comparison is descriptive
only, does not measure annotation correctness, and does not test significance.

```powershell
python scripts/prepare_annotation_effort_study.py `
  --input .pytest-tmp\pilot-pairs.json `
  --context .pytest-tmp\pilot-srs.txt `
  --reviewer-a reviewer-A `
  --reviewer-b reviewer-B `
  --seed 42 `
  --output .pytest-tmp\effort-schedule.json

python scripts/run_annotation_effort_study.py `
  --input .pytest-tmp\pilot-pairs.json `
  --schedule .pytest-tmp\effort-schedule.json `
  --context .pytest-tmp\pilot-srs.txt `
  --reviewer reviewer-A `
  --output .pytest-tmp\reviewer-A-effort.json

python scripts/run_annotation_effort_study.py `
  --input .pytest-tmp\pilot-pairs.json `
  --schedule .pytest-tmp\effort-schedule.json `
  --context .pytest-tmp\pilot-srs.txt `
  --reviewer reviewer-B `
  --output .pytest-tmp\reviewer-B-effort.json

python scripts/compare_annotation_effort.py `
  --reviewer-a .pytest-tmp\reviewer-A-effort.json `
  --reviewer-b .pytest-tmp\reviewer-B-effort.json `
  --schedule .pytest-tmp\effort-schedule.json `
  --output .pytest-tmp\effort-report.json
```

Replace the illustrative paths with a fixed, appropriately licensed sample and
fresh output paths. Run each reviewer command separately, with the other
reviewer's manifest inaccessible until completion. Do not report test-fixture
times as human-effort results.

## Imported “Implemented by” records

The source replication spreadsheet uses the text label `Implemented by`, while
the project schema uses `Implements`. The paper defines `Implements` direction
as a higher-level requirement fulfilled by a lower-level requirement, but the
available spreadsheet does not provide a separate mapping rule for its
`Implemented by` label. Preserve the raw source label and review each such item
with the source convention available. Do not automatically remap it based on
wording alone.
