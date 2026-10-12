# Replication Package Integration Note

## Source
- Original dataset: LEREDD replication package found at `docs/external/Replication package/`.
- Paper citation: arXiv:2602.22456 [cs.SE] (Darif et al., 2026); DOI `https://doi.org/10.48550/arXiv.2602.22456`.
- License reported in PDF metadata: CC BY 4.0 (Creative Commons Attribution 4.0 International).

## Dataset-specific licensing status
- The replication package documentation (manifests, RQ scripts, annotations spreadsheets) does not contain an independent dataset-use agreement, replication URL, or additional license clause beyond the paper's CC BY 4.0.
- The paper's stated license is evidence about the paper; it does not by itself establish redistribution rights for separately packaged annotation spreadsheets. Dataset-specific rights remain unverified. Preserve provenance and attribution, and obtain confirmation before redistributing the data.

## Integration approach
- Four annotated spreadsheets loaded read-only via `zipfile` + `xml.etree.ElementTree` (standard library, no `openpyxl` required).
- Rows preserved with synthetic IDs (`ADB_EVAL_ROW_2`, etc.), original worksheet row numbers, original `Dependency_Status` values (including `Implemented by`), and provenance from `AnnotationProvenance`.
- Original label direction (`Implemented by`) preserved as original; mapping to internal `Implements` left unresolved until Eq. 1/2 direction convention is confirmed from paper definitions.
- Pilot (100), ADB Evaluation (313), APA (200), TJA (200) kept separate; no pooled split created.

## Limitations / unresolved
- No openpyxl cell-level duplicate audit performed at integration time (would require openpyxl install).
- No replication URL or official code/data repository URL found inside package root.
- No provenance/status/annotator IDs embedded in original spreadsheet XML; provenance is reconstructed from package source.
- The local loader marks imported records `pending_review`; they are ineligible for evaluation until independently reviewed and explicitly marked `verified` through the local human review workflow. Do not change source annotations merely to enable the prototype.
- The replication evaluation runner uses MockLLM and synthetic requirement IDs and is not an exact paper reproduction. Current imported pending-review records make it fail closed before any metrics are generated.
