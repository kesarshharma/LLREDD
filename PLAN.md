# LEREDD Project Plan & Milestones

This document tracks the milestones and definition of done for the **LEREDD** project implementation.

## Milestones

- [x] **Milestone 1: Environment & Core Data Models**
  - Package structure initialized.
  - Configuration settings via Pydantic (`leredd/config.py`).
  - Core data models (`Requirement`, `RequirementPair`, `Prediction`, `EvaluationReport`).

- [x] **Milestone 2: Automotive SRS Datasets & Ground Truth**
  - Raw SRS text documents created for ADB, TJA, and APA systems (`data/raw/`).
  - Ground truth annotated requirement pair datasets created (`data/annotated/`).

- [x] **Milestone 3: Knowledge Retrieval & Prompt Engineering**
  - SBERT embedding wrapper with disk caching (`leredd/embeddings.py`).
  - Dynamic In-Context Learning example retrieval with Eq. 1 (Average) & Eq. 2 (Max-Avg) similarity formulas (`leredd/retrieval.py`).
  - Fixed-size RAG context chunking and vector retrieval (`leredd/retrieval.py`).
  - Paper Figure 2 & Table I prompt builder (`leredd/prompts.py`).
  - Abstract LLM client layer supporting OpenAI, Ollama, and Mock LLM (`leredd/llm.py`).

- [x] **Milestone 4: LEREDD Engine, Baselines & Evaluation Suite**
  - Detection pipeline with threshold-based re-annotation (`leredd/detector.py`).
  - Baseline 1: TF-IDF & LSA (`leredd/baselines/tfidf_lsa.py`).
  - Baseline 2: Fine-Tuned BERT multi-class classifier (`leredd/baselines/bert_classifier.py`).
  - Statistical significance tests (McNemar's test & Fisher's exact test) and metrics calculator (`leredd/evaluation.py`).
  - Cohen's Kappa score calculator & utils (`leredd/utils.py`).

- [x] **Milestone 5: CLI, REST API & Streamlit Web UI**
  - Typer CLI application (`leredd-cli` / `leredd/cli.py`).
  - FastAPI REST API (`api/main.py`, `api/routes.py`).
  - Streamlit multi-page web UI (`ui/app.py`).

- [x] **Milestone 6: Testing, Docker, CI/CD & Documentation**
  - Pytest unit and integration test suite with high coverage (`tests/`).
  - Multi-stage Docker setup & docker-compose (`Dockerfile`, `docker-compose.yml`).
  - GitHub Actions CI workflow (`.github/workflows/ci.yml`).
  - Documentation (`README.md`, `docs/API.md`, `docs/REPLICATION.md`).
