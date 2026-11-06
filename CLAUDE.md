# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

Product Review Intelligence Platform: NLP over product reviews (sentiment, keywords/aspects, summaries, pros/cons, complaint clustering). The README describes a larger target stack (FastAPI, Snowflake, dbt, PostgreSQL, Superset, OpenAI agent, Streamlit Cloud, Docker), but **none of that exists yet**. Real work currently lives in Jupyter notebooks; the app and `utils/` are mostly scaffolding.

There is no build system, test suite, or linter configured. Dependencies are in `requirements.txt` (unpinned); a local `venv/` is checked into the working dir (untracked — don't edit it).

```bash
source venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.1:8b     # one-time; Ollama server must be running
streamlit run app.py          # currently only a CSV upload + df.head() preview
jupyter lab notebooks/        # primary development surface
```

## Architecture

- **`notebooks/`** — the actual pipeline, numbered in execution order: `01` data loading/EDA → `02_01` preprocessing → `02_02` EDA → `03` sentiment → `04` pros/cons summaries (file is named `04_keyword_extraction`) → `05` complaint topics (file is named `05_review_summarization`) → `06` final pipeline test → `07_*` evaluation notebooks (sentiment metrics, summaries, clustering). `notebooks/ML-development/` holds later experiments. Logic is prototyped here first, then promoted into `utils/`.
- **Models** — exactly three, all wired through `utils/`:
  - **Llama 3.1 8B via Ollama** (`utils/llm.py`, `LLAMA_MODEL = "llama3.1:8b"`): sentiment (`utils/sentiment.py`, JSON-mode prompt → `llama_label`, `llama_score`, `is_complaint` = negative with confidence ≥ 0.85) and pros/cons summaries (`utils/summarizer.py`). Needs `ollama pull llama3.1:8b` and a running server; batch helpers use a thread pool, so set `OLLAMA_NUM_PARALLEL` to match `max_workers`.
  - **BGE-M3** (`utils/embeddings.py`, `BAAI/bge-m3` via sentence-transformers, normalised 1024-d vectors): all semantic embedding, including the relevance metric in `07_02`.
  - **BERTopic** (`utils/topics.py`): recurring complaints. It is fitted **once on all complaints** (not per product — UMAP/HDBSCAN need a large corpus), then topics are aggregated per product. Notebook 05 writes `complaint_clusters.parquet` and `complaint_topic_assignments.parquet`; `07_03` reads both.
- **`utils/`** — also `visualization.py` (matplotlib helpers) and `preprocessing.py` / `keyword_extraction.py` (mostly `pass` stubs; only `clean_text`/`load_reviews` are real).
- **`data/`** (untracked) — `Apparel_review.tsv` and `processed_reviews.csv` as raw/cleaned inputs; `data/processed/sentiment/` holds notebook outputs as parquet (scored reviews, monthly trends, per-category `*_summaries.parquet`, `complaint_clusters.parquet`, `clustering_eval_scores.parquet`) and `data/processed/ml_outputs/`. Notebooks read/write these paths relative to the notebook dir, so run them from `notebooks/`.
- **`models/`** (untracked) — a classical-ML sentiment baseline: scikit-learn SVM pipelines (baseline, grid-search best, params, label order, val/test splits as parquet/npy, macro-F1 scores) pickled from earlier experiments. Not used by `utils/`.
- **`dashboards/`, `assets/`** — empty placeholders for Superset/app assets.

## Notes

- Notebooks are modified frequently and committed with their outputs; expect large diffs.
- Some notebook/dir names contain typos (`07_03_clsutering_evaluate_metric`); don't rename without checking references.
