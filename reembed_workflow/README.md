# Re-embedding Workflow (Custom Embedding Model)

This folder contains scripts to re-embed an existing Cognee database using your **custom OpenAI-compatible embedding model**. Configuration is read from the project `.env` (e.g. `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL`, `EMBEDDING_ENDPOINT`, `EMBEDDING_DIMENSIONS`). Progress and logs stay in this folder and do not touch `scripts/embedding_migration/`.

## Prerequisites

1. **Python environment** with project dependencies (e.g. from repo root: `uv sync --dev --all-extras`).
2. **`.env`** at repo root with your embedding settings, for example:
   ```env
   EMBEDDING_PROVIDER="openai"
   EMBEDDING_MODEL="Qwen/Qwen3-Embedding-8B"
   EMBEDDING_ENDPOINT="http://10.213.191.186:8888/v1/embeddings"
   EMBEDDING_API_KEY=""
   EMBEDDING_DIMENSIONS="4096"
   # Optional: use server's /tokenize (e.g. vLLM) for token count instead of local tokenizer
   # EMBEDDING_USE_SERVER_TOKENIZER="true"
   ```
3. **PostgreSQL** and pgvector extension running; DB connection configured in `.env`.
4. **Backup the database** before altering schema or re-embedding.

## Step-by-Step

Run all commands from the **repository root** (e.g. `/path/to/cognee`).

### 1. Backup database

```bash
pg_dump -h YOUR_HOST -U YOUR_USER -d YOUR_DB -F c -f backup_pre_reembed_$(date +%Y%m%d).dump
```

### 2. Summary (impacted tables and config)

```bash
uv run python reembed_workflow/01_summary.py
```

Confirm the listed tables and that "Target vector columns" matches `EMBEDDING_DIMENSIONS` in `.env` (e.g. 4096).

### 3. Alter vector column dimension

If the current vector type is not already `vector(4096)`, run the SQL script (adjust `psql` connection to match your `.env`):

```bash
PGPASSWORD=YOUR_PASSWORD psql -h YOUR_HOST -U YOUR_USER -d YOUR_DB -f reembed_workflow/alter_dimension_4096.sql
```

This sets all vector columns to NULL and alters them to `vector(4096)`. The script **drops any index on the vector column** first (Step 1.5), then ALTERs. You must run this on the **exact same host and database** as in your `.env` (e.g. `DB_HOST`, `DB_NAME`).

**If you see "expected 3072 dimensions, not 4096" when re-embedding:** the column is still 3072 at runtime. Run the diagnostic (no embeddings, fast):

```bash
uv run python reembed_workflow/diagnose_dimension.py
```

If it prints "Probe FAILED" and "Column is still vector(3072)", re-run the ALTER script on this DB and ensure it completes without errors, then run the diagnostic again until it says "Probe OK".

### 4. Re-embed data

```bash
# Optional: dry run first
uv run python reembed_workflow/02_reembed.py --dry-run

# Full run
uv run python reembed_workflow/02_reembed.py

# With custom batch size
uv run python reembed_workflow/02_reembed.py --batch-size 100

# Resume if interrupted
uv run python reembed_workflow/02_reembed.py --resume

# Single table only
uv run python reembed_workflow/02_reembed.py --table DocumentChunk_text
```

Generated in this folder: `progress.json`, `reembed_YYYYMMDD_HHMMSS.log`.

### 5. Verify

```bash
uv run python reembed_workflow/03_verify.py
```

Checks: vector dimensions match `EMBEDDING_DIMENSIONS`, embedding engine, search on `DocumentChunk_text`, and no null vectors.

### 6. Run retrieval evaluation (testing)

After re-embedding and verification, run the existing retrieval evaluation script with your ground-truth CSV:

```bash
uv run python examples/python/retrieval_evaluation/run_eval_v2_compat.py \
  --input /path/to/ground_truth.csv \
  --output reembed_workflow/results/eval_output.csv \
  --log reembed_workflow/results/eval.log \
  --triplets-json reembed_workflow/results/eval_triplets.json \
  --top-k 5
```

Create the results directory if needed: `mkdir -p reembed_workflow/results`.

Alternatively, edit and run the wrapper script:

```bash
# Edit reembed_workflow/run_eval.sh to set INPUT_CSV and optional paths, then:
bash reembed_workflow/run_eval.sh
```

## Files in this folder

| File | Purpose |
|------|--------|
| `README.md` | This file |
| `01_summary.py` | Summary of impacted tables and embedding config (dimension from `.env`) |
| `02_reembed.py` | Batch re-embed with resume; dimension check from `EMBEDDING_DIMENSIONS` |
| `03_verify.py` | Verify dimensions, engine, search, integrity (dimension from `.env`) |
| `alter_dimension_4096.sql` | Set vectors to NULL, then ALTER to `vector(4096)` |
| `run_eval.sh` | Optional wrapper to run `run_eval_v2_compat.py` (edit paths inside) |
| `progress.json` | Created by `02_reembed.py`; used for `--resume` |
| `reembed_*.log` | Log files from `02_reembed.py` |

## Troubleshooting

- **Embedding engine error**: Check `.env` (endpoint, API key, model name). Ensure the embedding service is reachable.
- **Vector dimension mismatch**: Ensure you ran `alter_dimension_4096.sql` and that `EMBEDDING_DIMENSIONS` in `.env` is 4096 (or match the SQL dimension).
- **Script stops mid-run**: Run again with `--resume`. Check `progress.json` and the latest `reembed_*.log`.
- **Rate limits**: Use a smaller batch size, e.g. `--batch-size 20`.
