-- ============================================================
-- COGNEE RE-EMBEDDING WORKFLOW - ALTER COLUMN TO vector(4096)
-- ============================================================
--
-- pgvector cannot ALTER TYPE when the column already contains
-- data of a different dimension. Solution: set vector to NULL,
-- then ALTER, then run re-embedding.
--
-- ⚠️  BACK UP THE DATABASE BEFORE RUNNING THIS!
--
-- Run from repo root (adjust connection to match your .env):
--   PGPASSWORD=YOUR_PASSWORD psql -h YOUR_HOST -U YOUR_USER -d YOUR_DB -f reembed_workflow/alter_dimension_4096.sql
-- ============================================================

-- Step 1: Show current column types
\echo '======================================'
\echo 'STEP 1: Current column types'
\echo '======================================'

SELECT
    relname as table_name,
    pg_catalog.format_type(atttypid, atttypmod) as current_type
FROM pg_attribute
JOIN pg_class ON pg_class.oid = pg_attribute.attrelid
WHERE attname = 'vector'
AND relname IN (
    'DocumentChunk_text',
    'Entity_name',
    'EntityType_name',
    'EdgeType_relationship_name',
    'TextSummary_text',
    'TextDocument_name'
)
ORDER BY relname;

-- Step 1.5: Drop indexes on vector column (dimension-specific; can block ALTER / cause 3072 vs 4096 errors)
\echo ''
\echo '======================================'
\echo 'STEP 1.5: Drop indexes on vector column'
\echo '======================================'

DO $$
DECLARE
  idx RECORD;
BEGIN
  FOR idx IN
    SELECT n.nspname AS schema_name, i.relname AS index_name
    FROM pg_index ix
    JOIN pg_class i ON i.oid = ix.indexrelid
    JOIN pg_class c ON c.oid = ix.indrelid
    JOIN pg_namespace n ON n.oid = i.relnamespace
    JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum = ANY(ix.indkey) AND a.attname = 'vector' AND NOT a.attisdropped
    WHERE n.nspname = 'public' AND c.relname IN (
      'DocumentChunk_text', 'Entity_name', 'EntityType_name',
      'EdgeType_relationship_name', 'TextSummary_text', 'TextDocument_name'
    )
  LOOP
    EXECUTE format('DROP INDEX IF EXISTS %I.%I', idx.schema_name, idx.index_name);
  END LOOP;
END $$;

\echo 'Vector column indexes dropped (if any).'

-- Step 2: Set all vector columns to NULL
\echo ''
\echo '======================================'
\echo 'STEP 2: Set vector column to NULL'
\echo '======================================'

-- Use schema "public" so reembed script (default REEMBED_VECTOR_SCHEMA=public) hits same tables
UPDATE public."DocumentChunk_text" SET vector = NULL;
\echo 'DocumentChunk_text: vector set to NULL'

UPDATE public."Entity_name" SET vector = NULL;
\echo 'Entity_name: vector set to NULL'

UPDATE public."EntityType_name" SET vector = NULL;
\echo 'EntityType_name: vector set to NULL'

UPDATE public."EdgeType_relationship_name" SET vector = NULL;
\echo 'EdgeType_relationship_name: vector set to NULL'

UPDATE public."TextSummary_text" SET vector = NULL;
\echo 'TextSummary_text: vector set to NULL'

UPDATE public."TextDocument_name" SET vector = NULL;
\echo 'TextDocument_name: vector set to NULL'

-- Step 3: ALTER COLUMN TYPE to vector(4096)
\echo ''
\echo '======================================'
\echo 'STEP 3: ALTER COLUMN to vector(4096)'
\echo '======================================'

ALTER TABLE public."DocumentChunk_text" ALTER COLUMN vector TYPE vector(4096);
\echo 'DocumentChunk_text: altered to vector(4096)'

ALTER TABLE public."Entity_name" ALTER COLUMN vector TYPE vector(4096);
\echo 'Entity_name: altered to vector(4096)'

ALTER TABLE public."EntityType_name" ALTER COLUMN vector TYPE vector(4096);
\echo 'EntityType_name: altered to vector(4096)'

ALTER TABLE public."EdgeType_relationship_name" ALTER COLUMN vector TYPE vector(4096);
\echo 'EdgeType_relationship_name: altered to vector(4096)'

ALTER TABLE public."TextSummary_text" ALTER COLUMN vector TYPE vector(4096);
\echo 'TextSummary_text: altered to vector(4096)'

ALTER TABLE public."TextDocument_name" ALTER COLUMN vector TYPE vector(4096);
\echo 'TextDocument_name: altered to vector(4096)'

-- Step 4: Verify result
\echo ''
\echo '======================================'
\echo 'STEP 4: Verify ALTER result'
\echo '======================================'

SELECT
    relname as table_name,
    pg_catalog.format_type(atttypid, atttypmod) as new_type
FROM pg_attribute
JOIN pg_class ON pg_class.oid = pg_attribute.attrelid
WHERE attname = 'vector'
AND relname IN (
    'DocumentChunk_text',
    'Entity_name',
    'EntityType_name',
    'EdgeType_relationship_name',
    'TextSummary_text',
    'TextDocument_name'
)
ORDER BY relname;

\echo ''
\echo '======================================'
\echo 'ALTER COLUMN DONE.'
\echo ''
\echo 'Next step: run re-embedding (from repo root):'
\echo '  uv run python reembed_workflow/02_reembed.py'
\echo '======================================'
