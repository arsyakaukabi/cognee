-- ============================================================
-- COGNEE EMBEDDING MIGRATION - ALTER COLUMN WORKAROUND
-- ============================================================
-- 
-- Problem: pgvector tidak bisa langsung ALTER TYPE jika kolom 
-- sudah berisi data dengan dimensi berbeda.
-- 
-- Solution: Set vector ke NULL dulu, lalu ALTER, lalu re-embed.
--
-- ⚠️  PASTIKAN SUDAH BACKUP DATABASE SEBELUM MENJALANKAN INI!
--
-- Jalankan dengan:
--   psql -h 127.0.0.1 -U admin -d bribrain_knowledge_base_hnsw -f scripts/embedding_migration/alter_dimension.sql
-- ============================================================

-- Step 1: Verifikasi kondisi saat ini
\echo '======================================'
\echo 'STEP 1: Verifikasi kondisi saat ini'
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

-- Step 2: Set semua vector ke NULL
\echo ''
\echo '======================================'
\echo 'STEP 2: Set vector column ke NULL'
\echo '======================================'

UPDATE "DocumentChunk_text" SET vector = NULL;
\echo 'DocumentChunk_text: vector set to NULL'

UPDATE "Entity_name" SET vector = NULL;
\echo 'Entity_name: vector set to NULL'

UPDATE "EntityType_name" SET vector = NULL;
\echo 'EntityType_name: vector set to NULL'

UPDATE "EdgeType_relationship_name" SET vector = NULL;
\echo 'EdgeType_relationship_name: vector set to NULL'

UPDATE "TextSummary_text" SET vector = NULL;
\echo 'TextSummary_text: vector set to NULL'

UPDATE "TextDocument_name" SET vector = NULL;
\echo 'TextDocument_name: vector set to NULL'

-- Step 3: ALTER COLUMN TYPE ke vector(3072)
\echo ''
\echo '======================================'
\echo 'STEP 3: ALTER COLUMN ke vector(3072)'
\echo '======================================'

ALTER TABLE "DocumentChunk_text" ALTER COLUMN vector TYPE vector(3072);
\echo 'DocumentChunk_text: ✓ altered to vector(3072)'

ALTER TABLE "Entity_name" ALTER COLUMN vector TYPE vector(3072);
\echo 'Entity_name: ✓ altered to vector(3072)'

ALTER TABLE "EntityType_name" ALTER COLUMN vector TYPE vector(3072);
\echo 'EntityType_name: ✓ altered to vector(3072)'

ALTER TABLE "EdgeType_relationship_name" ALTER COLUMN vector TYPE vector(3072);
\echo 'EdgeType_relationship_name: ✓ altered to vector(3072)'

ALTER TABLE "TextSummary_text" ALTER COLUMN vector TYPE vector(3072);
\echo 'TextSummary_text: ✓ altered to vector(3072)'

ALTER TABLE "TextDocument_name" ALTER COLUMN vector TYPE vector(3072);
\echo 'TextDocument_name: ✓ altered to vector(3072)'

-- Step 4: Verifikasi hasil
\echo ''
\echo '======================================'
\echo 'STEP 4: Verifikasi hasil ALTER'
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
\echo 'ALTER COLUMN SELESAI!'
\echo ''
\echo 'Next step: Jalankan re-embedding script:'
\echo '  python scripts/embedding_migration/02_reembed.py'
\echo '======================================'
