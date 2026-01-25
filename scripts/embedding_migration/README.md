# Cognee Embedding Migration Scripts

Folder ini berisi script untuk migrasi model embedding dari `text-embedding-ada-002` (1536 dimensions) ke `text-embedding-3-large` (3072 dimensions).

## 📁 Struktur Folder

```
scripts/embedding_migration/
├── 01_summary.py       # Script untuk melihat summary tabel terdampak
├── 02_reembed.py       # Script untuk re-embedding dengan resume capability
├── 03_verify.py        # Script untuk verifikasi hasil migrasi
├── README.md           # Dokumentasi ini
├── progress.json       # [Auto-generated] Progress tracking
└── reembed_*.log       # [Auto-generated] Log files
```

---

## 🚀 Cara Menjalankan

### Prerequisites

1. Pastikan virtual environment aktif:
   ```bash
   source .venv/bin/activate  # atau sesuai setup Anda
   ```

2. Pastikan `.env` sudah dikonfigurasi dengan endpoint embedding baru:
   ```env
   EMBEDDING_MODEL="azure/text-embedding-3-large"
   EMBEDDING_ENDPOINT="https://your-endpoint.openai.azure.com/..."
   EMBEDDING_DIMENSIONS="3072"
   ```

3. Pastikan PostgreSQL dan pgvector extension sudah running.

---

## 📋 Step-by-Step Migration

### Step 1: Cek Summary Tabel Terdampak

Jalankan script summary untuk melihat kondisi saat ini:

```bash
python scripts/embedding_migration/01_summary.py
```

**Output yang diharapkan:**
- Daftar 6 tabel yang menyimpan vector embedding
- Jumlah record per tabel
- Dimensi vector saat ini (1536 atau 3072)
- Konfigurasi embedding engine yang aktif

### Step 2: Alter Table Dimensions

> ⚠️ **PENTING**: Backup database terlebih dahulu!

```bash
# Backup
pg_dump -h 127.0.0.1 -U admin -d bribrain_knowledge_base_hnsw \
    -F c -f backup_pre_migration_$(date +%Y%m%d).dump
```

> ⚠️ **CATATAN**: pgvector TIDAK BISA langsung ALTER TYPE jika kolom sudah berisi data dengan dimensi berbeda. 
> Gunakan script SQL yang sudah disediakan:

**Jalankan script SQL untuk ALTER column:**

```bash
PGPASSWORD=admin psql -h 127.0.0.1 -U admin -d bribrain_knowledge_base_hnsw \
    -f scripts/embedding_migration/alter_dimension.sql
```

Script ini akan:
1. Set semua vector column ke NULL
2. ALTER COLUMN TYPE ke vector(3072)
3. Verifikasi hasil

**Atau jalankan manual step-by-step:**

```sql
-- Set vector ke NULL dulu (WAJIB sebelum ALTER)
UPDATE "DocumentChunk_text" SET vector = NULL;
UPDATE "Entity_name" SET vector = NULL;
UPDATE "EntityType_name" SET vector = NULL;
UPDATE "EdgeType_relationship_name" SET vector = NULL;
UPDATE "TextSummary_text" SET vector = NULL;
UPDATE "TextDocument_name" SET vector = NULL;

-- Baru ALTER TYPE
ALTER TABLE "DocumentChunk_text" ALTER COLUMN vector TYPE vector(3072);
ALTER TABLE "Entity_name" ALTER COLUMN vector TYPE vector(3072);
ALTER TABLE "EntityType_name" ALTER COLUMN vector TYPE vector(3072);
ALTER TABLE "EdgeType_relationship_name" ALTER COLUMN vector TYPE vector(3072);
ALTER TABLE "TextSummary_text" ALTER COLUMN vector TYPE vector(3072);
ALTER TABLE "TextDocument_name" ALTER COLUMN vector TYPE vector(3072);
```

### Step 3: Re-embed Data

Jalankan script re-embedding:

```bash
# Fresh start
python scripts/embedding_migration/02_reembed.py

# Dengan batch size custom (default: 50)
python scripts/embedding_migration/02_reembed.py --batch-size 100

# Dry run (simulasi tanpa update database)
python scripts/embedding_migration/02_reembed.py --dry-run
```

**Jika gagal di tengah jalan**, resume dari progress terakhir:

```bash
python scripts/embedding_migration/02_reembed.py --resume
```

**Untuk re-embed tabel tertentu saja:**

```bash
python scripts/embedding_migration/02_reembed.py --table DocumentChunk_text
```

**Files yang dihasilkan:**
- `progress.json` - Tracking progress (last processed ID per table)
- `reembed_YYYYMMDD_HHMMSS.log` - Log detail proses

### Step 4: Verifikasi Hasil

Jalankan script verifikasi:

```bash
python scripts/embedding_migration/03_verify.py
```

**Verifikasi yang dilakukan:**
1. ✅ Dimensi vector sudah 3072
2. ✅ Embedding engine terkonfigurasi dengan benar
3. ✅ Search functionality berfungsi
4. ✅ Tidak ada null vectors

---

## 📊 Summary Tabel yang Terdampak

| Tabel | Source Field | Deskripsi |
|-------|--------------|-----------|
| `DocumentChunk_text` | `text` | Chunk dokumen untuk RAG |
| `Entity_name` | `name` | Entity dari knowledge graph |
| `EntityType_name` | `name` | Tipe entity |
| `EdgeType_relationship_name` | `relationship_name` | Tipe relasi/edge |
| `TextSummary_text` | `text` | Ringkasan dokumen |
| `TextDocument_name` | `name` | Metadata dokumen asli |

---

## ⏱️ Estimasi Waktu

| Jumlah Records | Estimasi Waktu |
|----------------|----------------|
| ~1,000 | 10-15 menit |
| ~3,000 | 30-45 menit |
| ~10,000 | 1.5-2 jam |

Catatan: Waktu bergantung pada rate limit Azure OpenAI dan kecepatan jaringan.

---

## 🔄 Rollback Plan

Jika migrasi gagal dan perlu rollback:

```bash
# Restore dari backup
pg_restore -h 127.0.0.1 -U admin -d bribrain_knowledge_base_hnsw \
    -c backup_pre_migration_XXXXXXXX.dump
```

Kemudian update `.env` kembali ke config lama:
```env
EMBEDDING_MODEL="azure/text-embedding-ada-002"
EMBEDDING_DIMENSIONS="1536"
```

---

## 📝 Troubleshooting

### Error: "Embedding engine error"
- Cek koneksi ke Azure OpenAI endpoint
- Verifikasi API key masih valid
- Pastikan deployment `text-embedding-3-large` sudah tersedia

### Error: "vector dimension mismatch"
- Pastikan sudah menjalankan ALTER TABLE di Step 2
- Verifikasi dengan: `\d "DocumentChunk_text"` di psql

### Script berhenti di tengah
- Jalankan kembali dengan `--resume`:
  ```bash
  python scripts/embedding_migration/02_reembed.py --resume
  ```
- Cek `progress.json` untuk melihat status terakhir

### Rate limit exceeded
- Kurangi batch size: `--batch-size 20`
- Script sudah memiliki delay 0.3s antar batch

---

## 📄 Log Files

Log files disimpan dengan format: `reembed_YYYYMMDD_HHMMSS.log`

Contoh isi log:
```
2026-01-20 12:00:00 | INFO     | Processing: DocumentChunk_text
2026-01-20 12:00:01 | INFO     | Batch 1: ✓ Updated 50 records. Total: 50/104
2026-01-20 12:00:02 | INFO     | Batch 2: ✓ Updated 50 records. Total: 100/104
2026-01-20 12:00:03 | INFO     | Batch 3: ✓ Updated 4 records. Total: 104/104
```

---

## ✅ Checklist Migrasi

- [ ] Backup database
- [ ] Jalankan `01_summary.py` - catat kondisi awal
- [ ] Update `.env` dengan config embedding baru
- [ ] ALTER TABLE untuk semua 6 tabel
- [ ] Jalankan `02_reembed.py`
- [ ] Jalankan `03_verify.py` - semua check ✅
- [ ] Test search functionality di aplikasi
- [ ] Hapus backup setelah yakin sukses
