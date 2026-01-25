# Instruksi untuk Teman yang Menerima Framework

## 📦 File yang Dikirim

Anda akan menerima:
- **Folder `retrieval_evaluation/`** (atau file zip/tar.gz-nya)
- Folder ini berisi semua file yang diperlukan untuk evaluasi retrieval

## 🎯 Opsi 1: Simple Search (Hanya Search, Tidak Evaluasi)

Jika Anda hanya ingin **search dan dapat top 10 knowledge IDs** tanpa evaluasi:

### Quick Start (3 Langkah)

1. **Edit `simple_search.py`**:
   ```python
   # Edit query di sini
   question = "Bagaimana cara mengaktifkan notifikasi?"
   ```

2. **Run**:
   ```bash
   cd examples/python/retrieval_evaluation
   uv run python simple_search.py
   ```

3. **Hasil**: Top 10 knowledge IDs akan di-print

**Atau dari command line:**
```bash
uv run python simple_search.py "Bagaimana cara mengaktifkan notifikasi?"
```

📖 Baca: `SIMPLE_SEARCH_README.md` untuk detail lebih lanjut.

---

## 🚀 Opsi 2: Full Evaluation (Dengan CSV dan Metrics)

### 1. Copy Folder ke Project Cognee Anda

Copy folder `retrieval_evaluation/` ke:
```
/path/to/your/cognee/project/examples/python/retrieval_evaluation/
```

### 2. Siapkan CSV File dengan Ground Truth

Buat file CSV dengan format:
```csv
question,context_ground_truth
"Bagaimana cara mengaktifkan notifikasi?","[""doc1"", ""doc2""]"
"Apa langkah-langkahnya?","[""doc3""]"
```

**Penting**:
- Kolom `question`: Query text
- Kolom `context_ground_truth`: JSON array string dari knowledge IDs
- Knowledge IDs harus = `TextDocument.name` di graph database Anda

Simpan sebagai: `examples/python/retrieval_evaluation/data/evaluation_data.csv`

### 3. Edit `run_evaluation.py`

Buka file `examples/python/retrieval_evaluation/run_evaluation.py`:

```python
# ============================================
# KONFIGURASI - EDIT BAGIAN INI
# ============================================

# Path ke CSV file (WAJIB)
CSV_PATH = "examples/python/retrieval_evaluation/data/evaluation_data.csv"

# Opsi A: Jika dokumen BELUM ditambahkan ke Cognee
DOCUMENT_PATHS = [
    "path/to/your/doc1.md",
    "path/to/your/doc2.md",
]

# Opsi B: Jika dokumen SUDAH ditambahkan
# DOCUMENT_PATHS = None
```

### 4. Pastikan Environment Variables

Pastikan file `.env` di root project sudah ada dengan konfigurasi:
- Database connection (PostgreSQL, dll)
- API keys (LLM, Embedding)
- `ENABLE_BACKEND_ACCESS_CONTROL=false`

### 5. Jalankan Evaluasi

```bash
cd examples/python/retrieval_evaluation
uv run python run_evaluation.py
```

## 📊 Output

Hasil evaluasi akan disimpan di:
```
examples/python/retrieval_evaluation/results/
├── evaluation_results.csv      ← Per-query results dengan metrics
├── evaluation_results.json      ← Full results dalam JSON
└── context_output.json          ← Reranked chunks untuk LLM
```

## ✅ Checklist Sebelum Run

- [ ] Folder `retrieval_evaluation/` sudah di-copy ke project
- [ ] CSV file sudah disiapkan di `data/evaluation_data.csv`
- [ ] `run_evaluation.py` sudah di-edit (CSV_PATH dan DOCUMENT_PATHS)
- [ ] `.env` sudah dikonfigurasi
- [ ] Cognee framework sudah terinstall dan berjalan
- [ ] Database sudah setup

## 🧪 Test Mapping (Opsional)

Sebelum run evaluasi penuh, bisa test mapping dulu:

```bash
cd examples/python/retrieval_evaluation
uv run python test_result_mapper.py
```

Ini akan test apakah mapping triplets ke knowledge IDs bekerja dengan benar.

## 📚 Dokumentasi

- `README.md` - Overview framework
- `SETUP_GUIDE.md` - Panduan setup detail
- `QUICK_START.md` - Quick start guide
- `SHARING_GUIDE.md` - Panduan sharing

## ❓ Troubleshooting

**Error: "Graph is not ready"**
→ Pastikan dokumen sudah di-add dan cognify, atau set `DOCUMENT_PATHS` di `run_evaluation.py`

**Error: "CSV file not found"**
→ Pastikan path CSV benar di `run_evaluation.py`

**Error: "Knowledge ID not found"**
→ Pastikan knowledge IDs di CSV = `TextDocument.name` di graph database

**Error: "Module not found"**
→ Pastikan folder `retrieval_evaluation/` ada di `examples/python/` di project Cognee

## 💡 Tips

1. **Test dulu dengan 1-2 query** di CSV untuk memastikan setup benar
2. **Cek graph database** untuk memastikan knowledge IDs di CSV ada di `TextDocument.name`
3. **Gunakan read-only mode** jika dokumen sudah ditambahkan sebelumnya
