# Quick Start Guide - Retrieval Evaluation Framework

## 🎯 Pilih Mode Penggunaan

### Mode 1: Simple Search (Hanya Search)
Jika hanya ingin **search dan dapat top 10 knowledge IDs**:
→ Baca: `SIMPLE_SEARCH_README.md`

### Mode 2: Full Evaluation (Dengan CSV dan Metrics)
Jika ingin **evaluasi dengan ground truth dan metrics**:
→ Lanjutkan ke bawah

---

## Untuk Teman yang Menerima Framework Ini (Full Evaluation Mode)

### Step 1: Copy Folder

Copy folder `retrieval_evaluation/` ke project Cognee Anda:
```
/path/to/your/cognee/project/examples/python/retrieval_evaluation/
```

### Step 2: Siapkan CSV

Buat file CSV dengan ground truth:
```csv
question,context_ground_truth
"Query 1","[""doc1"", ""doc2""]"
"Query 2","[""doc3""]"
```

Simpan di: `examples/python/retrieval_evaluation/data/evaluation_data.csv`

### Step 3: Edit `run_evaluation.py`

Buka `examples/python/retrieval_evaluation/run_evaluation.py`:

```python
# Ganti path CSV
CSV_PATH = "examples/python/retrieval_evaluation/data/evaluation_data.csv"

# Jika dokumen belum ditambahkan, isi path dokumen:
DOCUMENT_PATHS = ["path/to/doc1.md", "path/to/doc2.md"]

# Jika dokumen sudah ditambahkan, set:
# DOCUMENT_PATHS = None
```

### Step 4: Run

```bash
cd examples/python/retrieval_evaluation
uv run python run_evaluation.py
```

### Step 5: Cek Hasil

Hasil ada di: `examples/python/retrieval_evaluation/results/`

---

## File yang Dikirim

Kirimkan **seluruh folder** `retrieval_evaluation/` (atau file zip/tar.gz-nya).

## Requirements

- Cognee framework sudah terinstall
- Database sudah setup
- Environment variables sudah dikonfigurasi
- CSV file dengan ground truth

## Bantuan

Baca dokumentasi:
- `README.md` - Overview
- `SETUP_GUIDE.md` - Setup detail
- `SHARING_GUIDE.md` - Panduan sharing
