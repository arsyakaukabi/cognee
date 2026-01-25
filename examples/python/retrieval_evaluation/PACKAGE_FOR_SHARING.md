# Package untuk Sharing - Retrieval Evaluation Framework

## File yang Perlu Dikirim

Kirimkan **seluruh isi folder** `retrieval_evaluation/` ini ke teman Anda.

### Struktur Folder yang Perlu Dikirim

```
retrieval_evaluation/
├── __init__.py
├── config.py
├── csv_loader.py
├── document_manager.py
├── search_executor.py
├── result_mapper.py
├── result_expander.py
├── chunk_extractor.py
├── chunk_reranker.py
├── metrics_calculator.py
├── evaluator.py
├── report_generator.py
├── run_evaluation.py              ← SCRIPT UTAMA
├── test_result_mapper.py
├── example_usage.py
├── README.md
├── SETUP_GUIDE.md
├── SHARING_GUIDE.md               ← Panduan ini
├── EXPANSION_STRATEGY_EXPLAINED.md
├── TRIPLET_MAPPING_EXPLANATION.md
├── data/
│   └── .gitkeep
└── results/
    └── .gitkeep
```

## Cara Kirim

### Opsi 1: Zip Folder
```bash
cd examples/python
zip -r retrieval_evaluation.zip retrieval_evaluation/
# Kirim file retrieval_evaluation.zip
```

### Opsi 2: Git (Jika menggunakan Git)
```bash
# Di project teman Anda:
git clone <your-repo>
# Atau copy folder retrieval_evaluation/ secara manual
```

### Opsi 3: Copy Manual
Copy seluruh folder `retrieval_evaluation/` ke project Cognee teman Anda di:
```
/path/to/their/cognee/project/examples/python/retrieval_evaluation/
```

## Instruksi untuk Teman Anda

### 1. Copy Folder ke Project Mereka

Pastikan folder `retrieval_evaluation/` ada di:
```
/path/to/their/cognee/project/examples/python/retrieval_evaluation/
```

### 2. Siapkan CSV File

Buat file CSV dengan format:
```csv
question,context_ground_truth
"Bagaimana cara mengaktifkan notifikasi?","[""doc1"", ""doc2""]"
"Apa langkah-langkahnya?","[""doc3""]"
```

Simpan sebagai: `examples/python/retrieval_evaluation/data/evaluation_data.csv`

**Penting**: 
- Kolom `question`: Query text
- Kolom `context_ground_truth`: JSON array string dari knowledge IDs
- Knowledge IDs harus sama dengan `TextDocument.name` di graph database mereka

### 3. Edit `run_evaluation.py`

Buka file `examples/python/retrieval_evaluation/run_evaluation.py` dan edit bagian konfigurasi:

```python
# Path ke CSV file (WAJIB)
CSV_PATH = "examples/python/retrieval_evaluation/data/evaluation_data.csv"

# Opsi A: Jika dokumen BELUM ditambahkan
DOCUMENT_PATHS = [
    "path/to/doc1.md",
    "path/to/doc2.md",
]

# Opsi B: Jika dokumen SUDAH ditambahkan
# DOCUMENT_PATHS = None
```

### 4. Pastikan Environment Variables

Pastikan file `.env` di root project sudah ada dengan konfigurasi database dan API keys mereka.

### 5. Jalankan

```bash
cd examples/python/retrieval_evaluation
uv run python run_evaluation.py
```

## Output

Hasil evaluasi akan disimpan di:
```
examples/python/retrieval_evaluation/results/
├── evaluation_results.csv
├── evaluation_results.json
└── context_output.json
```

## Dokumentasi Lengkap

Teman Anda bisa baca:
- `README.md` - Overview dan usage
- `SETUP_GUIDE.md` - Panduan setup detail
- `SHARING_GUIDE.md` - Panduan ini

## Minimal Requirements

1. ✅ Cognee framework sudah terinstall dan berjalan
2. ✅ Database sudah setup (PostgreSQL + PGVector, atau lainnya)
3. ✅ Graph database sudah setup (Kuzu, Neo4j, atau lainnya)
4. ✅ Environment variables sudah dikonfigurasi
5. ✅ CSV file dengan ground truth sudah disiapkan

## Quick Test

Sebelum run evaluasi penuh, bisa test mapping dulu:

```bash
cd examples/python/retrieval_evaluation
uv run python test_result_mapper.py
```

Ini akan test apakah mapping triplets ke knowledge IDs bekerja dengan benar.
