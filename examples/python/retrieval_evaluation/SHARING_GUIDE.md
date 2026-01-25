# Sharing Guide - Retrieval Evaluation Framework

## File yang Perlu Dikirim

Kirimkan **seluruh folder** `retrieval_evaluation/` ke teman Anda:

```
examples/python/retrieval_evaluation/
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
├── run_evaluation.py          ← Script utama untuk run
├── test_result_mapper.py       ← Script untuk test mapping
├── example_usage.py            ← Contoh penggunaan
├── README.md
├── SETUP_GUIDE.md
├── EXPANSION_STRATEGY_EXPLAINED.md
├── TRIPLET_MAPPING_EXPLANATION.md
├── data/                       ← Folder untuk CSV
│   └── .gitkeep
└── results/                    ← Folder untuk output
    └── .gitkeep
```

**Cara Kirim:**
1. Zip folder `retrieval_evaluation/` atau
2. Copy seluruh folder ke tempat teman Anda

## Setup untuk Teman Anda

### 1. Pastikan Cognee Framework Sudah Terinstall

Teman Anda harus sudah punya Cognee framework yang berjalan dengan:
- Database sudah setup (PostgreSQL + PGVector, atau database lain)
- Graph database sudah setup (Kuzu, Neo4j, atau lainnya)
- Environment variables sudah dikonfigurasi

### 2. Copy Folder

Copy folder `retrieval_evaluation/` ke project Cognee mereka:
```
/path/to/their/cognee/project/
└── examples/python/
    └── retrieval_evaluation/  ← Copy folder ini
```

### 3. Siapkan CSV File

Teman Anda perlu menyiapkan CSV dengan format:
```csv
question,context_ground_truth
"Bagaimana cara mengaktifkan notifikasi?","[""doc1"", ""doc2""]"
"Apa langkah-langkahnya?","[""doc3""]"
```

Simpan di: `examples/python/retrieval_evaluation/data/evaluation_data.csv`

**Penting**: 
- `context_ground_truth` harus JSON array string
- Knowledge IDs di CSV harus sama dengan `TextDocument.name` di graph database

### 4. Edit Script `run_evaluation.py`

Buka `examples/python/retrieval_evaluation/run_evaluation.py` dan edit:

```python
# ============================================
# KONFIGURASI - EDIT BAGIAN INI
# ============================================

# Path ke CSV file (WAJIB)
CSV_PATH = "examples/python/retrieval_evaluation/data/evaluation_data.csv"

# Path ke dokumen (OPSIONAL - hanya jika dokumen belum ditambahkan)
DOCUMENT_PATHS = [
    # "path/to/doc1.md",
    # "path/to/doc2.md",
]

# Atau set ke None jika dokumen sudah ditambahkan
# DOCUMENT_PATHS = None
```

### 5. Pastikan Environment Variables

Pastikan file `.env` di root project sudah ada dengan konfigurasi:
```bash
LLM_API_KEY=your_key
LLM_ENDPOINT=your_endpoint
EMBEDDING_MODEL=azure/text-embedding-3-small
EMBEDDING_DIMENSIONS=1536
EMBEDDING_PROVIDER=openai
EMBEDDING_ENDPOINT=your_embedding_endpoint
EMBEDDING_API_KEY=your_key
ENABLE_BACKEND_ACCESS_CONTROL=false
# ... database config lainnya
```

## Cara Menjalankan

### Opsi 1: Dokumen Belum Ditambahkan

1. Edit `run_evaluation.py`:
   - Set `CSV_PATH` ke path CSV mereka
   - Set `DOCUMENT_PATHS` dengan path dokumen mereka
   - Script akan otomatis add & cognify

2. Jalankan:
   ```bash
   cd examples/python/retrieval_evaluation
   uv run python run_evaluation.py
   ```

### Opsi 2: Dokumen Sudah Ditambahkan (Read-Only Mode)

1. Edit `run_evaluation.py`:
   - Set `CSV_PATH` ke path CSV mereka
   - Set `DOCUMENT_PATHS = None`
   - Script akan skip add/cognify, langsung evaluasi

2. Jalankan:
   ```bash
   cd examples/python/retrieval_evaluation
   uv run python run_evaluation.py
   ```

## Output yang Dihasilkan

Setelah selesai, hasil akan disimpan di:
```
examples/python/retrieval_evaluation/results/
├── evaluation_results.csv      ← Per-query results dengan metrics
├── evaluation_results.json      ← Full results dalam JSON
└── context_output.json          ← Reranked chunks untuk LLM (jika enabled)
```

## Troubleshooting

### Error: "Graph is not ready"
- **Solusi**: Pastikan dokumen sudah di-add dan cognify, atau set `DOCUMENT_PATHS` di `run_evaluation.py`

### Error: "CSV file not found"
- **Solusi**: Pastikan path CSV benar (relative atau absolute path)

### Error: "No ground truth IDs found"
- **Solusi**: Cek format `context_ground_truth` di CSV (harus JSON array string)

### Error: "Knowledge ID not found in graph"
- **Solusi**: Pastikan knowledge IDs di CSV = `TextDocument.name` di graph database

### Error: "Module not found"
- **Solusi**: Pastikan folder `retrieval_evaluation/` ada di `examples/python/` di project Cognee mereka

## Quick Start Checklist

- [ ] Copy folder `retrieval_evaluation/` ke project Cognee
- [ ] Siapkan CSV file dengan ground truth
- [ ] Edit `run_evaluation.py` (set CSV_PATH dan DOCUMENT_PATHS)
- [ ] Pastikan `.env` sudah dikonfigurasi
- [ ] Pastikan Cognee framework sudah terinstall
- [ ] Jalankan: `uv run python run_evaluation.py`
- [ ] Cek hasil di folder `results/`

## Dependencies

Framework ini menggunakan:
- `cognee` (framework utama)
- `pandas` (untuk CSV handling)
- `numpy` (untuk similarity calculation)
- Standard library: `asyncio`, `pathlib`, `typing`, `dataclasses`

Semua dependencies seharusnya sudah terinstall jika Cognee framework sudah setup.
