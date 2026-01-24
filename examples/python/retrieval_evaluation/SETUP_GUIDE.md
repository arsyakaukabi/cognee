# Setup Guide - Retrieval Evaluation Framework

## Urutan Langkah untuk Menjalankan Evaluasi

### 1. Persiapan Data CSV

**Lokasi**: Simpan CSV di mana saja, catat path-nya.

**Format CSV**:
```csv
generator,style,case_type,question,context_ground_truth,gt_categories
RAFT,Bahasa indonesia formal,positive,"Bagaimana cara mengaktifkan notifikasi transaksi di aplikasi BRImo?","[""22y7j4AeCpcVQqdZ98bjra"", ""CXF3s535h4wvcsQ85Pgg9C""]","['helpdesk', 'working instruction']"
RAFT,Bahasa indonesia formal,positive,"gmana cara mengaktifkan notifikasi transaksi di plikasi BRImo?","[""22y7j4AeCpcVQqdZ98bjra"", ""CXF3s535h4wvcsQ85Pgg9C""]","['helpdesk', 'working instruction']"
```

**Kolom Penting**:
- `question`: Query text untuk search
- `context_ground_truth`: JSON array string dari knowledge IDs (ground truth)
  - Format: `'["id1", "id2", "id3"]'`
  - Knowledge ID = `TextDocument.name` di graph

**Contoh Lokasi**:
```
examples/python/retrieval_evaluation/data/evaluation_data.csv
```

### 2. Persiapan Dokumen (Opsional)

**Jika dokumen belum ditambahkan ke Cognee**:

Simpan dokumen `.md` di folder tertentu, contoh:
```
examples/python/data/wise/data/wi/
├── doc1.md
├── doc2.md
└── ...
```

**Jika dokumen sudah ditambahkan**:
- Skip langkah ini
- Gunakan `read_only_mode=True` di config

### 3. Setup Environment Variables

Pastikan file `.env` sudah ada di root project dengan konfigurasi:
```bash
LLM_API_KEY=your_key
LLM_ENDPOINT=your_endpoint
EMBEDDING_MODEL=azure/text-embedding-3-small
EMBEDDING_DIMENSIONS=1536
EMBEDDING_PROVIDER=openai
EMBEDDING_ENDPOINT=your_embedding_endpoint
EMBEDDING_API_KEY=your_key
ENABLE_BACKEND_ACCESS_CONTROL=false
```

### 4. Buat Script Runner

Buat file baru untuk menjalankan evaluasi, contoh:

**File**: `examples/python/retrieval_evaluation/run_evaluation.py`

```python
import asyncio
import os
from pathlib import Path

# Load environment
import dotenv
dotenv.load_dotenv(override=True)

os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "false"

from examples.python.retrieval_evaluation.config import EvaluationConfig
from examples.python.retrieval_evaluation.evaluator import run_evaluation
from examples.python.retrieval_evaluation.report_generator import generate_reports


async def main():
    # ============================================
    # KONFIGURASI - EDIT BAGIAN INI
    # ============================================
    
    # Path ke CSV file (WAJIB)
    CSV_PATH = "examples/python/retrieval_evaluation/data/evaluation_data.csv"
    
    # Path ke dokumen (OPSIONAL - hanya jika dokumen belum ditambahkan)
    DOCUMENT_PATHS = [
        # "examples/python/data/wise/data/wi/doc1.md",
        # "examples/python/data/wise/data/wi/doc2.md",
        # ... tambahkan path dokumen di sini
    ]
    
    # Atau gunakan None jika dokumen sudah ditambahkan
    # DOCUMENT_PATHS = None
    
    # ============================================
    # KONFIGURASI EVALUASI
    # ============================================
    
    config = EvaluationConfig(
        csv_path=CSV_PATH,
        question_column="question",
        gt_column="context_ground_truth",
        
        # Jika DOCUMENT_PATHS = None, set read_only_mode=True
        # Jika DOCUMENT_PATHS ada isinya, set read_only_mode=False
        document_paths=DOCUMENT_PATHS,
        read_only_mode=(DOCUMENT_PATHS is None or len(DOCUMENT_PATHS) == 0),
        
        # Target jumlah unique knowledge IDs yang ingin di-retrieve
        target_n_unique_ids=10,
        
        # Initial search top_k (akan di-expand otomatis jika perlu)
        initial_top_k=10,
        max_expansion_top_k=200,
        
        # Output directory
        output_dir="examples/python/retrieval_evaluation/results",
        
        # Generate context output untuk LLM (reranked chunks)
        generate_context_output=True,
        context_output_file="context_output.json",
    )
    
    # ============================================
    # JALANKAN EVALUASI
    # ============================================
    
    print("=" * 80)
    print("RETRIEVAL EVALUATION FRAMEWORK")
    print("=" * 80)
    print(f"CSV Path: {config.csv_path}")
    print(f"Document Paths: {config.document_paths or 'None (read-only mode)'}")
    print(f"Target N unique IDs: {config.target_n_unique_ids}")
    print(f"Output Directory: {config.output_dir}")
    print("=" * 80)
    
    try:
        # Run evaluation
        results = await run_evaluation(config)
        
        # Generate reports
        generate_reports(results, config)
        
        print("\n✅ Evaluation complete!")
        print(f"📊 Results saved to: {config.output_dir}/")
        print(f"   - evaluation_results.csv")
        print(f"   - evaluation_results.json")
        if config.generate_context_output:
            print(f"   - {config.context_output_file}")
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
```

### 5. Struktur Folder yang Disarankan

```
examples/python/retrieval_evaluation/
├── data/
│   └── evaluation_data.csv          # CSV dengan ground truth
├── results/                          # Output akan disimpan di sini
│   ├── evaluation_results.csv
│   ├── evaluation_results.json
│   └── context_output.json
├── run_evaluation.py                 # Script untuk run (buat file ini)
├── example_usage.py                 # Contoh (sudah ada)
└── ... (file-file framework lainnya)
```

### 6. Langkah-Langkah Menjalankan

#### Opsi A: Dokumen Belum Ditambahkan

1. **Siapkan CSV**:
   ```bash
   # Buat folder data jika belum ada
   mkdir -p examples/python/retrieval_evaluation/data
   
   # Simpan CSV di: examples/python/retrieval_evaluation/data/evaluation_data.csv
   ```

2. **Edit `run_evaluation.py`**:
   - Set `CSV_PATH` ke path CSV Anda
   - Set `DOCUMENT_PATHS` dengan path dokumen `.md`
   - Set `read_only_mode=False`

3. **Jalankan**:
   ```bash
   cd examples/python/retrieval_evaluation
   uv run python run_evaluation.py
   ```

#### Opsi B: Dokumen Sudah Ditambahkan (Read-Only Mode)

1. **Siapkan CSV**:
   ```bash
   mkdir -p examples/python/retrieval_evaluation/data
   # Simpan CSV di: examples/python/retrieval_evaluation/data/evaluation_data.csv
   ```

2. **Edit `run_evaluation.py`**:
   - Set `CSV_PATH` ke path CSV Anda
   - Set `DOCUMENT_PATHS = None`
   - Set `read_only_mode=True`

3. **Jalankan**:
   ```bash
   cd examples/python/retrieval_evaluation
   uv run python run_evaluation.py
   ```

### 7. Output yang Dihasilkan

Setelah selesai, di folder `results/` akan ada:

1. **evaluation_results.csv**: 
   - Per-query results dengan semua metrics
   - Satu row per query per method (graph_completion dan chunk)

2. **evaluation_results.json**:
   - Full results dalam format JSON
   - Termasuk aggregated metrics

3. **context_output.json** (jika `generate_context_output=True`):
   - Reranked chunks untuk LLM consumption
   - Format: `[{"query": "...", "knowledge_ids": [...], "reranked_chunks": [...]}]`

### 8. Checklist Sebelum Run

- [ ] CSV file sudah disiapkan dengan format yang benar
- [ ] Kolom `question` dan `context_ground_truth` ada di CSV
- [ ] Knowledge IDs di `context_ground_truth` = `TextDocument.name` di graph
- [ ] Jika dokumen belum ditambahkan: path dokumen sudah benar
- [ ] Jika dokumen sudah ditambahkan: `read_only_mode=True`
- [ ] Environment variables sudah di-set di `.env`
- [ ] Script `run_evaluation.py` sudah dibuat dan dikonfigurasi
- [ ] Folder `results/` akan dibuat otomatis (atau sudah ada)

### 9. Troubleshooting

**Error: "CSV file not found"**
- Pastikan path CSV benar (relative atau absolute)
- Cek apakah file benar-benar ada

**Error: "Graph is not ready"**
- Jika `read_only_mode=True`: pastikan dokumen sudah di-add dan cognify sebelumnya
- Jika `read_only_mode=False`: pastikan `document_paths` sudah di-set

**Error: "No ground truth IDs found"**
- Cek format `context_ground_truth` di CSV (harus JSON array string)
- Contoh format: `'["id1", "id2"]'`

**Error: "Knowledge ID not found in graph"**
- Pastikan knowledge ID di CSV = `TextDocument.name` di graph
- Cek apakah dokumen sudah di-cognify dengan benar

### 10. Contoh CSV Minimal

Jika CSV Anda sederhana, cukup 2 kolom:

```csv
question,context_ground_truth
"Bagaimana cara mengaktifkan notifikasi?","[""doc1"", ""doc2""]"
"Apa langkah-langkahnya?","[""doc3""]"
```

Simpan sebagai: `examples/python/retrieval_evaluation/data/evaluation_data.csv`
