# Simple Search Script - Quick Guide

## 📋 Deskripsi

Script sederhana untuk **search dan return top 10 knowledge IDs** dari sebuah query.

**Tidak perlu evaluasi, tidak perlu CSV, hanya search dan return knowledge IDs.**

## 🚀 Cara Menggunakan

### Opsi 1: Query dari Command Line

```bash
cd examples/python/retrieval_evaluation
uv run python simple_search.py "Bagaimana cara mengaktifkan notifikasi?"
```

### Opsi 2: Query Hardcoded

Edit file `simple_search.py`, bagian konfigurasi:

```python
# Opsi 2: Query hardcoded (edit di sini)
question = "Bagaimana cara mengaktifkan notifikasi?"
```

Kemudian run:
```bash
cd examples/python/retrieval_evaluation
uv run python simple_search.py
```

## ⚙️ Konfigurasi

Edit bagian konfigurasi di `simple_search.py`:

```python
# Search method: "graph" (GRAPH_COMPLETION) atau "chunks" (CHUNKS)
method = "graph"  # Ganti ke "chunks" jika ingin pakai CHUNKS search

# Target jumlah knowledge IDs
target_n = 10
```

## 📤 Output

Script akan print:
```
================================================================================
SIMPLE SEARCH - TOP 10 KNOWLEDGE IDs
================================================================================
Query: Bagaimana cara mengaktifkan notifikasi?
Method: GRAPH
Target: 10 unique knowledge IDs
================================================================================

📊 Graph status: 1234 nodes, 5678 edges

🔍 Searching using GRAPH_COMPLETION...

================================================================================
✅ SEARCH RESULTS
================================================================================
Found 10 unique knowledge IDs:

  1. doc1.md
  2. doc2.md
  3. doc3.md
  ...
```

## ✅ Requirements

1. ✅ Dokumen sudah di-add dan cognify
2. ✅ Environment variables sudah dikonfigurasi
3. ✅ Database sudah setup

## 🔧 Advanced Usage

### Menggunakan di Python Script Lain

```python
import asyncio
from examples.python.retrieval_evaluation.simple_search import search_knowledge_ids

async def my_function():
    # Search dengan GRAPH_COMPLETION
    knowledge_ids = await search_knowledge_ids(
        question="Bagaimana cara mengaktifkan notifikasi?",
        method="graph",
        target_n=10
    )
    
    print(f"Found {len(knowledge_ids)} knowledge IDs:")
    for kid in knowledge_ids:
        print(f"  - {kid}")

# Run
asyncio.run(my_function())
```

### Menggunakan CHUNKS Search

```python
knowledge_ids = await search_knowledge_ids(
    question="Bagaimana cara mengaktifkan notifikasi?",
    method="chunks",  # Ganti ke "chunks"
    target_n=10
)
```

## 📝 Catatan

- Script ini **hanya untuk search**, tidak ada evaluasi metrics
- Output adalah **knowledge IDs** (TextDocument.name), bukan chunks atau triplets
- Script akan otomatis expand search jika tidak cukup unique IDs
- Default menggunakan **GRAPH_COMPLETION** search (bisa diganti ke CHUNKS)

## 🆚 Perbedaan dengan `run_evaluation.py`

| Feature | `simple_search.py` | `run_evaluation.py` |
|---------|-------------------|---------------------|
| Input | Query (string) | CSV dengan ground truth |
| Output | Top 10 knowledge IDs | Metrics (Hit@k, Precision, Recall) |
| Evaluasi | ❌ Tidak ada | ✅ Ada |
| CSV | ❌ Tidak perlu | ✅ Perlu |
| Use Case | Simple search | Evaluation framework |

## ❓ Troubleshooting

**Error: "Graph is empty"**
→ Pastikan dokumen sudah di-add dan cognify:
```python
await cognee.add(['path/to/doc1.md'])
await cognee.cognify()
```

**Error: "Module not found"**
→ Pastikan folder `retrieval_evaluation/` ada di `examples/python/`

**Tidak dapat 10 knowledge IDs**
→ Mungkin tidak ada cukup dokumen yang relevan. Script akan return sebanyak yang ditemukan.
