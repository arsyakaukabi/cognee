# 📦 File yang Perlu Dikirim ke Teman

## Opsi 1: Kirim File ZIP/TAR.GZ (Recommended)

File yang sudah dibuat:
- **`retrieval_evaluation.tar.gz`** ← Kirim file ini

Atau buat zip manual:
```bash
cd examples/python
zip -r retrieval_evaluation.zip retrieval_evaluation/
```

## Opsi 2: Kirim Seluruh Folder

Kirimkan **seluruh folder** `retrieval_evaluation/` dengan semua isinya.

---

## 📋 Daftar File yang Dikirim

### Core Python Files (WAJIB)
- `__init__.py`
- `config.py`
- `csv_loader.py`
- `document_manager.py`
- `search_executor.py`
- `result_mapper.py`
- `result_expander.py`
- `chunk_extractor.py`
- `chunk_reranker.py`
- `metrics_calculator.py`
- `evaluator.py`
- `report_generator.py`
- `run_evaluation.py` ← **SCRIPT UTAMA (Full Evaluation)**
- `simple_search.py` ← **SCRIPT SEDERHANA (Hanya Search)**
- `test_result_mapper.py`
- `example_usage.py`

### Dokumentasi (PENTING)
- `README.md`
- `SETUP_GUIDE.md`
- `SHARING_GUIDE.md`
- `INSTRUCTIONS_FOR_FRIEND.md` ← **Beri tahu teman baca ini dulu**
- `QUICK_START.md`
- `SIMPLE_SEARCH_README.md` ← **Untuk simple search mode**
- `PACKAGE_FOR_SHARING.md`
- `EXPANSION_STRATEGY_EXPLAINED.md`
- `TRIPLET_MAPPING_EXPLANATION.md`

### Folder Structure
- `data/` (folder kosong, untuk CSV)
- `results/` (folder kosong, untuk output)

---

## 📧 Pesan untuk Teman Anda

Kirimkan pesan ini bersama file:

```
Halo! Ini adalah Retrieval Evaluation Framework untuk Cognee.

File yang dikirim:
- retrieval_evaluation.tar.gz (atau folder retrieval_evaluation/)

🎯 DUA MODE PENGGUNAAN:

Mode 1: Simple Search (Hanya Search)
- Edit simple_search.py (set query)
- Run: uv run python simple_search.py
- Output: Top 10 knowledge IDs
- Baca: SIMPLE_SEARCH_README.md

Mode 2: Full Evaluation (Dengan CSV dan Metrics)
- Extract folder ke: /path/to/your/cognee/project/examples/python/
- Baca file: INSTRUCTIONS_FOR_FRIEND.md
- Siapkan CSV dengan ground truth
- Edit run_evaluation.py (set CSV_PATH)
- Run: uv run python run_evaluation.py
- Baca: QUICK_START.md dan SETUP_GUIDE.md

Requirements:
- Cognee framework sudah terinstall
- Database sudah setup
- Environment variables sudah dikonfigurasi
- Dokumen sudah di-add dan cognify (untuk simple search)
```

---

## ✅ Checklist Sebelum Kirim

- [ ] File tar.gz/zip sudah dibuat
- [ ] Semua file Python ada di dalamnya
- [ ] Dokumentasi lengkap (README, SETUP_GUIDE, dll)
- [ ] Folder `data/` dan `results/` ada (walaupun kosong)
- [ ] Tidak ada file `__pycache__` atau `.pyc`
- [ ] Tidak ada file hasil evaluasi (results/*.csv, *.json)

---

## 🚀 Cara Teman Anda Menggunakan

1. **Extract** file tar.gz/zip ke project Cognee mereka
2. **Baca** `INSTRUCTIONS_FOR_FRIEND.md` atau `QUICK_START.md`
3. **Siapkan** CSV file dengan ground truth
4. **Edit** `run_evaluation.py` (set CSV_PATH)
5. **Run** evaluasi

---

## 📝 Catatan Penting

1. **Tidak perlu kirim**:
   - File hasil evaluasi (results/*.csv, *.json)
   - File `__pycache__/`
   - File `.pyc`
   - File `.env` (jangan kirim environment variables!)

2. **Yang perlu teman Anda siapkan sendiri**:
   - CSV file dengan ground truth
   - Environment variables (.env)
   - Database setup
   - Dokumen yang akan dievaluasi

3. **Framework ini standalone**:
   - Tidak perlu modifikasi ke Cognee core
   - Hanya perlu import dari `cognee` package
   - Bisa langsung digunakan dengan database mereka
