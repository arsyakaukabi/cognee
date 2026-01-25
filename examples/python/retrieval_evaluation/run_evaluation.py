"""
Script untuk menjalankan retrieval evaluation framework.

EDIT KONFIGURASI DI BAWAH SEBELUM RUN!
"""

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
    # Ganti dengan path CSV Anda
    CSV_PATH = "examples/python/retrieval_evaluation/data/evaluation_data.csv"
    
    # Path ke dokumen (OPSIONAL)
    # Opsi 1: Jika dokumen BELUM ditambahkan ke Cognee, isi dengan path dokumen
    DOCUMENT_PATHS = [
        # Contoh:
        # "examples/python/data/wise/data/wi/doc1.md",
        # "examples/python/data/wise/data/wi/doc2.md",
    ]
    
    # Opsi 2: Jika dokumen SUDAH ditambahkan, set ke None
    # DOCUMENT_PATHS = None
    
    # ============================================
    # KONFIGURASI EVALUASI
    # ============================================
    
    # Tentukan apakah dokumen sudah ditambahkan atau belum
    is_read_only = (DOCUMENT_PATHS is None or len(DOCUMENT_PATHS) == 0)
    
    config = EvaluationConfig(
        csv_path=CSV_PATH,
        question_column="question",
        gt_column="context_ground_truth",
        
        # Jika DOCUMENT_PATHS = None atau kosong, gunakan read-only mode
        document_paths=DOCUMENT_PATHS if not is_read_only else None,
        read_only_mode=is_read_only,
        
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
    print(f"Read-only mode: {config.read_only_mode}")
    print(f"Target N unique IDs: {config.target_n_unique_ids}")
    print(f"Output Directory: {config.output_dir}")
    print("=" * 80)
    
    # Validasi CSV path
    if not Path(config.csv_path).exists():
        print(f"\n❌ ERROR: CSV file not found: {config.csv_path}")
        print("   Pastikan path CSV sudah benar!")
        return
    
    # Validasi document paths (jika tidak read-only)
    if not config.read_only_mode and config.document_paths:
        missing_files = [p for p in config.document_paths if not Path(p).exists()]
        if missing_files:
            print(f"\n❌ ERROR: Document files not found:")
            for f in missing_files:
                print(f"   - {f}")
            return
    
    try:
        print("\n🚀 Starting evaluation...")
        
        # Run evaluation
        results = await run_evaluation(config)
        
        # Generate reports
        generate_reports(results, config)
        
        print("\n" + "=" * 80)
        print("✅ EVALUATION COMPLETE!")
        print("=" * 80)
        print(f"📊 Results saved to: {config.output_dir}/")
        print(f"   - evaluation_results.csv")
        print(f"   - evaluation_results.json")
        if config.generate_context_output:
            print(f"   - {config.context_output_file}")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
