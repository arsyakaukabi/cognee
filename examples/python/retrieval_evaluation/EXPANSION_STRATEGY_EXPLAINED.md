# Strategi Expansion untuk Memastikan 10 Unique Knowledge IDs

## Masalah

Dari triplet search, kita dapat triplets (misalnya top-10 triplets), tapi setelah mapping ke knowledge IDs, mungkin hanya dapat 5 unique knowledge IDs. Kita perlu **selalu mendapatkan 10 unique knowledge IDs**.

## Solusi: Expansion Strategy

### Alur Expansion

```
1. Initial Search (top_k=10)
   → Dapat 10 triplets
   → Map ke knowledge IDs
   → Dapat 5 unique knowledge IDs ❌ (kurang dari 10)

2. Expansion Iteration 1 (top_k=20)
   → Dapat 20 triplets
   → Map ke knowledge IDs
   → Dapat 8 unique knowledge IDs ❌ (masih kurang)

3. Expansion Iteration 2 (top_k=40)
   → Dapat 40 triplets
   → Map ke knowledge IDs
   → Dapat 12 unique knowledge IDs ✅ (sudah cukup!)

4. Return top 10 unique knowledge IDs
```

### Implementasi di `result_expander.py`

```python
async def expand_graph_completion_to_n_ids(
    triplets: List[Edge],
    target_n: int,  # Target: 10 unique knowledge IDs
    ...
) -> List[str]:
    # 1. Map initial triplets
    knowledge_ids = await map_triplet_results_to_knowledge_ids(triplets, graph_engine)
    unique_ids = list(dict.fromkeys(knowledge_ids))
    
    # 2. Jika sudah cukup, return
    if len(unique_ids) >= target_n:
        return unique_ids[:target_n]
    
    # 3. Expansion loop
    current_top_k = initial_top_k  # Start dari 10
    expansion_factor = 2  # Exponential: 10 → 20 → 40 → 80 → 160
    
    while len(unique_ids) < target_n and current_top_k < max_top_k:
        # Increase top_k
        current_top_k = min(current_top_k * expansion_factor, max_top_k)
        
        # Fetch more triplets
        new_triplets = await search_executor.execute_graph_completion_search(
            query, top_k=current_top_k
        )
        
        # Map to knowledge IDs
        new_knowledge_ids = await map_triplet_results_to_knowledge_ids(
            new_triplets, graph_engine
        )
        
        # Add new unique IDs
        for kid in new_knowledge_ids:
            if kid not in unique_ids:
                unique_ids.append(kid)
        
        # Check if enough
        if len(unique_ids) >= target_n:
            break
    
    # 4. Return exactly N
    return unique_ids[:target_n]
```

## Parameter Konfigurasi

Di `config.py`:

```python
@dataclass
class EvaluationConfig:
    target_n_unique_ids: int = 10  # Target: selalu dapat 10 unique knowledge IDs
    initial_top_k: int = 50  # Start dari 50 triplets/chunks (increased to reduce expansion)
    max_expansion_top_k: int = 200  # Maximum expansion (prevent infinite loop)
    graph_completion_initial_top_k: int = 50  # Initial top_k for Graph Completion
    chunk_initial_top_k: int = 50  # Initial top_k for Chunk search
```

**Note**: `initial_top_k` dinaikkan dari 10 ke 50 untuk mengurangi jumlah expansion iterations.

## Expansion Strategy Details

### Expansion Strategy (Optimized)

**Initial**: Start dengan `top_k = 50` (bukan 10) untuk mengurangi expansion iterations.

```
Initial: top_k = 50  → Map → 8 unique IDs ❌
Iteration 1: top_k = 75  → Map → 12 unique IDs ✅
```

**Formula**: `current_top_k = previous_top_k * 1.5` (smaller factor karena start tinggi)

**Sebelum optimasi** (start dari 10):
```
Iteration 1: top_k = 10  → Map → 5 unique IDs ❌
Iteration 2: top_k = 20  → Map → 8 unique IDs ❌
Iteration 3: top_k = 40  → Map → 12 unique IDs ✅
```

**Setelah optimasi** (start dari 50):
```
Initial: top_k = 50  → Map → 8 unique IDs ❌
Iteration 1: top_k = 75  → Map → 12 unique IDs ✅
```

**Keuntungan**: Hanya perlu 1 expansion iteration (bukan 3), lebih cepat dan efisien!

### Safety Limits

1. **max_top_k**: Maximum top_k untuk expansion (default: 200)
   - Mencegah infinite loop
   - Jika sudah mencapai max_top_k tapi masih kurang, return yang ada

2. **max_iterations**: Maximum expansion iterations (default: 10)
   - Mencegah infinite loop jika tidak ada progress

3. **No new unique IDs check**: Jika tidak ada new unique IDs, stop expansion
   - Menandakan sudah tidak ada knowledge IDs baru yang bisa ditemukan

## Contoh Skenario

### Skenario 1: Cukup dari Initial Search (Most Common)

```
Query: "Bagaimana cara mengaktifkan notifikasi?"

Initial Search (top_k=50):
  → 50 triplets
  → Map ke knowledge IDs
  → 15 unique knowledge IDs ✅

Result: Return top 10 unique knowledge IDs
```

### Skenario 2: Perlu 1 Expansion (Optimized)

```
Query: "Apa langkah-langkahnya?"

Initial Search (top_k=50):
  → 50 triplets
  → Map ke knowledge IDs
  → 7 unique knowledge IDs ❌ (kurang)

Expansion Iteration 1 (top_k=75):
  → 75 triplets
  → Map ke knowledge IDs
  → 12 unique knowledge IDs ✅ (sudah cukup!)

Result: Return top 10 unique knowledge IDs
```

**Note**: Dengan `initial_top_k=50`, kebanyakan query akan langsung dapat 10 unique IDs tanpa perlu expansion!

### Skenario 3: Tidak Cukup (Edge Case)

```
Query: "Very specific query with limited results"

Initial Search (top_k=10):
  → 10 triplets
  → Map ke knowledge IDs
  → 3 unique knowledge IDs ❌

Expansion Iteration 1-5:
  → top_k: 20, 40, 80, 160, 200
  → Map ke knowledge IDs
  → 6 unique knowledge IDs ❌ (masih kurang, tapi sudah max)

Result: Return 6 unique knowledge IDs (dengan warning)
```

## Logging

Expansion process akan di-log untuk debugging:

```
INFO: Initial mapping: 10 triplets → 5 unique knowledge IDs (target: 10)
INFO: Expanding Graph Completion search (iteration 1): top_k=10 → 20, current_unique_ids=5, target=10
INFO:    → Retrieved 20 triplets → 15 knowledge IDs → 3 new unique IDs → Total: 8 unique IDs
INFO: Expanding Graph Completion search (iteration 2): top_k=20 → 40, current_unique_ids=8, target=10
INFO:    → Retrieved 40 triplets → 25 knowledge IDs → 4 new unique IDs → Total: 12 unique IDs
INFO: ✅ Reached target: 12 unique knowledge IDs (target: 10)
INFO: Final result: 10 unique knowledge IDs
```

## Optimasi

### Mengapa Exponential Backoff?

1. **Efisien**: Tidak perlu expand sedikit-sedikit (10 → 11 → 12 → ...)
2. **Cepat mencapai target**: 10 → 20 → 40 → 80 (4 iterations max untuk mencapai 200)
3. **Balance**: Tidak terlalu agresif (linear: 10 → 20 → 30 → ... terlalu lambat)

### Alternatif Strategy (Jika Perlu)

Jika exponential terlalu agresif, bisa gunakan linear:

```python
expansion_factor = 1.5  # Linear: 10 → 15 → 22 → 33 → ...
# atau
current_top_k = initial_top_k + (iteration * 10)  # 10 → 20 → 30 → 40 → ...
```

## Kesimpulan

✅ **Expansion strategy memastikan selalu mendapatkan 10 unique knowledge IDs**

- Start dari `initial_top_k=10`
- Expand secara exponential (x2) sampai dapat 10 unique knowledge IDs
- Safety limits mencegah infinite loop
- Return exactly 10 (atau semua yang ada jika tidak cukup)

**Hasil**: Selalu dapat 10 unique knowledge IDs (atau maksimal yang tersedia jika kurang dari 10).
