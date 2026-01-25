# Penjelasan Triplet Mapping ke Knowledge IDs

## Pertanyaan: Dari node mana kita cari TextDocument?

**Jawaban: KEDUA-DUANYA (node1 DAN node2)**

## Alur Mapping Triplet

### Struktur Triplet
```
Triplet = (node1) -[edge/relationship]-> (node2)
```

Contoh:
- `(Entity: "Notifikasi") -[contains]-> (Entity: "Aplikasi")`
- `(Entity: "Python") -[is_a]-> (EntityType: "Programming Language")`

### Proses Mapping

Untuk **setiap triplet**, kita memetakan **KEDUA node** ke knowledge IDs:

```python
# Dari result_mapper.py line 262-284
for triplet in triplets:
    # 1. Map node1 ke knowledge IDs
    if node1_id and node1_type:
        mapped_ids = await map_node_to_knowledge_ids(node1_id, node1_type, graph_engine)
        # Tambahkan ke list knowledge_ids
    
    # 2. Map node2 ke knowledge IDs  
    if node2_id and node2_type:
        mapped_ids = await map_node_to_knowledge_ids(node2_id, node2_type, graph_engine)
        # Tambahkan ke list knowledge_ids
```

## Mengapa Kedua Node?

### Alasan 1: Satu Triplet = Dua Entitas yang Relevan
- **node1** dan **node2** sama-sama relevan dengan query
- Keduanya muncul dalam triplet karena ada hubungan yang relevan
- Jadi, **kedua node** harus dipertimbangkan untuk mencari TextDocument

### Alasan 2: Coverage yang Lebih Baik
- Jika hanya node1: bisa kehilangan dokumen yang hanya muncul di node2
- Jika hanya node2: bisa kehilangan dokumen yang hanya muncul di node1
- Dengan kedua node: coverage lebih lengkap

### Alasan 3: Preserve Ranking
- Triplet diurutkan berdasarkan relevansi
- Node1 dan node2 dalam triplet yang sama memiliki relevansi yang sama
- Jadi, keduanya harus dipertimbangkan

## Contoh Mapping

### Contoh 1: Entity - Entity
```
Triplet: (Entity: "Notifikasi") -[contains]-> (Entity: "Aplikasi")

Mapping:
1. Node1 (Entity: "Notifikasi"):
   - Cari DocumentChunk yang contains "Notifikasi"
   - Cari TextDocument dari DocumentChunk tersebut
   - Hasil: ["doc1.md", "doc2.md"]

2. Node2 (Entity: "Aplikasi"):
   - Cari DocumentChunk yang contains "Aplikasi"
   - Cari TextDocument dari DocumentChunk tersebut
   - Hasil: ["doc1.md", "doc3.md"]

Final Knowledge IDs: ["doc1.md", "doc2.md", "doc3.md"]
(Deduplicated, preserve order)
```

### Contoh 2: Entity - EntityType
```
Triplet: (Entity: "Python") -[is_a]-> (EntityType: "Programming Language")

Mapping:
1. Node1 (Entity: "Python"):
   - Cari DocumentChunk yang contains "Python"
   - Hasil: ["doc1.md"]

2. Node2 (EntityType: "Programming Language"):
   - Cari Entity yang is_a "Programming Language"
   - Untuk setiap Entity, cari DocumentChunk yang contains Entity tersebut
   - Hasil: ["doc1.md", "doc2.md", "doc3.md"] (dari berbagai Entity)

Final Knowledge IDs: ["doc1.md", "doc2.md", "doc3.md"]
```

### Contoh 3: DocumentChunk - Entity
```
Triplet: (DocumentChunk: chunk_123) -[contains]-> (Entity: "Notifikasi")

Mapping:
1. Node1 (DocumentChunk: chunk_123):
   - Langsung cari TextDocument via is_part_of
   - Hasil: ["doc1.md"]

2. Node2 (Entity: "Notifikasi"):
   - Cari DocumentChunk yang contains "Notifikasi"
   - Cari TextDocument dari DocumentChunk tersebut
   - Hasil: ["doc1.md", "doc2.md"]

Final Knowledge IDs: ["doc1.md", "doc2.md"]
```

## Deduplication

Karena kita memetakan **kedua node**, bisa jadi ada **duplicate knowledge IDs**:

```python
# Dari result_mapper.py line 260-284
seen_ids = set()  # Untuk deduplication

for triplet in triplets:
    # Map node1
    mapped_ids = await map_node_to_knowledge_ids(node1_id, node1_type, graph_engine)
    for kid in mapped_ids:
        if kid not in seen_ids:  # ← Cek duplicate
            knowledge_ids.append(kid)
            seen_ids.add(kid)
    
    # Map node2
    mapped_ids = await map_node_to_knowledge_ids(node2_id, node2_type, graph_engine)
    for kid in mapped_ids:
        if kid not in seen_ids:  # ← Cek duplicate
            knowledge_ids.append(kid)
            seen_ids.add(kid)
```

**Hasil**: Knowledge IDs yang unique, dengan order preserved (first occurrence wins).

## Visual Flow

```
Triplet: (n1) -[e]-> (n2)
           ↓           ↓
    Map n1 to      Map n2 to
    Knowledge IDs  Knowledge IDs
           ↓           ↓
    ["doc1",      ["doc1",
     "doc2"]       "doc3"]
           ↓           ↓
           └─────┬─────┘
                 ↓
         Deduplicate & Merge
                 ↓
    ["doc1", "doc2", "doc3"]
    (preserve order, unique)
```

## Kesimpulan

✅ **KEDUA node (node1 DAN node2) dipetakan ke knowledge IDs**

Alasan:
1. Kedua node relevan dengan query (muncul dalam triplet yang sama)
2. Coverage lebih lengkap (tidak kehilangan dokumen)
3. Preserve ranking dari triplet search
4. Deduplication memastikan tidak ada duplicate

**Hasil**: List knowledge IDs yang unique, ordered berdasarkan ranking triplet, dengan coverage maksimal dari kedua node.
