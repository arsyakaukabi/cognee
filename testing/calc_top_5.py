
import csv
import ast

def calculate_top_k(filename, k=5):
    hits = 0
    total = 0
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            # Parse ground truth which seems to be a list string e.g. "['id1', 'id2']"
            # distinct format in header sample: "['helpdesk__...', ...]"
            try:
                ground_truths = ast.literal_eval(row['id_ground_truth'])
            except:
                # Fallback if it's not a list representation or purely single id
                ground_truths = [row['id_ground_truth']]
            
            # Get retrieved context columns
            retrieved = []
            for i in range(1, k+1):
                col_name = f'context_{i}'
                if col_name in row and row[col_name]:
                    retrieved.append(row[col_name])
            
            # Check for hit
            # We assume if ANY ground truth is in retrieved, it's a hit.
            match = False
            for gt in ground_truths:
                if gt in retrieved:
                    match = True
                    break
            if match:
                hits += 1
                
    return hits, total

files = [
    "/home/usr_00345779_hq_bri_co_id/cognee/testing/eval_output.csv",
    "/home/usr_00345779_hq_bri_co_id/cognee/testing/eval_chunks_output.csv"
]

print(f"{'File':<60} | {'Top-5 Hit Rate':<15} | {'Count'}")
print("-" * 90)

for filepath in files:
    try:
        hits, total = calculate_top_k(filepath, 5)
        rate = (hits / total * 100) if total > 0 else 0
        filename = filepath.split('/')[-1]
        print(f"{filename:<60} | {rate:.2f}%          | {hits}/{total}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
