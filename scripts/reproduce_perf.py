import requests
import json
import time

API_URL = "http://127.0.0.1:8000"

def measure_retrieval(debug=True):
    url = f"{API_URL}/api/v1/search/retrieval"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        # "Authorization": "Bearer {{bearerToken}}", # Assuming auth is handled or disabled in dev
        "X-Debug-Trace": "true" if debug else "false"
    }
    payload = {
        "query": "Apa syarat utk mendapatkan diskon d SUSHI OK Banjarmasin?",
        "top_k": 50,
        "search_type": "graph_completion_custom"
    }
    
    print(f"\n--- Measuring Retrieval (Debug={debug}) ---")
    start = time.time()
    try:
        response = requests.post(url, json=payload, headers=headers)
        duration = (time.time() - start) * 1000
        
        if response.status_code == 200:
            print(f"Success! Duration: {duration:.2f}ms")
            data = response.json()
            print(f"Correlation ID: {data.get('correlationId', data.get('correlation_id', 'N/A'))}")
            print(f"Results: {len(data.get('data', []))}")
        else:
            print(f"Failed: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Error: {e}")

def measure_context(debug=True):
    url = f"{API_URL}/api/v1/search/context"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        # "Authorization": "Bearer {{bearerToken}}",
        "X-Debug-Trace": "true" if debug else "false"
    }
    payload = {
        "query": "Apa syarat utk mendapatkan diskon d SUSHI OK Banjarmasin?",
        "top_k": 3,
        "dataset_ids": [
            "9107cf1c-e8e8-5869-9c87-64e9922843df"
        ]
    }

    print(f"\n--- Measuring Context (Debug={debug}) ---")
    start = time.time()
    try:
        response = requests.post(url, json=payload, headers=headers)
        duration = (time.time() - start) * 1000

        if response.status_code == 200:
            print(f"Success! Duration: {duration:.2f}ms")
            data = response.json()
            print(f"Correlation ID: {data.get('correlationId', data.get('correlation_id', 'N/A'))}")
            print(f"Context Length: {len(str(data.get('data', '')))}")
        else:
            print(f"Failed: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Measure
    measure_retrieval(debug=True)
    measure_context(debug=True)
