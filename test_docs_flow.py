import requests
import time

API_URL = "http://localhost:8000"

def test_docs_flow():
    # 1. Add a document
    file_content = "This is a test document for verification. It has some content to be summarized."
    files = {'file': ('test_verification.txt', file_content)}
    data = {'datasetName': 'main_dataset'}
    print("Ingesting document...")
    response = requests.post(f"{API_URL}/api/v1/add", files=files, data=data, headers={"Authorization": "Bearer dummy"}) # Auth might be skipped or mocked?
    # If auth is required, we might need a token. The client.py has auth routers but maybe we can skip if not enabled.
    # checking client.py: REQUIRE_AUTHENTICATION is imported.
    print(f"Ingest response: {response.status_code} {response.text}")
    
    if response.status_code != 200:
        return

    # Wait for processing
    time.sleep(5)
    
    # 2. Search to get the ID/Name
    print("Searching for document...")
    search_payload = {
        "query": "verification",
        "search_type": "chunks", 
        "top_k": 1
    }
    response = requests.post(f"{API_URL}/api/v1/search/retrieval", json=search_payload)
    print(f"Search response: {response.status_code} {response.text}")
    
    results = response.json().get("data", [])
    if not results:
        print("No results found.")
        return
        
    first_result = results[0]
    id_knowledge = first_result.get("idKnowledge")
    knowledge_type = first_result.get("knowledgeType")
    
    if not id_knowledge or not knowledge_type:
        print("Invalid result format.")
        return
        
    doc_name = f"{knowledge_type}_{id_knowledge}"
    print(f"Constructed doc_name: {doc_name}")
    
    # 3. Call /docs endpoint
    print("Calling /docs endpoint...")
    docs_payload = {"name": doc_name}
    response = requests.post(f"{API_URL}/api/v1/docs/", json=docs_payload)
    print(f"Docs response: {response.status_code} {response.text}")

if __name__ == "__main__":
    test_docs_flow()
