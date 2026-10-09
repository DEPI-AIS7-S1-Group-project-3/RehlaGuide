from typing import List, Dict, Any
from rag.retriever import search_attractions


EVAL_DATASET = [
    {
        "query": "famous ancient pyramids and royal tombs",
        "filters": {"region": "Giza"},
        "expected_ids": [5, 6, 7],  
        "top_k": 3
    },
    {
        "query": "coptic manuscripts and religious heritage",
        "filters": {"region": "Cairo", "category": "Museum"},
        "expected_ids": [3],  
        "top_k": 2
    },
    {
        "query": "underwater diving coral reefs marine life",
        "filters": {"region": "South Sinai"},
        "expected_ids": [24], 
        "top_k": 2
    },
    {
        "query": "ancient temples dedicated to Horus and Sobek",
        "filters": {"region": "Aswan"},
        "expected_ids": [18, 19],  
        "top_k": 3
    },
    {
        "query": "mummies and ancient royal mummies gallery",
        "filters": {"category": "Museum"},
        "expected_ids": [2, 13],  
        "top_k": 3
    }
]

def evaluate_retriever():
    print("=== Starting Retrieval Evaluation Pipeline ===\n")
    total_queries = len(EVAL_DATASET)
    hit_count = 0
    total_recall = 0.0

    for i, test_case in enumerate(EVAL_DATASET, start=1):
        query = test_case["query"]
        filters = test_case.get("filters", {})
        expected_ids = set(test_case["expected_ids"])
        top_k = test_case.get("top_k", 5)

       
        results = search_attractions(
            query=query,
            region=filters.get("region"),
            category=filters.get("category"),
            max_duration=filters.get("max_duration"),
            top_k=top_k
        )

        retrieved_ids = {res["metadata"]["id"] for res in results}
        
     
        relevant_retrieved = retrieved_ids.intersection(expected_ids)
        is_hit = len(relevant_retrieved) > 0
        recall = len(relevant_retrieved) / len(expected_ids) if expected_ids else 0.0

        if is_hit:
            hit_count += 1
        total_recall += recall

        print(f"Query [{i}/{total_queries}]: '{query}'")
        print(f"  - Filters applied: {filters}")
        print(f"  - Expected IDs: {list(expected_ids)} | Retrieved IDs: {list(retrieved_ids)}")
        print(f"  - Hit@{top_k}: {'✅ PASS' if is_hit else '❌ FAIL'} | Recall@{top_k}: {recall:.2f}\n")

    mean_hit_at_k = hit_count / total_queries
    mean_recall_at_k = total_recall / total_queries

    print("=" * 40)
    print("=== EVALUATION RESULTS SUMMARY ===")
    print(f"Mean Hit@K:    {mean_hit_at_k * 100:.1f}%")
    print(f"Mean Recall@K: {mean_recall_at_k * 100:.1f}%")
    print("=" * 40)

if __name__ == "__main__":
    evaluate_retriever()