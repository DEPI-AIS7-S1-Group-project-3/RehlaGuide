from pathlib import Path
from typing import List, Optional, Dict, Any
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

DB_PATH = Path("data/chroma_db")

def get_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(
        persist_directory=str(DB_PATH),
        embedding_function=embeddings
    )

def search_attractions(
    query: str,
    region: Optional[str] = None,
    category: Optional[str] = None,
    max_duration: Optional[float] = None,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    vector_store = get_vector_store()

    conditions = []
    if region:
        conditions.append({"region": {"$eq": region}})
    if category:
        conditions.append({"category": {"$eq": category}})
    if max_duration is not None:
        conditions.append({"estimated_duration_hours": {"$lte": max_duration}})

    if len(conditions) == 1:
        where_filter = conditions[0]
    elif len(conditions) > 1:
        where_filter = {"$and": conditions}
    else:
        where_filter = None

   
    results = vector_store.similarity_search_with_score(
        query=query,
        k=top_k,
        filter=where_filter
    )

    formatted_results = []
    for doc, score in results:
        formatted_results.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "relevance_score": float(score)
        })

    return formatted_results

if __name__ == "__main__":
    
    print("Testing Retriever Contract...")
    test_results = search_attractions(
        query="famous ancient royal monuments and mummies",
        region="Cairo",
        category="Museum",
        max_duration=3.0,
        top_k=2
    )

    print(f"\nRetrieved {len(test_results)} results:\n")
    for res in test_results:
        print("—" * 40)
        print(f"Name: {res['metadata']['name']}")
        print(f"Region: {res['metadata']['region']}")
        print(f"Category: {res['metadata']['category']}")
        print(f"Est. Duration (hrs): {res['metadata']['estimated_duration_hours']}")
        print(f"Score: {res['relevance_score']:.4f}")