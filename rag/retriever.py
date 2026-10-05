from pathlib import Path
from typing import List, Optional, Dict, Any
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

DB_PATH = Path("data/chroma_db")

def get_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(
        persist_directory=str(DB_PATH),
        embedding_function=embeddings
    )
    return vector_store

def search_attractions(query: str, city_filter: Optional[str] = None, top_k: int = 3) -> List[Dict[str, Any]]:
    vector_store = get_retriever()
    
    where_filter = {"city": city_filter} if city_filter else None

    results = vector_store.similarity_search(
        query=query,
        k=top_k,
        filter=where_filter
    )

    formatted_results = []
    for doc in results:
        formatted_results.append({
            "content": doc.page_content,
            "metadata": doc.metadata
        })

    return formatted_results

if __name__ == "__main__":
    sample_results = search_attractions("museums with royal mummies", city_filter="Cairo", top_k=2)
    print(f"Retrieved {len(sample_results)} results:")
    for res in sample_results:
        print("—" * 30)
        print(f"Name: {res['metadata']['name']}")
        print(f"City: {res['metadata']['city']}")
        print(f"Price EGP (Egyptian): {res['metadata']['ticket_price_egyptian_adult_egp']}")