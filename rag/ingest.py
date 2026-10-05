import json
from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

DATA_PATH = Path("data/raw_attractions.json")
DB_PATH = Path("data/chroma_db")

def load_attractions():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def build_vector_store():
    raw_data = load_attractions()
    documents = []

    for item in raw_data:
    
        text_content = (
            f"Name: {item.get('name', '')}\n"
            f"Region/City: {item.get('region', item.get('city', ''))}\n"
            f"Category: {item.get('category', '')}\n"
            f"Description: {item.get('description', '')}\n"
            f"Opening Hours: {item.get('opening_hours', '')}\n"
            f"Egyptian Adult Price (EGP): {item.get('ticket_price_egyptian_adult_egp', 0)}\n"
            f"Foreigner Adult Price (EGP): {item.get('ticket_price_foreigner_adult_egp', 0)}\n"
            f"Estimated Visit Duration (Minutes): {item.get('estimated_visit_duration_min', 0)}"
        ).strip()

       
        duration_hours = float(item.get("estimated_visit_duration_min", 0)) / 60.0

       
        metadata = {
            "id": int(item.get("id", 0)),
            "name": str(item.get("name", "")),
            "region": str(item.get("region", item.get("city", ""))),
            "category": str(item.get("category", "")),
            "ticket_price_egyptian_adult_egp": float(item.get("ticket_price_egyptian_adult_egp", 0)),
            "ticket_price_foreigner_adult_egp": float(item.get("ticket_price_foreigner_adult_egp", 0)),
            "estimated_duration_hours": duration_hours
        }

        doc = Document(page_content=text_content, metadata=metadata)
        documents.append(doc)

    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(DB_PATH)
    )
    print(f"Successfully indexed {len(documents)} attractions into ChromaDB at '{DB_PATH}'.")
    return vector_store

if __name__ == "__main__":
    build_vector_store()