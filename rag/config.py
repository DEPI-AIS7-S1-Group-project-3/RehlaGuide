import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


RAW_DATA_PATH = DATA_DIR / "raw_attractions.json"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

DEFAULT_TOP_K = 5