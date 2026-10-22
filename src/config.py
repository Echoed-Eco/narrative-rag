import os
from dotenv import load_dotenv

load_dotenv()

def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing environment variable: {name}")
    return value

AZURE_OPENAI_ENDPOINT = _require("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = _require("AZURE_OPENAI_API_KEY")
CHAT_DEPLOYMENT = os.getenv("CHAT_DEPLOYMENT", "chat")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
SEARCH_ENDPOINT = os.getenv("SEARCH_ENDPOINT")
SEARCH_API_KEY = os.getenv("SEARCH_API_KEY")
SEARCH_INDEX_NAME = os.getenv("SEARCH_INDEX_NAME")
RETRIEVER = os.getenv("RETRIEVER", "local")         #local or azure
SEARCH_MODE = os.getenv("SEARCH_MODE", "hybrid")    #hybrid or vector
DATA_DIR = "data/sample"

# Experimental
CHUNK_SIZE = 150
CHUNK_OVERLAP = 25
TOP_K = 5