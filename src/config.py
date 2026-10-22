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
DATA_DIR = "data/sample"

# Experimental
CHUNK_SIZE = 150
CHUNK_OVERLAP = 25
TOP_K = 5