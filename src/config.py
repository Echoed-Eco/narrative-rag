import os
from dotenv import load_dotenv

load_dotenv()

def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing environment variable: {name}")
    return value

# Azure OpenAI
AZURE_OPENAI_ENDPOINT = _require("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = _require("AZURE_OPENAI_API_KEY")
CHAT_DEPLOYMENT = os.getenv("CHAT_DEPLOYMENT", "chat")

# Experiment knobs (you'll change these during eval)
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 5