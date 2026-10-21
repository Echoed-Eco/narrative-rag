from openai import OpenAI
from src import config

_client = OpenAI(
    base_url=config.AZURE_OPENAI_ENDPOINT.rstrip("/") + "/openai/v1/",
    api_key=config.AZURE_OPENAI_API_KEY,
)

def chat(messages, deployment=None, **kwargs):
    response = _client.chat.completions.create(
        model=deployment or config.CHAT_DEPLOYMENT,
        messages=messages,
        **kwargs,
    )
    return response.choices[0].message.content