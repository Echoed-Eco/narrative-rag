from src.llm import chat

print(chat(
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hi in one sentence."},
    ],
    reasoning_effort="low",
))