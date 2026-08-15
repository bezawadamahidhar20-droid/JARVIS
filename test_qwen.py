from ollama import chat

response = chat(
    model="qwen3:8b",
    messages=[
        {
            "role": "user",
            "content": "Hello JARVIS. Introduce yourself in one short sentence."
        }
    ],
)

print("\nJARVIS:", response.message.content)