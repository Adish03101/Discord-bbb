# test_llm_json.py

import os
import json
import requests
from dotenv import load_dotenv
from pathlib import Path

def load_environment():
    # Try commands/.env first, then project root .env
    env_path = Path(__file__).resolve().parent.parent / "commands" / ".env"
    if not env_path.exists():
        env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)

def main():
    load_environment()

    GEN_MODEL = os.getenv("GEN_MODEL", "gemini-2.5-flash")
    API_KEY   = os.getenv("GEMINI_API_KEY", "")
    CHAT_URL  = f"https://generativelanguage.googleapis.com/v1/models/{GEN_MODEL}:generateContent?key={API_KEY}"

    # Customize your context & query here:
    context = "This is a test context."
    query   = "Hello, how are you?"

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            "You are a concise assistant. Use the context to answer.\n\n"
                            f"Context:\n{context}\n\nQuestion: {query}"
                        )
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 300,
            "topP": 0.9
        }
    }

    try:
        resp = requests.post(CHAT_URL, json=payload, timeout=10)
        resp.raise_for_status()
        # Pretty-print the raw JSON
        print(json.dumps(resp.json(), indent=2))
    except Exception as e:
        print("Request failed:", e)

if __name__ == "__main__":
    main()
