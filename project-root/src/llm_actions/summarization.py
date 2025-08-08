from pathlib import Path
from dotenv import load_dotenv
import os
import requests
import time
from typing import Optional

# ─── Load .env from sibling 'commands/.env' ─────────────────────────────────────
env_path = Path(__file__).resolve().parent.parent / "commands" / ".env"
load_dotenv(dotenv_path=env_path)

# ─── Prompt Context ─────────────────────────────────────────────────────────────
lvl_prompt = {
    "client": "You are a concise summarizer that produces a 50-70 word summary of the input text, staying neutral and fact-forward.",
    "project": "You are a project-document summarizer. Summarize in 50-70 words, emphasizing purpose, status, and key points.",
    "subproject": "You are a subproject document summarizer. Summarize in 50-70 words, focusing on scope and outstanding actions.",
}

# ─── Gemini API Configuration ──────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# Allow override of model via env; default to a more conservative general model
GEN_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-lite")
CHAT_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEN_MODEL}:generateContent"


class Summary:
    def __init__(self, lvl_prompt_map: dict = None):
        self.lvl_prompt = lvl_prompt_map or lvl_prompt
        self.api_key = GEMINI_API_KEY
        self.model = GEN_MODEL
        self.url = CHAT_URL

    def _enforce_word_cap(self, text: str, min_words=50, max_words=70) -> str:
        words = text.strip().split()
        if len(words) <= max_words:
            return text.strip()
        capped = " ".join(words[:max_words])
        if not capped.endswith(('.', '?', '!')):
            capped = capped.rstrip(',') + "..."
        return capped

    def summarize(self, text: str, context: str = "") -> str:
        if not text or not text.strip():
            return "[Summary Error] Empty input text."

        system_msg = self.lvl_prompt.get(context, "You are a concise summarizer that summarizes text in 50-70 words.")
        input_snippet = text.strip()
        if len(input_snippet) > 15000:
            input_snippet = input_snippet[:15000] + "..."

        user_msg = (
            f"{system_msg} Keep the summary between 50 and 70 words. "
            f"Summarize the following content:\n\n{input_snippet}"
        )

        payload = {
            "contents": [
                {"role": "system", "parts": [{"text": system_msg}]},
                {"role": "user", "parts": [{"text": user_msg}]}
            ],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 180,
                "topP": 0.9,
                "candidateCount": 1,
            }
        }

        headers = {
            "Content-Type": "application/json"
        }

        attempt = 0
        while attempt < 3:
            try:
                response = requests.post(
                    f"{self.url}?key={self.api_key}",
                    headers=headers,
                    json=payload,
                    timeout=10,
                )
                response.raise_for_status()
                result = response.json()
                candidate = (
                    result.get("candidates", [{}])[0]
                          .get("content", {})
                          .get("parts", [{}])[0]
                          .get("text", "")
                          .strip()
                )
                if not candidate:
                    return "[Summary Error] No summary content returned."

                capped = self._enforce_word_cap(candidate, min_words=50, max_words=70)
                return capped

            except requests.HTTPError as e:
                status = None
                try:
                    status = e.response.status_code
                except Exception:
                    pass
                if status and 400 <= status < 500:
                    return f"[Summary Error] HTTP error {status}: {e.response.text}"
                attempt += 1
                time.sleep(1 + attempt * 1.5)
            except Exception as e:
                attempt += 1
                time.sleep(1 + attempt * 1.5)
        return "[Summary Error] Failed after retries."

# ─── Factory Method ─────────────────────────────────────────────────────────────
def get_summarizer() -> Summary:
    return Summary()
