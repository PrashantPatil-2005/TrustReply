"""LLM client wrapper with demo-mode fallback."""
import os
from dotenv import load_dotenv

load_dotenv()

DEMO_MODE = os.getenv("DEMO_MODE", "0") == "1"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def call_llm(system: str, user: str, temperature: float = 0.3) -> str:
    """Call OpenAI API or return empty string in demo mode."""
    if DEMO_MODE or not OPENAI_API_KEY:
        return ""

    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()
