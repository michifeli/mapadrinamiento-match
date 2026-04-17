import os
from dotenv import load_dotenv


load_dotenv()


def load_app_config() -> dict:
    """Lee configuración desde variables de entorno y la devuelve en un diccionario."""
    use_ai = os.getenv("USE_AI", "0") == "1"
    allowed_categories = {
        token.strip()
        for token in os.getenv("AI_ALLOWED_CATEGORIES", "Pref").split(",")
        if token.strip()
    }

    return {
        "api_key": os.getenv("API_KEY"),
        "use_ai": use_ai,
        "hf_model": os.getenv("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
        "ai_min_confidence_to_skip": float(os.getenv("AI_MIN_CONFIDENCE_TO_SKIP", "0.85")),
        "ai_max_calls": int(os.getenv("AI_MAX_CALLS", "12")),
        "ai_allowed_categories": allowed_categories,
    }
