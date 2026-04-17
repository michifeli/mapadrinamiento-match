import re
import pandas as pd
import unicodedata


def remove_accents(input_str: str) -> str:
    if pd.isna(input_str) or not isinstance(input_str, str):
        return ""
    nfkd_form = unicodedata.normalize("NFKD", input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])


def normalize_token(text: str) -> str:
    text = remove_accents(str(text)).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def split_response(text: str) -> list[str]:
    if pd.isna(text):
        return []
    raw = str(text).strip()
    if not raw:
        return []
    parts = re.split(r"[;|,/\n]+", raw)
    cleaned = [p.strip() for p in parts if p and p.strip()]
    return cleaned if cleaned else [raw]


def infer_pref_from_free_text(text_norm: str) -> list[str]:
    if not text_norm:
        return []

    mapped = set()
    has_carrete = any(k in text_norm for k in ["carrete", "carretear", "fiesta"])
    has_casa = "casa" in text_norm
    has_jugar = any(k in text_norm for k in ["jugar", "play", "videojuego"])
    has_series = any(k in text_norm for k in ["series", "pelicula", "pelis", "anime", "netflix"])
    has_estudio = any(k in text_norm for k in ["estudi", "tarea", "universidad", "u "])
    has_todo = any(k in text_norm for k in ["todo", "todas", "igual", "50/50", "de todo"])

    if has_todo:
        mapped.update(["Salir a carretear", "Quedarte en casa viendo series/peliculas"])
    if has_carrete and has_casa:
        mapped.add("Carretear en casa")
    elif has_carrete:
        mapped.add("Salir a carretear")

    if has_casa and has_jugar:
        mapped.add("Quedarte en casa jugando")
    if has_casa and has_series:
        mapped.add("Quedarte en casa viendo series/peliculas")
    if has_casa and has_estudio:
        mapped.add("Quedarte en casa viendo series/peliculas")

    # Si solo menciona quedarse en casa y no hay otra señal clara,
    # se usa una opción oficial de "quedarse en casa" para evitar texto libre.
    if has_casa and not mapped:
        mapped.add("Quedarte en casa viendo series/peliculas")

    return sorted(mapped)
