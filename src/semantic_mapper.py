from difflib import SequenceMatcher
import requests
import time
import pandas as pd

from .catalogs import ALIASES, OFFICIAL_OPTIONS
from .text_normalization import infer_pref_from_free_text, normalize_token, split_response


def create_mapper_state(config: dict) -> dict:
    """Crea el estado del mapeador en un diccionario simple."""
    return {
        "config": config,
        "ai_runtime_available": True,
        "ai_call_count": 0,
    }


def reset_mapper_state(mapper_state: dict) -> None:
    """Resetea el estado temporal del mapeador."""
    mapper_state["ai_runtime_available"] = True
    mapper_state["ai_call_count"] = 0


def deterministic_map_response(text_input: str, category: str, valid_options: list[str] | None):
    """Mapea texto a opciones oficiales usando reglas locales."""
    if pd.isna(text_input) or str(text_input).strip() == "":
        return "", "Vacio", 1.0

    options = valid_options or OFFICIAL_OPTIONS.get(category, [])
    options_norm = {normalize_token(option): option for option in options}
    aliases = ALIASES.get(category, {})
    aliases_norm = {normalize_token(alias): value for alias, value in aliases.items()}

    if category == "Pref":
        pref_inferred = infer_pref_from_free_text(normalize_token(text_input))
        if pref_inferred:
            return "; ".join(sorted(set(pref_inferred))), "Regla semántica pref", 0.92

    mapped = []
    reasons = []

    for piece in split_response(text_input):
        piece_norm = normalize_token(piece)
        if not piece_norm:
            continue

        if piece_norm in aliases_norm:
            mapped.append(aliases_norm[piece_norm])
            reasons.append(f"Alias exacto: '{piece}'")
            continue

        alias_containment_match = None
        for alias_norm, alias_value in aliases_norm.items():
            if (alias_norm in piece_norm or piece_norm in alias_norm) and len(alias_norm) >= 4:
                alias_containment_match = alias_value
                break

        if alias_containment_match:
            mapped.append(alias_containment_match)
            reasons.append(f"Alias contención: '{piece}' -> {alias_containment_match}")
            continue

        if piece_norm in options_norm:
            mapped.append(options_norm[piece_norm])
            reasons.append(f"Opción exacta: '{piece}'")
            continue

        option_containment_match = None
        for option_norm, option_value in options_norm.items():
            if option_norm in piece_norm or piece_norm in option_norm:
                option_containment_match = option_value
                break

        if option_containment_match:
            mapped.append(option_containment_match)
            reasons.append(f"Contención: '{piece}' -> {option_containment_match}")
            continue

        best_option = None
        best_ratio = 0.0
        for option_norm, option_value in options_norm.items():
            ratio = SequenceMatcher(None, piece_norm, option_norm).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_option = option_value

        if best_option and best_ratio >= 0.84:
            mapped.append(best_option)
            reasons.append(f"Fuzzy({best_ratio:.2f}): '{piece}' -> {best_option}")

    if mapped:
        unique_mapped = sorted(dict.fromkeys(mapped))
        return "; ".join(unique_mapped), " | ".join(reasons[:4]), 0.9

    return str(text_input).strip(), "Sin mapeo confiable", 0.35


def normalize_with_reasoning(
    mapper_state: dict,
    text_input: str,
    valid_options: list[str] | None,
    category: str,
):
    """Elige entre mapeo local y externo según confianza y configuración."""
    if pd.isna(text_input) or str(text_input).strip() == "":
        return "", "Vacio", "local", 1.0

    config = mapper_state["config"]
    text_input = str(text_input).strip()
    local_result, local_reason, local_conf = deterministic_map_response(
        text_input,
        category,
        valid_options if isinstance(valid_options, list) else [],
    )

    use_external = config["use_ai"] and config["api_key"] and mapper_state["ai_runtime_available"]
    if not use_external:
        return local_result, f"Local({local_conf:.2f}): {local_reason}", "local", local_conf

    if category not in config["ai_allowed_categories"]:
        return local_result, f"Local({local_conf:.2f}): {local_reason}", "local", local_conf

    if mapper_state["ai_call_count"] >= config["ai_max_calls"]:
        return (
            local_result,
            f"Local({local_conf:.2f}): presupuesto de llamadas agotado",
            "local",
            local_conf,
        )

    if local_conf >= config["ai_min_confidence_to_skip"]:
        return local_result, f"Local({local_conf:.2f}): {local_reason}", "local", local_conf

    prompt = f"""Eres un experto en clasificación semántica estricta.
CATEGORIA: {category}
OPCIONES OFICIALES: {valid_options}
ENTRADA USUARIO: \"{text_input}\"

TAREA:
1. Analiza la entrada y ASOCIALA OBLIGATORIAMENTE a una o más OPCIONES OFICIALES.
2. Si el usuario usó sinónimos (ej: 'balompie' -> 'Futbol'), haz la conversión.
3. Si el usuario puso algo vago (ej: 'como de todo'), asígnalo a 'Omnívora'.
4. NO puedes inventar categorías. Si no hay relación alguna, usa la opción más cercana o ignora términos basura.

FORMATO DE RESPUESTA (ESTRICTO):
Razon: <tu razonamiento lógico de por qué lo asociaste a esas opciones>
Resultado: <opciones oficiales resultantes separadas por ;>"""

    api_url = "https://router.huggingface.co/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {config['api_key']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config["hf_model"],
        "messages": [
            {"role": "system", "content": "Clasifica texto a categorías oficiales sin inventar opciones."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 120,
        "temperature": 0.1,
    }

    try:
        mapper_state["ai_call_count"] += 1
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        time.sleep(0.5)
        data = response.json()
        raw_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")

        reason, result = "No se pudo extraer razón", text_input
        if "Razon:" in raw_text and "Resultado:" in raw_text:
            reason = raw_text.split("Razon:")[1].split("Resultado:")[0].strip()
            result = raw_text.split("Resultado:")[1].strip()
        elif "Resultado:" in raw_text:
            result = raw_text.split("Resultado:")[1].strip()
            reason = "Respuesta IA parcial"

        if result and str(result).strip():
            validated, _, validated_conf = deterministic_map_response(
                result.strip(),
                category,
                valid_options,
            )
            if validated_conf >= 0.7:
                return validated, reason, "ia", 0.8
            return (
                local_result,
                f"Fallback Local({local_conf:.2f}) por salida externa no confiable",
                "local_fallback",
                local_conf,
            )

        return local_result, f"Fallback Local({local_conf:.2f}): {local_reason}", "local_fallback", local_conf

    except requests.exceptions.HTTPError as e:
        status = getattr(e.response, "status_code", None)
        if status in (401, 402, 403, 429):
            mapper_state["ai_runtime_available"] = False
            return (
                local_result,
                f"Fallback Local({local_conf:.2f}) - IA deshabilitada por estado HTTP {status}",
                "local_fallback",
                local_conf,
            )
        return (
            local_result,
            f"Fallback Local({local_conf:.2f}) por error HTTP externo {status}",
            "local_fallback",
            local_conf,
        )
    except Exception as e:
        return (
            local_result,
            f"Fallback Local({local_conf:.2f}) por error externo: {str(e)}",
            "local_fallback",
            local_conf,
        )
