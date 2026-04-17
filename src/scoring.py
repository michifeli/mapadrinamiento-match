from .text_normalization import normalize_token


WEIGHTS = {
    "Pref": 7.0,
    "Hobby": 7.0,
    "Juegos": 4.0,
    "Deportes": 4.0,
    "Comida": 4.0,
    "Musica": 4.0,
    "Series": 2.0,
    "Idiomas": 2.0,
    "Dieta": 1.0,
}

SCORE_FLOOR = 8.0
LOW_SCORE_PENALTY_FACTOR = 0.55


def get_set(val):
    """Convierte una cadena separada por ';' en conjunto normalizado."""
    return {normalize_token(x.strip()) for x in str(val).split(";") if x.strip()}


def calculate_match_components(m_row, p_row):
    """Calcula detalle completo de puntaje entre mechón y padrino."""
    raw_total = 0.0
    details = []
    cat_scores = {}
    cat_sims = {}

    # Similaridad por categoría con Jaccard.
    for cat, weight in WEIGHTS.items():
        s1 = get_set(m_row[cat])
        s2 = get_set(p_row[cat])
        if not s1 or not s2:
            cat_scores[cat] = 0.0
            cat_sims[cat] = 0.0
            continue

        inter = len(s1.intersection(s2))
        union = len(s1.union(s2))
        sim = inter / union
        score = sim * weight if sim > 0 else 0.0

        cat_sims[cat] = sim
        cat_scores[cat] = score

        if sim > 0:
            raw_total += score
            details.append(f"{cat}(x{weight}): {round(sim, 2)}")

    coverage_bonus = 0.15 * sum(1 for v in cat_sims.values() if v > 0)
    low_gap = max(0.0, SCORE_FLOOR - raw_total)
    low_score_penalty = LOW_SCORE_PENALTY_FACTOR * ((low_gap**2) / SCORE_FLOOR)

    pref_sim = cat_sims.get("Pref", 0.0)
    hobby_sim = cat_sims.get("Hobby", 0.0)
    if pref_sim == 0 and hobby_sim == 0:
        vital_multiplier = 0.80
    elif pref_sim == 0 or hobby_sim == 0:
        vital_multiplier = 0.92
    else:
        vital_multiplier = 1.0

    adjusted_base = raw_total + coverage_bonus - low_score_penalty
    effective = max(0.0, adjusted_base * vital_multiplier)
    dominance = (max(cat_scores.values()) / raw_total) if raw_total > 0 else 1.0

    return {
        "raw_total": raw_total,
        "effective_total": effective,
        "details": " | ".join(details),
        "cat_scores": cat_scores,
        "cat_sims": cat_sims,
        "dominance": dominance,
        "coverage_bonus": coverage_bonus,
        "low_score_penalty": low_score_penalty,
        "vital_multiplier": vital_multiplier,
    }


def calculate_match_score(m_row, p_row):
    """Compatibilidad con interfaz previa: retorna puntaje y detalle textual."""
    comp = calculate_match_components(m_row, p_row)
    return comp["raw_total"], comp["details"]
