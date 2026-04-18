import numpy as np
import pandas as pd
import os
from scipy.optimize import linear_sum_assignment

from .scoring import calculate_match_components, SCORE_FLOOR


BIG_COST = 1_000_000_000.0
# Piso mínimo recomendado para evitar matches débiles.
# Se puede ajustar desde .env con MIN_EXPERIENCE_SCORE.
MIN_EXPERIENCE_SCORE = float(os.getenv("MIN_EXPERIENCE_SCORE", "13.0"))


def _build_effective_matrix(mechon: pd.DataFrame, mapadrinos: pd.DataFrame) -> np.ndarray:
    """Construye la matriz de score ajustado (effective_total)."""
    effective = np.zeros((len(mechon), len(mapadrinos)))
    for i in range(len(mechon)):
        m = mechon.iloc[i]
        for j in range(len(mapadrinos)):
            p = mapadrinos.iloc[j]
            comp = calculate_match_components(m, p)
            effective[i, j] = comp["effective_total"]
    return effective


def _is_threshold_feasible(effective: np.ndarray, threshold: float) -> bool:
    """Valida si existe matching completo (del lado menor) con score >= threshold."""
    allowed = effective >= threshold
    cost = np.where(allowed, 0.0, BIG_COST)
    row_idx, col_idx = linear_sum_assignment(cost)
    return bool(np.all(allowed[row_idx, col_idx]))


def _find_best_fair_threshold(effective: np.ndarray) -> float:
    """Busca el mayor umbral posible para maximizar el peor score asignado."""
    unique_vals = np.unique(effective)
    unique_vals.sort()

    lo = 0
    hi = len(unique_vals) - 1
    best_idx = 0

    while lo <= hi:
        mid = (lo + hi) // 2
        t = float(unique_vals[mid])
        if _is_threshold_feasible(effective, t):
            best_idx = mid
            lo = mid + 1
        else:
            hi = mid - 1

    return float(unique_vals[best_idx])


def _solve_fair_assignment(effective: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Resuelve matching maximizando total sujeto a un umbral mínimo de fairness."""
    allowed = effective >= threshold
    cost = np.where(allowed, -effective, BIG_COST)
    row_idx, col_idx = linear_sum_assignment(cost)
    return row_idx, col_idx


def _get_top_k_suggestions_for_mechon(
    mechon_idx: int,
    effective: np.ndarray,
    available_mapadrino_indices: set[int],
    k: int = 2,
) -> list[tuple[int, float]]:
    """Retorna top-k sugerencias (índice mapadrino, score) para un mechón."""
    candidates = [
        (j, float(effective[mechon_idx, j]))
        for j in available_mapadrino_indices
    ]
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:k]


def _build_top2_suggestion_fields(
    mechon_idx: int,
    effective: np.ndarray,
    mapadrinos: pd.DataFrame,
    candidate_indices: set[int],
) -> dict:
    """Arma campos de sugerencia 1 y 2 para un mechón."""
    top = _get_top_k_suggestions_for_mechon(
        mechon_idx,
        effective,
        candidate_indices,
        k=2,
    )

    name1 = ""
    score1 = 0.0
    name2 = ""
    score2 = 0.0

    if len(top) >= 1:
        j1, s1 = top[0]
        name1 = str(mapadrinos.iloc[j1]["Name"])
        score1 = round(float(s1), 2)
    if len(top) >= 2:
        j2, s2 = top[1]
        name2 = str(mapadrinos.iloc[j2]["Name"])
        score2 = round(float(s2), 2)

    return {
        "Sugerencia_Padrino": name1,
        "Sugerencia_Score_Ajustado": score1,
        "Sugerencia2_Padrino": name2,
        "Sugerencia2_Score_Ajustado": score2,
    }


def match_algorithm(mechon: pd.DataFrame, mapadrinos: pd.DataFrame) -> pd.DataFrame:
    """Aplica asignación uno-a-uno con criterio de fairness + óptimo global."""
    # Estructura única de columnas para que el CSV salga consistente.
    output_columns = [
        "Mechon",
        "Padrino",
        "Score_Total",
        "Score_Ajustado",
        "Dominancia_Max",
        "Multiplicador_Vital",
        "Bonus_Cobertura",
        "Penalizacion_Cola",
        "Alertas",
        "Justificacion",
        "Sugerencia_Padrino",
        "Sugerencia_Score_Ajustado",
        "Sugerencia2_Padrino",
        "Sugerencia2_Score_Ajustado",
    ]

    if len(mechon) == 0 and len(mapadrinos) == 0:
        return pd.DataFrame(columns=output_columns)

    if len(mechon) == 0:
        return pd.DataFrame(
            [
                {
                    "Mechon": "",
                    "Padrino": p["Name"],
                    "Score_Total": 0.0,
                    "Score_Ajustado": 0.0,
                    "Dominancia_Max": 0.0,
                    "Multiplicador_Vital": 0.0,
                    "Bonus_Cobertura": 0.0,
                    "Penalizacion_Cola": 0.0,
                    "Alertas": "sin_emparejar_por_cupo",
                    "Justificacion": "No hay mechones disponibles para emparejar.",
                    "Sugerencia_Padrino": "",
                    "Sugerencia_Score_Ajustado": 0.0,
                    "Sugerencia2_Padrino": "",
                    "Sugerencia2_Score_Ajustado": 0.0,
                }
                for _, p in mapadrinos.iterrows()
            ]
        )

    if len(mapadrinos) == 0:
        return pd.DataFrame(
            [
                {
                    "Mechon": m["Name"],
                    "Padrino": "",
                    "Score_Total": 0.0,
                    "Score_Ajustado": 0.0,
                    "Dominancia_Max": 0.0,
                    "Multiplicador_Vital": 0.0,
                    "Bonus_Cobertura": 0.0,
                    "Penalizacion_Cola": 0.0,
                    "Alertas": "sin_emparejar_por_cupo",
                    "Justificacion": "No hay mapadrinos disponibles para emparejar.",
                    "Sugerencia_Padrino": "",
                    "Sugerencia_Score_Ajustado": 0.0,
                    "Sugerencia2_Padrino": "",
                    "Sugerencia2_Score_Ajustado": 0.0,
                }
                for _, m in mechon.iterrows()
            ]
        )

    effective = _build_effective_matrix(mechon, mapadrinos)
    all_mapadrino_indices = set(range(len(mapadrinos)))

    # 1) Subimos fairness lo más posible.
    fair_threshold = _find_best_fair_threshold(effective)

    # 2) Con ese fairness fijo, maximizamos total.
    row_idx, col_idx = _solve_fair_assignment(effective, fair_threshold)

    results = []
    accepted_rows = set()
    accepted_cols = set()

    for i, j in zip(row_idx, col_idx):
        m, p = mechon.iloc[i], mapadrinos.iloc[j]
        comp = calculate_match_components(m, p)

        # Si no llega al mínimo, preferimos dejarlo sin match.
        if comp["effective_total"] < MIN_EXPERIENCE_SCORE:
            continue

        accepted_rows.add(i)
        accepted_cols.add(j)

        red_flags = []
        if comp["raw_total"] < SCORE_FLOOR:
            red_flags.append("score_bajo")
        if comp["dominance"] > 0.55:
            red_flags.append("dominancia_categoria")
        has_pref_match = comp["cat_sims"].get("Pref", 0.0) > 0
        has_hobby_match = comp["cat_sims"].get("Hobby", 0.0) > 0
        if not has_pref_match and not has_hobby_match:
            red_flags.append("sin_afinidad_vital")

        results.append(
            {
                "Mechon": m["Name"],
                "Padrino": p["Name"],
                "Score_Total": round(comp["raw_total"], 2),
                "Score_Ajustado": round(comp["effective_total"], 2),
                "Dominancia_Max": round(comp["dominance"], 2),
                "Multiplicador_Vital": round(comp["vital_multiplier"], 2),
                "Bonus_Cobertura": round(comp["coverage_bonus"], 2),
                "Penalizacion_Cola": round(comp["low_score_penalty"], 2),
                "Alertas": ",".join(red_flags) if red_flags else "",
                "Justificacion": comp["details"],
                **_build_top2_suggestion_fields(i, effective, mapadrinos, all_mapadrino_indices),
            }
        )

    # Segundo barrido: emparejar sueltos que sí superen mínimo.
    unmatched_row_indices = set(range(len(mechon))) - accepted_rows
    unmatched_col_indices = set(range(len(mapadrinos))) - accepted_cols

    while unmatched_row_indices and unmatched_col_indices:
        best_score = -1.0
        best_pair = None

        for i in unmatched_row_indices:
            for j in unmatched_col_indices:
                score = float(effective[i, j])
                if score >= MIN_EXPERIENCE_SCORE and score > best_score:
                    best_score = score
                    best_pair = (i, j)

        if best_pair is None:
            break

        i, j = best_pair
        m, p = mechon.iloc[i], mapadrinos.iloc[j]
        comp = calculate_match_components(m, p)

        red_flags = []
        if comp["raw_total"] < SCORE_FLOOR:
            red_flags.append("score_bajo")
        if comp["dominance"] > 0.55:
            red_flags.append("dominancia_categoria")
        has_pref_match = comp["cat_sims"].get("Pref", 0.0) > 0
        has_hobby_match = comp["cat_sims"].get("Hobby", 0.0) > 0
        if not has_pref_match and not has_hobby_match:
            red_flags.append("sin_afinidad_vital")

        results.append(
            {
                "Mechon": m["Name"],
                "Padrino": p["Name"],
                "Score_Total": round(comp["raw_total"], 2),
                "Score_Ajustado": round(comp["effective_total"], 2),
                "Dominancia_Max": round(comp["dominance"], 2),
                "Multiplicador_Vital": round(comp["vital_multiplier"], 2),
                "Bonus_Cobertura": round(comp["coverage_bonus"], 2),
                "Penalizacion_Cola": round(comp["low_score_penalty"], 2),
                "Alertas": ",".join(red_flags) if red_flags else "",
                "Justificacion": comp["details"],
                **_build_top2_suggestion_fields(i, effective, mapadrinos, all_mapadrino_indices),
            }
        )

        accepted_rows.add(i)
        accepted_cols.add(j)
        unmatched_row_indices.remove(i)
        unmatched_col_indices.remove(j)

    if results:
        matched_df = pd.DataFrame(results).sort_values(
            by=["Score_Ajustado", "Score_Total"],
            ascending=False,
        )
    else:
        matched_df = pd.DataFrame(columns=output_columns)

    unmatched_rows = []
    remaining_mapadrino_indices = set(range(len(mapadrinos))) - accepted_cols

    for i in range(len(mechon)):
        if i in accepted_rows:
            continue

        has_quality_option = bool(np.any(effective[i, :] >= MIN_EXPERIENCE_SCORE))
        if has_quality_option:
            alert = "sin_emparejar_por_cupo"
            reason = "Quedó sin cupo en el matching global."
        else:
            alert = "sin_emparejar_por_calidad_minima"
            reason = "No existe mapadrino con score suficiente para el piso mínimo de experiencia."

        # Para texto de alerta, sugerimos entre los mapadrinos libres.
        top_free = _build_top2_suggestion_fields(
            i,
            effective,
            mapadrinos,
            remaining_mapadrino_indices,
        )

        # Para columnas, mostramos siempre top-2 global del mechón.
        top_global = _build_top2_suggestion_fields(
            i,
            effective,
            mapadrinos,
            all_mapadrino_indices,
        )

        if top_free["Sugerencia_Padrino"]:
            reason = (
                f"{reason} Sugerencia fuera del mínimo: {top_free['Sugerencia_Padrino']} "
                f"(score_ajustado={top_free['Sugerencia_Score_Ajustado']})."
            )
            alert = f"{alert},sugerencia_fuera_minimo"

        unmatched_rows.append(
            {
                "Mechon": mechon.iloc[i]["Name"],
                "Padrino": "",
                "Score_Total": 0.0,
                "Score_Ajustado": 0.0,
                "Dominancia_Max": 0.0,
                "Multiplicador_Vital": 0.0,
                "Bonus_Cobertura": 0.0,
                "Penalizacion_Cola": 0.0,
                "Alertas": alert,
                "Justificacion": reason,
                **top_global,
            }
        )

    for j in range(len(mapadrinos)):
        if j in accepted_cols:
            continue
        unmatched_rows.append(
            {
                "Mechon": "",
                "Padrino": mapadrinos.iloc[j]["Name"],
                "Score_Total": 0.0,
                "Score_Ajustado": 0.0,
                "Dominancia_Max": 0.0,
                "Multiplicador_Vital": 0.0,
                "Bonus_Cobertura": 0.0,
                "Penalizacion_Cola": 0.0,
                "Alertas": "sin_emparejar_por_cupo",
                "Justificacion": "Quedó sin cupo en el matching global.",
                "Sugerencia_Padrino": "",
                "Sugerencia_Score_Ajustado": 0.0,
                "Sugerencia2_Padrino": "",
                "Sugerencia2_Score_Ajustado": 0.0,
            }
        )

    if not unmatched_rows:
        return matched_df[output_columns]

    unmatched_df = pd.DataFrame(unmatched_rows)
    return pd.concat([matched_df, unmatched_df], ignore_index=True)[output_columns]
