import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .scoring import calculate_match_components, SCORE_FLOOR


def match_algorithm(mechon: pd.DataFrame, mapadrinos: pd.DataFrame) -> pd.DataFrame:
    """Aplica asignación uno-a-uno maximizando puntaje ajustado global."""
    matrix = np.zeros((len(mechon), len(mapadrinos)))

    # La librería resuelve minimización, por eso se guarda el negativo del score.
    for i in range(len(mechon)):
        m = mechon.iloc[i]
        for j in range(len(mapadrinos)):
            p = mapadrinos.iloc[j]
            comp = calculate_match_components(m, p)
            matrix[i, j] = -comp["effective_total"]

    row_idx, col_idx = linear_sum_assignment(matrix)

    results = []
    for i, j in zip(row_idx, col_idx):
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
            }
        )

    result_df = pd.DataFrame(results)
    return result_df.sort_values(
        by=["Score_Ajustado", "Score_Total"],
        ascending=False,
    )
