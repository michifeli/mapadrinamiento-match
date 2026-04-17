import glob
import pandas as pd

from .catalogs import OFFICIAL_OPTIONS
from .semantic_mapper import normalize_with_reasoning


COLUMNS_BY_ROLE = {
    "Mechón": {
        "name": "Dinos tu nombre y apellido\xa0",
        "email": "Indícanos tu correo USM\xa0",
        "suffix": "",
    },
    "Padrino": {
        "name": "Dinos tu nombre y apellido",
        "email": "Indícanos tu correo USM",
        "suffix": "2",
    },
}

ROLE_COLUMN = (
    "Indícanos si eres mechón / mechona o de generación anterior 🤔 "
    "(Si te cambiaste a TEL este año, también cuentas como mechón/a 😊)"
)

NORMALIZABLE_COLUMNS = [
    "Deportes",
    "Juegos",
    "Series",
    "Pref",
    "Comida",
    "Musica",
    "Dieta",
    "Trago",
    "Hobby",
    "Idiomas",
]


def read_data(file_path: str) -> pd.DataFrame:
    """Lee el primer Excel que coincide con el patrón entregado."""
    archivos = glob.glob(file_path)
    if not archivos:
        raise FileNotFoundError("No se encontró el archivo Excel.")
    return pd.read_excel(archivos[0])


def _get_role_info(role_value: str) -> tuple[str, str, str, str]:
    """Devuelve (rol, sufijo, columna nombre, columna correo)."""
    if "Mechón" in role_value:
        cfg = COLUMNS_BY_ROLE["Mechón"]
        return "Mechón", cfg["suffix"], cfg["name"], cfg["email"]
    cfg = COLUMNS_BY_ROLE["Padrino"]
    return "Padrino", cfg["suffix"], cfg["name"], cfg["email"]


def _build_unified_row(row, role_label: str, suffix: str, name_col: str, email_col: str) -> dict:
    """Normaliza una fila de origen a la estructura interna del proyecto."""
    series_question = f"Eres fan de las series{'/películas' if suffix else ''} de..."
    musica_question = f"Tipo de música favorita (Elige {'hasta ' if suffix else ''}tu top 3)"

    return {
        "Role": role_label,
        "Name": row[name_col],
        "Email": row[email_col],
        "Deportes": str(row[f"¿Qué deportes te gusta practicar?{suffix}"]),
        "Juegos": str(row[f"¿Juegas alguno de estos juegos?{suffix}"]),
        "Series": str(row[series_question]),
        "Pref": str(row[f"Me gusta mas...{suffix}"]),
        "Comida": str(row[f"¿Cuál de estas comidas te parece mas deliciosa? (Elige tu top 3){suffix}"]),
        "Musica": str(row[musica_question]),
        "Dieta": str(row[f"Tipo de dieta{suffix}"]),
        "Trago": str(row[f"Tienes algún trago favorito? (Elige tu top 3){suffix}"]),
        "Hobby": str(row[f"¿Tienes algún Hobby?{suffix}"]),
        "Idiomas": str(row[f"¿Cuál de estos idiomas sabes o te gustaría aprender?\xa0{suffix}"]),
    }


def preprocess_data(df: pd.DataFrame, mapper_state: dict):
    """Prepara datos para matching y genera trazabilidad de normalización."""
    unified_data = []

    # Construcción del dataframe unificado con columnas homogéneas.
    for _, row in df.iterrows():
        role = row[ROLE_COLUMN]
        if pd.isna(role):
            continue

        role_label, suffix, name_col, email_col = _get_role_info(str(role))
        unified_data.append(
            _build_unified_row(row, role_label, suffix, name_col, email_col)
        )

    df_unified = pd.DataFrame(unified_data).replace("nan", "")
    ai_log = []

    mode_label = "determinístico"
    config = mapper_state["config"]
    if config["use_ai"] and config["api_key"]:
        mode_label += " + apoyo de modelo"
    print(f"Normalizando respuestas ({mode_label})...")

    # Normalización por columna para reutilizar caché de respuestas repetidas.
    for col in NORMALIZABLE_COLUMNS:
        print(f"  Procesando columna: {col}")
        valid_opts = OFFICIAL_OPTIONS.get(col, [])
        respuestas_unicas = df_unified[col].unique()
        map_dict = {}

        for res in respuestas_unicas:
            if not res or res == "":
                continue

            clean_res, razon, metodo, conf = normalize_with_reasoning(
                mapper_state,
                res,
                valid_opts,
                col,
            )
            map_dict[res] = clean_res

            if str(res).strip() != str(clean_res).strip() or metodo != "local":
                ai_log.append(
                    {
                        "Columna": col,
                        "Original": res,
                        "Saneado": clean_res,
                        "Metodo": metodo,
                        "Confianza": round(conf, 2),
                        "Razonamiento": razon,
                    }
                )

        df_unified[col] = df_unified[col].map(map_dict).fillna(df_unified[col])

    pd.DataFrame(ai_log).to_csv(
        "reporte_ia.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Separación final para algoritmo de matching.
    mechones = df_unified[df_unified["Role"] == "Mechón"].reset_index(drop=True)
    padrinos = df_unified[df_unified["Role"] == "Padrino"].reset_index(drop=True)
    return mechones, padrinos