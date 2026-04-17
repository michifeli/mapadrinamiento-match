from src.catalogs import OFFICIAL_OPTIONS
from src.config import load_app_config
from src.data_pipeline import preprocess_data, read_data
from src.matching import match_algorithm
from src.scoring import calculate_match_components
from src.semantic_mapper import create_mapper_state, reset_mapper_state
from src.text_normalization import normalize_token


_MAPPER_STATE: dict | None = None


def build_mapper_from_env() -> dict:
    """Crea una vez el estado del mapeador."""
    global _MAPPER_STATE
    if _MAPPER_STATE is None:
        _MAPPER_STATE = create_mapper_state(load_app_config())
    return _MAPPER_STATE


def print_results(resultados) -> None:
    """Muestra resultados en consola en formato tabla."""
    print("\n" + "=" * 70)
    print(f"{'MECHON':<30} | {'PADRINO':<30} | {'RAW':<6} | {'AJUST':<6} | ALERTAS")
    print("=" * 70)

    for _, fila in resultados.iterrows():
        print(
            f"{fila['Mechon'][:30]:<30} | {fila['Padrino'][:30]:<30} | "
            f"{fila['Score_Total']:<6} | {fila['Score_Ajustado']:<6} | {fila['Alertas']}"
        )
        print(f"   Detalle: {fila['Justificacion']}")
        print("-" * 70)


def run_pipeline() -> None:
    """Ejecuta el flujo completo del script."""
    print("Paso 1/4: leyendo data...")
    datos_crudos = read_data("data/*.xlsx")

    print("Paso 2/4: limpiando y normalizando respuestas...")
    mapper_state = build_mapper_from_env()
    reset_mapper_state(mapper_state)
    mechones, padrinos = preprocess_data(datos_crudos, mapper_state)

    print("Paso 3/4: calculando emparejamientos...")
    resultados = match_algorithm(mechones, padrinos)

    print("Paso 4/4: guardando resultados...")
    resultados.to_csv("match.csv", index=False, encoding="utf-8-sig")

    print_results(resultados)

    print("\nListo. Archivos generados:")
    print("- match.csv")
    print("- reporte_ia.csv")

if __name__ == "__main__":
    run_pipeline()