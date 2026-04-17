# Mapadrinamiento TEL 2026

Script para emparejar mechones con mapadrinos maximizando afinidad global.

## Quick Start

### 1) Clonar repositorio

```bash
git clone https://github.com/michifeli/mapadrinamiento.git
cd mapadrinamiento
```

### 2) Crear entorno e instalar dependencias

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3) Preparar datos

- Deja el Excel de respuestas dentro de la carpeta `data/`.
- El script lee el primer archivo que coincida con `data/*.xlsx`.

### 4) Configurar `.env` (opcional, para apoyo IA)

Si quieres usar IA en casos ambiguos, crea un archivo `.env` en la raíz con:

```env
USE_AI=1
API_KEY=tu_token
HF_MODEL=meta-llama/Llama-3.1-8B-Instruct
AI_ALLOWED_CATEGORIES=Pref
AI_MAX_CALLS=12
AI_MIN_CONFIDENCE_TO_SKIP=0.85
```

Si no configuras IA, el sistema funciona 100% local (determinístico).

### 5) Ejecutar matching

```bash
python main.py
```

Se generan:

- `match.csv`: resultado final de emparejamientos.
- `reporte_ia.csv`: trazabilidad del saneamiento de respuestas.

### 6) Ejecutar tests

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

## Estructura del proyecto

```text
mapadrinamiento/
├── main.py                      -> Punto de entrada. Ejecuta todo el flujo.
├── requirements.txt             -> Dependencias del proyecto.
├── README.md                    -> Guía de uso y explicación general.
├── .env                         -> Configuración local (opcional, no versionar con claves).
│
├── data/
│   └── *.xlsx                   -> Archivo(s) fuente con respuestas del formulario.
│
├── src/
│   ├── __init__.py              -> Exporta funciones principales del módulo.
│   ├── config.py                -> Lee variables de entorno.
│   ├── catalogs.py              -> Catálogos oficiales y aliases.
│   ├── text_normalization.py    -> Limpieza y tokenización de texto.
│   ├── semantic_mapper.py       -> Mapeo semántico (local + IA opcional).
│   ├── data_pipeline.py         -> Lectura del Excel y preprocesamiento.
│   ├── scoring.py               -> Cálculo de similitudes y score.
│   └── matching.py              -> Matching global (algoritmo húngaro).
│
├── tests/
│   └── test_main.py             -> Pruebas unitarias base.
│
├── match.csv                    -> Salida final de emparejamientos (se genera al correr).
└── reporte_ia.csv               -> Log de saneamiento/mapeo (se genera al correr).
```

## Matemática

### Problema

Queremos asignar cada mechón a un mapadrino maximizando el puntaje total de afinidad.

**Me queda de tarea explicar la matematica**
