# dietas-sentimiento

Análisis de sentimiento y ABSA en dietas (keto, ayuno intermitente, flexible) en español.

## Estructura
```
dietas-sentimiento/
 ├─ data/
 │   ├─ raw/              # CSV/JSON originales por fuente (sin tocar)
 │   ├─ interim/          # limpios pero sin etiquetas finales
 │   └─ processed/        # listos para modelar (train/val/test)
 ├─ notebooks/
 ├─ src/
 │   ├─ collect/          # scrapers/APIs
 │   ├─ preprocess/       # limpieza, normalización, lematización
 │   ├─ features/         # TF-IDF, tokenizado HF
 │   └─ models/           # entrenamiento y evaluación
 ├─ reports/              # figuras, tablas
 ├─ config/
 └─ requirements.txt
```

## Instalación
```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# Si falla spaCy model:
# python -m spacy download es_core_news_md
```

## Colecta de datos (opciones)
- **YouTube**: comentarios en español con `python -m src.collect.youtube_collect`
  - Coloca tu `YOUTUBE_API_KEY` en `.env`
- **Google Play**: reseñas de apps con `python -m src.collect.google_play_collect`
  - Agrega PACKAGE_IDS de apps de ayuno/keto/flexible en el script
- **Blogs/foros (RSS)**: `python -m src.collect.rss_collect` (edita la lista FEEDS)

Luego une todo:
```bash
python -m src.collect.merge_sources
```
