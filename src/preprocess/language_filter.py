# src/preprocess/language_filter.py
import pandas as pd
from langdetect import detect, DetectorFactory
from src.common.paths import pjoin
from src.common.logging import get_logger

DetectorFactory.seed = 0
log = get_logger("preprocess.lang")

# ---- Parámetros (ajusta aquí) ----
MIN_CHARS = 25  # mínimo de caracteres
DROP_DUP_BY = "texto"  # "texto" o "url"
KEEP_COLS = None  # None para mantener todas; o lista de columnas a conservar
# -----------------------------------


def is_spanish(t: str) -> bool:
    t = str(t)
    if len(t) < MIN_CHARS:
        return False
    try:
        return detect(t) == "es"
    except Exception:
        return False


def run():
    df = pd.read_csv(pjoin("data", "interim", "union.csv"))
    before = len(df)

    # Filtra por longitud
    df = df[df["texto"].fillna("").str.len() >= MIN_CHARS]

    # Filtra por idioma
    df = df[df["texto"].apply(is_spanish)]

    # Dedup por columna definida
    df.drop_duplicates(subset=[DROP_DUP_BY], inplace=True)

    if KEEP_COLS:
        df = df[KEEP_COLS]

    out = pjoin("data", "interim", "filtrado.csv")
    df.to_csv(out, index=False)
    log.info("Filtrado -> %s (antes=%d, después=%d)", out, before, len(df))

    # Resumen útil
    by_src = df.groupby("fuente").size().rename("cuenta").reset_index()
    by_src.to_csv(
        pjoin("data", "interim", "filtrado_resumen_por_fuente.csv"), index=False
    )


if __name__ == "__main__":
    run()
