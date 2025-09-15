# src/collect/merge_sources.py
import pandas as pd
from glob import glob
from datetime import datetime
from src.common.paths import pjoin
from src.common.logging import get_logger

log = get_logger("collect.merge")

CANONICAL = [
    "id",
    "fuente",
    "url",
    "fecha",
    "texto",
    "rating",
    "dieta",
    "video_id",
    "package",
    "pais",
    "lang",
]


def coerce_cols(df, path):
    # crear columnas faltantes
    for c in CANONICAL:
        if c not in df.columns:
            df[c] = None
    # tipados básicos
    df["texto"] = df["texto"].astype(str)
    # rating numérico (si existe)
    try:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    except Exception:
        df["rating"] = None

    # fecha a ISO
    def to_iso(x):
        if pd.isna(x):
            return None
        try:
            return pd.to_datetime(x, utc=True, errors="coerce").isoformat()
        except Exception:
            return None

    df["fecha"] = df["fecha"].apply(to_iso)
    # metadata
    df["source_file"] = path
    df["length_chars"] = df["texto"].str.len()
    # recorta a columnas canónicas + auxiliares
    keep = CANONICAL + ["source_file", "length_chars"]
    return df[keep]


def run():
    files = glob(str(pjoin("data", "raw", "*.csv")))
    if not files:
        log.info("No hay archivos en data/raw.")
        return
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df = coerce_cols(df, f)
            dfs.append(df)
            log.info("OK: %s (%d)", f, len(df))
        except Exception as e:
            log.warning("Error leyendo %s: %s", f, e)

    df = pd.concat(dfs, ignore_index=True)
    # dedup suave por texto + url, mantén el más largo
    df.sort_values("length_chars", ascending=False, inplace=True)
    df = df.drop_duplicates(subset=["texto", "url"], keep="first")

    out = pjoin("data", "interim", "union.csv")
    df.to_csv(out, index=False)
    log.info("Guardado %s (filas=%d, de %d archivos)", out, len(df), len(files))

    # Resumen útil
    summary = df.groupby("fuente").size().rename("cuenta").reset_index()
    summary_out = pjoin("data", "interim", "union_resumen_por_fuente.csv")
    summary.to_csv(summary_out, index=False)
    log.info("Resumen -> %s", summary_out)


if __name__ == "__main__":
    run()
