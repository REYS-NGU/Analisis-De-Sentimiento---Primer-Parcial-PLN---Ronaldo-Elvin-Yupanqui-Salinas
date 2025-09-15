# src/preprocess/clean_es.py (versión robusta)
import pandas as pd, spacy
from src.common.paths import pjoin, load_config
from src.common.utils import basic_clean, replace_emojis, marcar_negacion_spacy
from src.common.logging import get_logger

log = get_logger("preprocess.clean_es")

CHUNKSIZE = 10_000  # ajusta a tu RAM
N_PROCESS = 2  # núcleos para spaCy; 0 o 1 si no quieres paralelo
BATCH_SIZE = 200


def run():
    cfg = load_config()
    in_path = pjoin("data", "interim", "filtrado.csv")
    out_path = pjoin("data", "interim", "limpio.csv")

    nlp = spacy.load("es_core_news_md", disable=["ner", "textcat"])
    # procesa por lotes
    writer = None
    total = 0
    for chunk in pd.read_csv(in_path, chunksize=CHUNKSIZE):
        # texto_raw
        chunk["texto_raw"] = chunk["texto"].map(
            lambda s: replace_emojis(basic_clean(str(s)))
        )
        # texto_proc con nlp.pipe
        docs = nlp.pipe(
            chunk["texto_raw"].tolist(), batch_size=BATCH_SIZE, n_process=N_PROCESS
        )
        chunk["texto_proc"] = [
            marcar_negacion_spacy(doc, cfg["preprocess"]["negation_window"])
            for doc in docs
        ]

        # append incremental
        if writer is None:
            chunk.to_csv(out_path, index=False)
            writer = True
        else:
            chunk.to_csv(out_path, index=False, mode="a", header=False)
        total += len(chunk)
        log.info("Procesadas %d filas...", total)

    log.info("Guardado %s", out_path)


if __name__ == "__main__":
    run()
