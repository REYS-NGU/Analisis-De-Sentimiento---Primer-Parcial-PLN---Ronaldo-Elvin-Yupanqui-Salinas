# src/models/topics_lda.py
import os, sys, math
import pandas as pd
from gensim import corpora, models
from src.common.paths import pjoin
from src.common.logging import get_logger

log = get_logger("models.lda")

# Parámetros rápidos
NUM_TOPICS = 10
PASSES = 6
NO_BELOW = 5  # palabra debe aparecer en ≥ 5 docs
NO_ABOVE = 0.30  # y en ≤ 30% de docs
KEEP_N = 50000


def pick_input():
    for name in ["labeled.csv", "limpio_final.csv", "limpio.csv"]:
        path = pjoin("data", "interim", name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "No encontré data/interim/(labeled|limpio_final|limpio).csv"
    )


def run():
    path = pick_input()
    df = pd.read_csv(path)
    if "texto_proc" not in df.columns:
        raise KeyError(f"{path} no tiene columna texto_proc")
    texts = [str(t).split() for t in df["texto_proc"].fillna("")]
    n_docs = len(texts)
    non_empty = sum(1 for t in texts if len(t) > 0)
    log.info("Archivo: %s | docs=%d | docs_no_vacios=%d", path, n_docs, non_empty)
    if non_empty == 0:
        raise RuntimeError("Todos los textos están vacíos para LDA (revisa clean_es).")

    # Diccionario y filtro
    dic = corpora.Dictionary(texts)
    log.info("Vocabulario inicial: %d términos", len(dic))
    dic.filter_extremes(no_below=NO_BELOW, no_above=NO_ABOVE, keep_n=KEEP_N)
    log.info("Vocabulario tras filtro: %d términos", len(dic))
    if len(dic) == 0:
        raise RuntimeError(
            "Vocabulario quedó vacío tras filter_extremes. Baja NO_BELOW o sube NO_ABOVE."
        )

    corpus = [dic.doc2bow(t) for t in texts]
    nnz_docs = sum(1 for bow in corpus if len(bow) > 0)
    if nnz_docs == 0:
        raise RuntimeError(
            "Todos los bow están vacíos. Revisa los parámetros de filtro."
        )

    # Entrena LDA (silencioso pero rápido)
    lda = models.LdaModel(
        corpus=corpus,
        id2word=dic,
        num_topics=NUM_TOPICS,
        passes=PASSES,
        random_state=42,
        alpha="auto",
        eta="auto",
        eval_every=None,  # quita cálculo de perplexidad en cada pass
        minimum_probability=0.0,
        minimum_phi_value=0.0,
    )

    # Exporta términos top
    rows = []
    for k in range(NUM_TOPICS):
        terms = ", ".join(w for w, _ in lda.show_topic(k, topn=10))
        rows.append({"topic": k, "terms": terms})
    out_terms = pjoin("reports", "topics_global.csv")
    pd.DataFrame(rows).to_csv(out_terms, index=False)

    # 2 ejemplos por tópico (docs con mayor probabilidad del tópico)
    examples = []
    for k in range(NUM_TOPICS):
        scores = []
        for i, bow in enumerate(corpus):
            dist = lda.get_document_topics(bow, minimum_probability=0.0)
            prob_k = dict(dist).get(k, 0.0)
            scores.append((prob_k, i))
        scores.sort(reverse=True)
        top_idx = [i for _, i in scores[:2]]
        for rank, idx in enumerate(top_idx, 1):
            examples.append(
                {"topic": k, "rank": rank, "texto_proc": " ".join(texts[idx])[:300]}
            )
    out_ex = pjoin("reports", "topics_global_examples.csv")
    pd.DataFrame(examples).to_csv(out_ex, index=False)

    print(f"Tópicos -> {out_terms}")
    print(f"Ejemplos -> {out_ex}")


if __name__ == "__main__":
    run()
