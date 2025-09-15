# src/models/train_baseline.py
import argparse, json, os
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump
import matplotlib

matplotlib.use("Agg")  # evitar backends interactivos
import matplotlib.pyplot as plt
from src.common.paths import pjoin
from src.common.logging import get_logger

log = get_logger("models.baseline")


def make_groups(df: pd.DataFrame) -> pd.Series:
    """
    Construye un vector de grupos sin NaN para GroupShuffleSplit
    usando, en orden: video_id → package → url → fuente → id → índice de fila.
    Normaliza strings vacíos a NaN y fuerza dtype str.
    """
    cols = ["video_id", "package", "url", "fuente", "id"]
    g = pd.Series(pd.NA, index=df.index, dtype="object")
    for c in cols:
        if c in df.columns:
            s = df[c].astype("string").replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
            g = g.fillna(s)
    # Fallback final: índice de fila
    g = g.fillna(pd.Series([f"row_{i}" for i in range(len(df))], index=df.index))
    return g.astype(str)


def plot_confusion(cm, labels, out_png):
    plt.figure(figsize=(4.6, 3.8))
    plt.imshow(cm, interpolation="nearest")
    plt.xticks(range(len(labels)), labels, rotation=0)
    plt.yticks(range(len(labels)), labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.title("Matriz de confusión")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def run(args):
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    reports_dir = "reports"
    models_dir = "models"
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    df = pd.read_csv(pjoin("data", "interim", "limpio_final.csv")).dropna(
        subset=["texto_proc", "sentimiento"]
    )
    if len(df) < 200:
        log.warning(
            "Muy pocos ejemplos (%d). Considera reunir más datos o bajar --min-df.",
            len(df),
        )

    # === Split train/test (evitar fuga por grupos) ===
    groups = make_groups(df)
    try:
        gss = GroupShuffleSplit(
            n_splits=1, test_size=args.test_size, random_state=args.seed
        )
        tr_idx, te_idx = next(gss.split(df, groups=groups))
        tr, te = df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()
        split_note = "group"
    except Exception as e:
        log.warning(
            "Split por grupos falló (%s). Uso fallback estratificado por etiqueta.", e
        )
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=args.test_size, random_state=args.seed
        )
        tr_idx, te_idx = next(sss.split(df, df["sentimiento"]))
        tr, te = df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()
        split_note = "stratified_fallback"

    log.info(
        "Split=%s | train=%d | test=%d | grupos=%d",
        split_note,
        len(tr),
        len(te),
        pd.Series(groups).nunique(),
    )

    # === Modelo ===
    vec = TfidfVectorizer(
        min_df=args.min_df,
        max_df=args.max_df,
        ngram_range=(1, args.ngram),
        sublinear_tf=True,
        strip_accents="unicode",  # normaliza tildes (español)
    )

    if args.model == "svm":
        clf = LinearSVC(class_weight="balanced")
        model_name = f"baseline_svm_{ts}"
    else:
        clf = LogisticRegression(max_iter=500, class_weight="balanced")
        model_name = f"baseline_lr_{ts}"

    pipe = Pipeline([("tfidf", vec), ("clf", clf)])
    pipe.fit(tr["texto_proc"], tr["sentimiento"])

    # === Evaluación ===
    preds = pipe.predict(te["texto_proc"])
    labels = sorted(df["sentimiento"].dropna().unique().tolist())
    cm = confusion_matrix(te["sentimiento"], preds, labels=labels)

    rep_dict = classification_report(
        te["sentimiento"], preds, labels=labels, digits=3, output_dict=True
    )

    # === Guardados ===
    pd.DataFrame(rep_dict).to_csv(pjoin(reports_dir, f"{model_name}_report.csv"))
    with open(
        pjoin(reports_dir, f"{model_name}_report.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(rep_dict, f, ensure_ascii=False, indent=2)

    pd.DataFrame(cm, index=labels, columns=labels).to_csv(
        pjoin(reports_dir, f"{model_name}_confusion.csv")
    )
    plot_confusion(cm, labels, pjoin(reports_dir, f"{model_name}_confusion.png"))

    # Errores (misclasificaciones)
    err = te.copy()
    err["pred"] = preds
    err_err = err[err["pred"] != err["sentimiento"]]
    cols = [
        c
        for c in [
            "id",
            "fuente",
            "dieta_heuristica",
            "video_id",
            "package",
            "url",
            "texto_proc",
            "sentimiento",
            "pred",
        ]
        if c in err_err.columns
    ]
    err_err[cols].to_csv(pjoin(reports_dir, f"{model_name}_errors.csv"), index=False)

    # Modelo persistido
    dump(pipe, pjoin(models_dir, f"{model_name}.joblib"))

    log.info("Listo. Reportes en %s, modelo en %s", reports_dir, models_dir)
    print(f"Split: {split_note}")
    print(f"Macro-F1: {rep_dict['macro avg']['f1-score']:.3f}")
    print(f"Reporte  -> {pjoin(reports_dir, f'{model_name}_report.csv')}")
    print(f"Matriz   -> {pjoin(reports_dir, f'{model_name}_confusion.png')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lr", "svm"], default="lr")
    ap.add_argument("--min-df", type=int, default=5)
    ap.add_argument("--max-df", type=float, default=0.99)
    ap.add_argument("--ngram", type=int, default=2, help="1=unigram, 2=uni+bi")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run(args)
