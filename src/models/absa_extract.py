# src/models/absa_extract.py
import os, re, pandas as pd, numpy as np
from src.common.paths import pjoin
from src.common.logging import get_logger

log = get_logger("models.absa")

# === Léxicos de aspectos (ajústalos si quieres) ===
ASPECTOS = {
    "hambre": {
        "hambre",
        "apetito",
        "antojo",
        "antojos",
        "saciedad",
        "ansia",
        "craving",
        "cravings",
        "picoteo",
    },
    "energía": {
        "energía",
        "energia",
        "fatiga",
        "cansancio",
        "vitalidad",
        "ánimo",
        "animo",
        "sueño",
        "sueno",
    },
    "adherencia": {
        "adherencia",
        "sostener",
        "sostenible",
        "abandono",
        "constancia",
        "rutina",
        "compromiso",
        "seguir",
        "seguimiento",
    },
    "costo": {"caro", "carísimo", "carisimo", "costoso", "barato", "precio", "gasto"},
    "social": {
        "social",
        "salir",
        "eventos",
        "reunión",
        "reunion",
        "familia",
        "amigos",
        "restaurante",
        "fiesta",
    },
}

# === Patrones de dieta (mismos del paso 5) ===
PATS = {
    "keto": [r"\bketo\b", r"\bcetog(e|é)nica\b", r"\blow\s*carb\b"],
    "ayuno": [r"\bayuno\b", r"\bintermitente\b", r"\b16/8\b", r"\b18/6\b", r"\bOMAD\b"],
    "flexible": [r"\bflexible\b", r"\bIIFYM\b", r"\bcontar\s*macros\b"],
    "mediterranea": [
        r"\bmediterr(a|á)nea\b",
        r"\balimentaci[oó]n\s*mediterr(a|á)nea\b",
    ],
    "paleo": [r"\bpaleo\b", r"\bpaleol[ií]tica\b"],
    "vegana": [r"\bvegana\b", r"\bvegano\b", r"\bvegetariana\b", r"\bplant\s*based\b"],
}

POS_KW = [
    "recomiendo",
    "me encantó",
    "me encanto",
    "me fue bien",
    "me funcionó",
    "me funciono",
    ":smile:",
    ":muscle:",
    ":fire:",
    "excelente",
    "mejoró",
    "mejoro",
    "mejor",
]
NEG_KW = [
    "horrible",
    "malo",
    "pésimo",
    "pesimo",
    "fatal",
    "no_recomendar",
    "no lo recomiendo",
    ":cry:",
    ":thumbsdown:",
    "mareo",
    "mareos",
    "dolor",
    "no_pude",
    "no_funciona",
    "no_funcionó",
]


def frases(t: str):
    return [
        s.strip() for s in re.split(r"[.!?]\s+", str(t)) if s and len(s.strip()) > 0
    ]


def rule_sentiment(s: str):
    s = s.lower()
    pos = any(k in s for k in POS_KW)
    neg = (
        any(k in s for k in NEG_KW) or "no_" in s
    )  # favorece detectar negación del prepro
    if pos and not neg:
        return "pos"
    if neg and not pos:
        return "neg"
    return "neu"


def infer_diet(text: str) -> str | None:
    if not isinstance(text, str):
        return None
    for diet, pats in PATS.items():
        for pat in pats:
            if re.search(pat, text, flags=re.IGNORECASE):
                return diet
    return None


def load_input_df():
    # preferencia de archivos
    candidates = [
        pjoin("data", "interim", "labeled.csv"),  # debería tener dieta_heuristica
        pjoin("data", "interim", "limpio_final.csv"),  # puede o no tenerla
        pjoin("data", "interim", "limpio.csv"),  # limpio básico
    ]
    for path in candidates:
        if os.path.exists(path):
            df = pd.read_csv(path)
            log.info("Leyendo %s (%d filas)", path, len(df))
            return df
    raise FileNotFoundError(
        "No encontré ninguno de: labeled.csv, limpio_final.csv, limpio.csv en data/interim/"
    )


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Necesitamos texto_proc y dieta_heuristica
    if "texto_proc" not in df.columns:
        # Fallback: usa texto_raw si existe
        if "texto_raw" in df.columns:
            df["texto_proc"] = df["texto_raw"].astype(str).str.lower()
            log.warning("No había texto_proc; usando texto_raw como proxy (sin lemas).")
        else:
            raise KeyError("Falta columna texto_proc (o texto_raw).")
    if "dieta_heuristica" not in df.columns:
        # inferir a partir de texto (proc o raw)
        base_text = df["texto_proc"].fillna(df.get("texto_raw", "").astype(str))
        df["dieta_heuristica"] = base_text.apply(infer_diet)
        n_inferred = df["dieta_heuristica"].notna().sum()
        log.warning(
            "No había dieta_heuristica; inferidas %d filas por patrones.", n_inferred
        )
    # Limpia
    df = df.dropna(subset=["texto_proc", "dieta_heuristica"]).copy()
    return df


def run():
    df = load_input_df()
    df = ensure_columns(df)

    rows = []
    for _, r in df.iterrows():
        diet = r["dieta_heuristica"]
        for s in frases(r["texto_proc"]):
            toks = s.split()
            presentes = [
                a for a, lex in ASPECTOS.items() if any(w in toks for w in lex)
            ]
            if presentes:
                pred = rule_sentiment(s)
                for a in presentes:
                    rows.append(
                        {"dieta": diet, "frase": s, "aspecto": a, "pred_sent": pred}
                    )

    if not rows:
        log.info(
            "No se detectaron frases con aspectos. Revisa léxicos o columnas de entrada."
        )
        return

    absa = pd.DataFrame(rows)

    # score = %pos - %neg
    def score_func(s):
        s = list(s)
        return np.mean(np.array(s) == "pos") - np.mean(np.array(s) == "neg")

    pivot = absa.pivot_table(
        index="aspecto", columns="dieta", values="pred_sent", aggfunc=score_func
    ).fillna(0.0)
    counts = absa.groupby(["aspecto", "dieta"]).size().unstack(fill_value=0)

    os.makedirs("reports", exist_ok=True)
    pivot.to_csv(pjoin("reports", "matriz_dieta_aspecto.csv"))
    counts.to_csv(pjoin("reports", "matriz_dieta_aspecto_counts.csv"))
    print("Scores  -> reports/matriz_dieta_aspecto.csv")
    print("Counts  -> reports/matriz_dieta_aspecto_counts.csv")

    # ejemplos por celda (top 3 pos/neg)
    examples = []
    for (a, d), g in absa.groupby(["aspecto", "dieta"]):
        gp = g[g["pred_sent"] == "pos"].head(3)["frase"].tolist()
        gn = g[g["pred_sent"] == "neg"].head(3)["frase"].tolist()
        examples.append({"aspecto": a, "dieta": d, "ej_pos": gp, "ej_neg": gn})
    pd.DataFrame(examples).to_csv(pjoin("reports", "absa_ejemplos.csv"), index=False)
    print("Ejemplos -> reports/absa_ejemplos.csv")


if __name__ == "__main__":
    run()
