# src/preprocess/label_sentimiento.py
import re
import pandas as pd
from pathlib import Path

# --- Entrada/Salida ---
# Prioriza labeled.csv (trae 'dieta_heuristica'); si no existe, usa limpio.csv
CANDIDATES = [Path("data/interim/labeled.csv"), Path("data/interim/limpio.csv")]
IN = next((p for p in CANDIDATES if p.exists()), Path("data/interim/limpio.csv"))

OUT = Path("data/interim/limpio_sent.csv")
SAMPLE_BAL = Path("data/interim/para_anotar_balanceado.csv")
SAMPLE_HARD = Path("data/interim/para_anotar_dificiles.csv")
SAMPLE_BAL_CIEGO = Path("data/interim/para_anotar_balanceado_ciego.csv")

print(f">> label_sentimiento leyendo de: {IN}")

# ---------- Léxicos y reglas ----------
# Frases/lemmas positivos/negativos en texto_proc (con negación "no_" posible)
POS_KW = {
    "recomiendo",
    "recomendable",
    "me_encantar",
    "me_encanto",
    "excelente",
    "funcionar",
    "me_funciono",
    "me_fue_bien",
    "mejorar",
    "progreso",
    "genial",
    "buenisimo",
    "bueno",
    "efectivo",
    "eficaz",
    ":smile:",
    ":muscle:",
    ":fire:",
    ":thumbsup:",
    ":grinning:",
    ":heart:",
}
NEG_KW = {
    "horrible",
    "malo",
    "pesimo",
    "pésimo",
    "fatal",
    "terrible",
    "fracaso",
    "mareo",
    "dolor",
    "ansiedad",
    "no_recomendar",
    "no_servir",
    "no_funcionar",
    "no_bueno",
    "no_efectivo",
    "abandono",
    "difícil",
    "dificil",
    "imposible",
    "caro",
    "carísimo",
    "carisimo",
    ":cry:",
    ":thumbsdown:",
    ":angry:",
    ":weary:",
    ":frowning:",
}

# Indicadores de contradicción / giro
CONTRAST = {"pero", "aunque", "sin_embargo", "no_obstante"}

# Intensificadores y atenuadores (afectan confianza)
INTENSIF = {"muy", "super", "súper", "bastante", "demasiado", "re_muy"}
ATENUAD = {"un_poco", "algo", "ligeramente"}

# Expresiones interrogativas comunes -> probable "neu" si no hay polares
QUESTION_PAT = re.compile(r"[¿?]")


def has_any(tokens, vocab):
    return any(t in vocab for t in tokens)


def rating_to_sent(r):
    try:
        r = int(r)
        if r in (1, 2):
            return "neg"
        if r == 3:
            return "neu"
        if r in (4, 5):
            return "pos"
    except Exception:
        pass
    return None


def split_clauses(s: str):
    # divide por conectores y puntuación fuerte
    return re.split(
        r"(?:\s(?:pero|aunque|sin embargo|no obstante)\s|[.!?])", s, flags=re.IGNORECASE
    )


def heur_sent(text_proc: str, text_raw: str):
    """
    Devuelve (label, conf, why)
    - label: pos/neg/neu/None
    - conf: 0..1
    - why: breve motivo para auditoría
    """
    s = (text_proc or "").lower().strip()
    if not s:
        return None, 0.0, "empty"

    tokens = s.split()
    pos = has_any(tokens, POS_KW)
    neg = has_any(tokens, NEG_KW)
    is_question = bool(QUESTION_PAT.search(text_raw or ""))

    # regla de negación: si hay "no_" en muchos tokens, sesgo a negativo
    negation_bias = sum(1 for t in tokens if t.startswith("no_"))

    # Contraste: mira última cláusula
    clauses = [c.strip() for c in split_clauses(s) if c and c.strip()]
    last_clause = clauses[-1] if clauses else s
    last_toks = last_clause.split()
    last_pos = has_any(last_toks, POS_KW)
    last_neg = has_any(last_toks, NEG_KW)

    # Intensificadores/Atenuadores
    intens = has_any(tokens, INTENSIF)
    atten = has_any(tokens, ATENUAD)

    # Reglas
    if pos and not neg:
        base = "pos"
    elif neg and not pos:
        base = "neg"
    elif last_pos and not last_neg:
        base = "pos"
    elif last_neg and not last_pos:
        base = "neg"
    else:
        base = "neu"

    # Ajustes por interrogación sin polaridad
    if base == "neu" and is_question and not (pos or neg):
        return "neu", 0.4, "question_neutral"

    # Ajuste por negación fuerte
    if base == "pos" and negation_bias >= 2:
        base = "neg"

    # Confianza
    conf = 0.6
    if pos ^ neg:  # solo uno presente
        conf = 0.75
    if intens:
        conf += 0.1
    if atten:
        conf -= 0.1
    if last_pos ^ last_neg:
        conf += 0.05
    if negation_bias >= 2:
        conf += 0.05
    conf = max(0.0, min(1.0, conf))

    why = f"base={base}, pos={pos}, neg={neg}, last_pos={last_pos}, last_neg={last_neg}, neg_bias={negation_bias}, intens={intens}, atten={atten}"
    return base, conf, why


def main():
    df = pd.read_csv(IN)

    # Asegura columna de dieta para no romper groupby
    if "dieta_heuristica" not in df.columns:
        df["dieta_heuristica"] = "sin_dieta"
    else:
        df["dieta_heuristica"] = df["dieta_heuristica"].fillna("sin_dieta").astype(str)

    # 1) etiqueta por rating si existe
    df["sentimiento"] = df["rating"].apply(rating_to_sent)

    # 2) aplica heurística donde falta
    mask = df["sentimiento"].isna()
    tmp = df.loc[mask, ["texto_proc", "texto_raw"]].fillna("")
    res = tmp.apply(lambda r: heur_sent(r["texto_proc"], r["texto_raw"]), axis=1)
    df.loc[mask, "sentimiento"] = [x[0] for x in res]
    df.loc[mask, "sent_conf"] = [x[1] for x in res]
    df.loc[mask, "sent_why"] = [x[2] for x in res]

    # donde había rating, confianza alta:
    df.loc[~mask, "sent_conf"] = 0.9
    df.loc[~mask, "sent_why"] = "rating"

    # 3) guarda limpio_sent
    df.to_csv(OUT, index=False)
    print(f"Guardado {OUT} ({len(df)} filas)")

    # 4) crear muestra balanceada dieta × sent_prov
    base = df.copy()
    base["sent_prov"] = base["sentimiento"].fillna("unk")
    # (opcional) excluir "sin_dieta" si no quieres muestrearla:
    # base = base[base["dieta_heuristica"] != "sin_dieta"]

    N_PER_CELL = 25
    parts = []
    for (d, s), g in base.groupby(["dieta_heuristica", "sent_prov"]):
        if len(g) == 0:
            continue
        k = min(N_PER_CELL, len(g))
        parts.append(g.sample(k, random_state=42))

    if parts:
        balanced = pd.concat(parts, ignore_index=True).drop_duplicates("id")
        # Con y sin “pistas”
        balanced["sentimiento_gold"] = ""
        balanced[
            [
                "id",
                "texto_raw",
                "texto_proc",
                "dieta_heuristica",
                "sent_prov",
                "sentimiento_gold",
            ]
        ].to_csv(SAMPLE_BAL, index=False)
        # versión ciega para evitar sesgo (sin dieta_heuristica/sent_prov)
        balanced_blind = balanced[["id", "texto_raw", "texto_proc"]].copy()
        balanced_blind["sentimiento_gold"] = ""
        balanced_blind.to_csv(SAMPLE_BAL_CIEGO, index=False)
        print(f"Muestra balanceada -> {SAMPLE_BAL}")
        print(f"Muestra balanceada (ciega) -> {SAMPLE_BAL_CIEGO}")
    else:
        print("No se pudo crear muestra balanceada (faltan datos).")

    # 5) muestra de “difíciles”: baja confianza o contradicción
    def is_contradictory(s):
        s = str(s or "")
        return any(c in s for c in ["pero", "aunque", "sin embargo", "no obstante"])

    hard = df[
        (df["sent_conf"].fillna(0) <= 0.6) | df["texto_proc"].apply(is_contradictory)
    ]
    hard = hard.sample(min(400, len(hard)), random_state=42) if len(hard) > 0 else hard
    if len(hard) > 0:
        hard = hard[
            [
                "id",
                "texto_raw",
                "texto_proc",
                "dieta_heuristica",
                "sentimiento",
                "sent_conf",
                "sent_why",
            ]
        ]
        hard["sentimiento_gold"] = ""
        hard.to_csv(SAMPLE_HARD, index=False)
        print(f"Muestra difíciles -> {SAMPLE_HARD}")
    else:
        print("No se generó muestra de difíciles.")


if __name__ == "__main__":
    main()
