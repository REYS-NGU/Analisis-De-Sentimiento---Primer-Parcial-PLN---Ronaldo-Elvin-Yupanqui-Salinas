# src/preprocess/label_diet.py
import re, pandas as pd
from src.common.paths import pjoin
from src.common.logging import get_logger

log = get_logger("preprocess.label_diet")

PATS = {
    "keto": [r"\bketo\b", r"\bcetog(e|é)nica\b", r"\blow\s*carb\b", r"\bcetosis\b"],
    "ayuno": [
        r"\bayuno\b",
        r"\bintermitente\b",
        r"\b16/8\b",
        r"\b18/6\b",
        r"\bOMAD\b",
        r"\buna\s*comida\s*al\s*d[ií]a\b",
    ],
    "flexible": [r"\bflexible\b", r"\bIIFYM\b", r"\bcontar\s*macros\b", r"\bmacros\b"],
    "mediterranea": [
        r"\bmediterr(a|á)nea\b",
        r"\balimentaci[oó]n\s*mediterr(a|á)nea\b",
        r"\bpatr[oó]n\s*mediterr[aá]neo\b",
    ],
    "paleo": [r"\bpaleo\b", r"\bpaleol[ií]tica\b", r"\bprimal\b"],
    "vegana": [r"\bvegana\b", r"\bvegano\b", r"\bvegetariana\b", r"\bplant\s*based\b"],
}

PRIORIDAD = [
    "ayuno",
    "keto",
    "flexible",
    "mediterranea",
    "paleo",
    "vegana",
]  # en caso de empate


def match_diets(t: str):
    t = (t or "").lower()
    hits = []
    for d, pats in PATS.items():
        if any(re.search(p, t) for p in pats):
            hits.append(d)
    if not hits:
        return None, []
    # resolver conflicto por prioridad fija
    for d in PRIORIDAD:
        if d in hits:
            return d, hits
    return hits[0], hits


def run():
    df = pd.read_csv(pjoin("data", "interim", "limpio.csv"))
    diet, multi = [], []
    for txt in df["texto_raw"].fillna(""):
        d, hits = match_diets(txt)
        diet.append(d)
        multi.append(",".join(hits) if hits else "")
    df["dieta_heuristica"] = diet
    df["dietas_match"] = multi
    out = pjoin("data", "interim", "labeled.csv")
    df.to_csv(out, index=False)
    log.info("Guardado %s (con dieta_heuristica y dietas_match)", out)


if __name__ == "__main__":
    run()
