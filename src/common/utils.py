import re, emoji
NEGADORES = {"no","nunca","jamás","tampoco"}
def basic_clean(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s
def replace_emojis(s: str) -> str:
    return emoji.replace_emoji(s, replace=lambda e,d: f" :{d['en'].split(':')[0]}: ")
def marcar_negacion_spacy(doc, ventana: int = 3) -> str:
    marcado, neg, v = [], False, 0
    for tok in doc:
        if tok.text.lower() in NEGADORES:
            neg, v = True, ventana
            continue
        w = tok.lemma_.lower()
        if neg and v>0:
            w = f"no_{w}"; v -= 1
        marcado.append(w)
        if v==0: neg=False
    return " ".join(marcado)
