# src/collect/rss_collect.py
from bs4 import BeautifulSoup
import feedparser, requests, pandas as pd
from datetime import datetime
from src.common.paths import pjoin
from src.common.logging import get_logger

log = get_logger("collect.rss")

# 3 feeds por dieta (18 total), en español (LatAm) y país BO
# Notas:
# - Google News RSS acepta consultas con when:365d y locale (hl, gl, ceid). Ver doc extraoficial.
# - Cada consulta es específica por dieta para maximizar relevancia.
FEEDS = [
    # AYUNO
    {
        "url": "https://news.google.com/rss/search?q=ayuno%20intermitente%20when:365d&hl=es-419&gl=BO&ceid=BO:es-419",
        "dieta": "ayuno",
    },
    {
        "url": "https://news.google.com/rss/search?q=%2216%2F8%22%20OR%20%22ayuno%2016%2F8%22%20when:365d&hl=es-419&gl=BO&ceid=BO:es-419",
        "dieta": "ayuno",
    },
    {
        "url": "https://news.google.com/rss/search?q=OMAD%20OR%20%22una%20comida%20al%20d%C3%ADa%22%20ayuno%20when:365d&hl=es-419&gl=BO&ceid=BO:es-419",
        "dieta": "ayuno",
    },
    # KETO / CETOGÉNICA
    {
        "url": "https://news.google.com/rss/search?q=dieta%20cetog%C3%A9nica%20when:365d&hl=es-419&gl=BO&ceid=BO:es-419",
        "dieta": "keto",
    },
    {
        "url": "https://news.google.com/rss/search?q=keto%20OR%20%22baja%20en%20carbohidratos%22%20when:365d&hl=es-419&gl=BO&ceid=BO:es-419",
        "dieta": "keto",
    },
    {
        "url": "https://news.google.com/rss/search?q=cetosis%20dieta%20when:365d&hl=es-419&gl=BO&ceid=BO:es-419",
        "dieta": "keto",
    },
    # FLEXIBLE / IIFYM
    {
        "url": "https://news.google.com/rss/search?q=%22dieta%20flexible%22%20OR%20IIFYM%20when:365d&hl=es-419&gl=BO&ceid=BO:es-419",
        "dieta": "flexible",
    },
    {
        "url": "https://news.google.com/rss/search?q=%22contar%20macros%22%20dieta%20when:365d&hl=es-419&gl=BO&ceid=BO:es-419",
        "dieta": "flexible",
    },
    {
        "url": "https://news.google.com/rss/search?q=macros%20IIFYM%20when:365d&hl=es-419&gl=BO&ceid=BO:es-419",
        "dieta": "flexible",
    },
    # MEDITERRÁNEA
    {
        "url": "https://news.google.com/rss/search?q=%22dieta%20mediterr%C3%A1nea%22%20when:365d&hl=es-419&gl=BO&ceid=BO:es-419",
        "dieta": "mediterranea",
    },
    {
        "url": "https://news.google.com/rss/search?q=%22alimentaci%C3%B3n%20mediterr%C3%A1nea%22%20when:365d&hl=es-419&gl=BO&ceid=BO:es-419",
        "dieta": "mediterranea",
    },
    {
        "url": "https://news.google.com/rss/search?q=%22patr%C3%B3n%20mediterr%C3%A1neo%22%20dieta%20when:365d&hl=es-419&gl=BO&ceid=BO:es-419",
        "dieta": "mediterranea",
    },
    # PALEO
    {
        "url": "https://news.google.com/rss/search?q=%22dieta%20paleo%22%20when:365d&hl=es-419&gl=BO&ceid=BO:es-419",
        "dieta": "paleo",
    },
    {
        "url": "https://news.google.com/rss/search?q=%22dieta%20paleol%C3%ADtica%22%20when:365d&hl=es-419&gl=BO&ceid=BO:es-419",
        "dieta": "paleo",
    },
    {
        "url": "https://news.google.com/rss/search?q=%22dieta%20primal%22%20when:365d&hl=es-419&gl=BO&ceid=BO:es-419",
        "dieta": "paleo",
    },
    # VEGANA
    {
        "url": "https://news.google.com/rss/search?q=%22dieta%20vegana%22%20when:365d&hl=es-419&gl=BO&ceid=BO:es-419",
        "dieta": "vegana",
    },
    {
        "url": "https://news.google.com/rss/search?q=%22alimentaci%C3%B3n%20vegana%22%20when:365d&hl=es-419&gl=BO&ceid=BO:es-419",
        "dieta": "vegana",
    },
    {
        "url": "https://news.google.com/rss/search?q=%22plant%20based%22%20espa%C3%B1ol%20when:365d&hl=es-419&gl=BO&ceid=BO:es-419",
        "dieta": "vegana",
    },
]

TIMEOUT = 15
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
}


def fetch_full_text(url: str) -> str:
    try:
        html = requests.get(
            url, timeout=TIMEOUT, headers=HEADERS, allow_redirects=True
        ).text
        soup = BeautifulSoup(html, "html.parser")
        # Texto básico: concatenar párrafos
        body = " ".join(p.get_text(" ", strip=True) for p in soup.find_all("p"))
        return body
    except Exception:
        return ""


def run():
    rows = []
    for feed in FEEDS:
        dieta = feed["dieta"] if isinstance(feed, dict) else None
        u = feed["url"] if isinstance(feed, dict) else feed
        try:
            f = feedparser.parse(u)
            for e in f.entries:
                url = e.link
                # 1) summary del feed
                text = ""
                summary = e.get("summary", "")
                if summary:
                    soup = BeautifulSoup(summary, "html.parser")
                    text = soup.get_text(" ", strip=True)

                # 2) intentar cuerpo completo
                full_body = fetch_full_text(url)
                if len(full_body.split()) > len(text.split()):
                    text = full_body

                if not text:
                    continue

                rows.append(
                    {
                        "id": e.get("id", url),
                        "fuente": "blog",
                        "url": url,
                        "fecha": e.get("published", datetime.utcnow().isoformat()),
                        "texto": text,
                        "rating": None,
                        "dieta": dieta or "",
                    }
                )
        except Exception as ex:
            log.warning("Error en feed %s: %s", u, ex)

    if not rows:
        log.info("Sin filas; revisa FEEDS.")
        return

    df = pd.DataFrame(rows).drop_duplicates(subset=["texto", "url"])
    out = pjoin("data", "raw", "blogs.csv")
    df.to_csv(out, index=False)
    log.info("Guardado %s (%d filas)", out, len(df))


if __name__ == "__main__":
    run()
