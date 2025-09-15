# src/collect/google_play_collect.py
import csv
from datetime import datetime
from google_play_scraper import reviews_all, Sort  # pip install google-play-scraper
from src.common.paths import pjoin
from src.common.logging import get_logger

log = get_logger("collect.googleplay")

# Mapea apps por dieta (RELLENA con apps reales)
# Mapea apps por dieta (3 por cada una, según tu lista)
APPS_BY_DIET = {
    "ayuno": [
        # BodyFast – Intermittent Fasting
        "com.weightloss.zero.intermittent.bodyfast.fastingtracker",
        # Kompanion – Fasting
        "com.kompanion.fasting.android",
        # EasyFast
        "com.easyfastapp.app",
    ],
    "keto": [
        # Keto Diet Tracker
        "keto.droid.lappir.com.ketodiettracker",
        # Keto Diet – Low Carb Recipes
        "com.aquila.ketodiet.ketogenicdiet.lowcarbrecipes",
        # Carb Manager
        "com.wombatapps.carbmanager",
    ],
    "flexible": [
        # FatSecret
        "com.fatsecret.android",
        # MyFitnessPal
        "com.myfitnesspal.android",
        # Fitia (tal como viene en tu enlace)
        "com.nutrition.technologies.Fitia",
    ],
    "mediterranea": [
        # Dieta Mediterránea (AppCode)
        "net.appcode.dietamediterranea",
        # Mediterranean Diet (Eduven)
        "com.eduven.cc.mediterranean",
        # My Mediterranean Diet
        "com.forfit.mymediterranean",
    ],
    "paleo": [
        # Dieta Paleo (AppCode)
        "net.appcode.dietapaleo",
        # Paleo Diet App
        "paleo.diet.app",
        # Paleo Robbie
        "com.food.paleorobbie",
    ],
    "vegana": [
        # Vegan Recipes (HCCE&G)
        "com.hcceg.veg.compassionfree",
        # Nutrición para Veganos
        "com.asara.nutricionparaveganos",
        # Go Vegan
        "com.appsniver.govegan",
    ],
}


# Países y lenguas: varias variantes de español
COUNTRIES = ["es", "mx", "ar", "co", "cl", "pe"]  # puedes ampliar
LANGS = ["es"]  # español

MIN_WORDS = 5  # descarta reseñas muy cortas
PUBLISHED_AFTER = None  # por ej. "2024-01-01" (YYYY-MM-DD) o None
SORT_ORDER = Sort.NEWEST  # o Sort.MOST_RELEVANT


def ok_text(t: str) -> bool:
    if not t:
        return False
    return len(str(t).split()) >= MIN_WORDS


def ok_date(dt) -> bool:
    if not PUBLISHED_AFTER:
        return True
    try:
        cutoff = datetime.fromisoformat(PUBLISHED_AFTER)
        return dt is not None and dt >= cutoff
    except Exception:
        return True


def run():
    out = pjoin("data", "raw", "google_play_reviews.csv")
    fieldnames = [
        "id",
        "fuente",
        "url",
        "fecha",
        "texto",
        "rating",
        "package",
        "dieta",
        "pais",
        "lang",
    ]
    seen = set()
    count_total = 0

    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for diet, pkgs in APPS_BY_DIET.items():
            for pkg in pkgs:
                for country in COUNTRIES:
                    for lang in LANGS:
                        log.info(
                            "Descargando %s | %s [%s-%s] ...", diet, pkg, lang, country
                        )
                        try:
                            rows = reviews_all(
                                pkg,
                                sleep_milliseconds=0,
                                lang=lang,
                                country=country,
                                sort=SORT_ORDER,
                            )
                        except Exception as e:
                            log.warning("Fallo en %s %s-%s: %s", pkg, lang, country, e)
                            continue

                        base_url = (
                            f"https://play.google.com/store/apps/details?id={pkg}"
                        )
                        for r in rows:
                            rid = r.get("reviewId")
                            if not rid or rid in seen:
                                continue
                            seen.add(rid)

                            text = r.get("content", "")
                            dt = r.get("at")  # datetime
                            if not ok_text(text) or not ok_date(dt):
                                continue

                            w.writerow(
                                {
                                    "id": rid,
                                    "fuente": "googleplay",
                                    "url": base_url,
                                    "fecha": dt.isoformat() if dt else "",
                                    "texto": text,
                                    "rating": r.get("score"),
                                    "package": pkg,
                                    "dieta": diet,
                                    "pais": country,
                                    "lang": lang,
                                }
                            )
                            count_total += 1
                        log.info("OK %s | %s [%s-%s]", diet, pkg, lang, country)

    log.info("Guardado %s (%d reseñas)", out, count_total)


if __name__ == "__main__":
    run()
