# src/collect/youtube_collect.py
import os, csv, time, requests, json
from urllib.parse import urlencode
from dotenv import load_dotenv
from src.common.paths import pjoin
from src.common.logging import get_logger

log = get_logger("collect.youtube")
load_dotenv()

API_KEY = os.getenv("YOUTUBE_API_KEY")
if not API_KEY:
    log.error("Falta YOUTUBE_API_KEY en .env")
    raise SystemExit(1)

# Opcional: filtra por fecha (ISO 8601, ej. "2024-01-01T00:00:00Z")
PUBLISHED_AFTER = os.getenv("YOUTUBE_PUBLISHED_AFTER", "").strip() or None

SESSION = requests.Session()
BASE = "https://www.googleapis.com/youtube/v3"

# ===== Dietas y consultas (6+ con sinónimos) =====
DIET_QUERIES = {
    "keto": [
        "dieta keto español",
        "dieta cetogénica español",
        "low carb español",
    ],
    "ayuno": [
        "ayuno intermitente 16/8 español",
        "ayuno intermitente 18/6 español",
        "OMAD español",  # una comida al día
        "ayuno una comida al día español",
    ],
    "flexible": [
        "dieta flexible español",
        "dieta IIFYM español",
        "contar macros español",
        "if it fits your macros español",
    ],
    "mediterranea": [
        "dieta mediterránea español",
        "alimentación mediterránea español",
    ],
    "paleo": [
        "dieta paleo español",
        "dieta paleolítica español",
    ],
    "vegana": [
        "dieta vegana español",
        "alimentación plant based español",
        "dieta vegetariana español",
    ],
    # Puedes añadir más:
    # "dash": ["dieta DASH español"],
    # "carnivora": ["dieta carnívora español"],
    # "whole30": ["dieta whole30 español"],
}

# Parámetros de cuota/control
MAX_SEARCH_PAGES = int(
    os.getenv("YT_MAX_SEARCH_PAGES", "2")
)  # páginas de búsqueda por query
MAX_COMMENT_PAGES = int(
    os.getenv("YT_MAX_COMMENT_PAGES", "5")
)  # páginas de commentThreads por video
REQUEST_SLEEP = float(os.getenv("YT_REQUEST_SLEEP", "0.5"))  # segundos entre queries
MAX_VIDEOS_PER_QUERY = int(os.getenv("YT_MAX_VIDEOS_PER_QUERY", "12"))  # ← 12 por query
MIN_COMMENTS_PER_VIDEO = int(os.getenv("YT_MIN_COMMENTS_PER_VIDEO", "150"))  # ← ≥150


# --- helpers / excepciones:
class SkipVideo(Exception): ...


class StopAll(Exception): ...


def _parse_api_error(resp):
    """Devuelve (status, reason, message, raw_text) si la respuesta es de error del API."""
    status = getattr(resp, "status_code", None)
    text = ""
    try:
        text = resp.text
    except Exception:
        pass
    reason = message = None
    try:
        data = resp.json()
        err = (data or {}).get("error", {})
        reason = (err.get("errors") or [{}])[0].get("reason")
        message = err.get("message")
    except Exception:
        pass
    return status, reason, message, text


def yt_get(path, params, max_retries=3, sleep_base=1.5):
    """Llama endpoint con reintentos + logging claro de errores + partial responses."""
    params = dict(params)
    params["key"] = API_KEY
    # Partial response para ahorrar cuota
    if path == "search":
        params.setdefault(
            "fields",
            "items(id/videoId,snippet/title,snippet/publishedAt),nextPageToken",
        )
    elif path == "commentThreads":
        params.setdefault(
            "fields",
            "items(id,snippet/topLevelComment/snippet(textDisplay,publishedAt),"
            "replies/comments/snippet(textDisplay,publishedAt)),nextPageToken",
        )
    elif path == "videos":
        params.setdefault("fields", "items(id,statistics/commentCount)")
    url = f"{BASE}/{path}?{urlencode(params)}"

    for i in range(max_retries):
        try:
            r = SESSION.get(url, timeout=25)
            r.raise_for_status()
            try:
                return r.json()
            except json.JSONDecodeError:
                log.warning(
                    "JSON inválido en %s (intento %d/%d)", path, i + 1, max_retries
                )
                time.sleep(sleep_base * (i + 1))
                continue
        except requests.HTTPError as e:
            resp = getattr(e, "response", None)
            if resp is not None:
                status, reason, message, _ = _parse_api_error(resp)
                log.warning(
                    "HTTP %s en %s (razón=%s) — %s (intento %d/%d)",
                    status,
                    path,
                    reason,
                    message,
                    i + 1,
                    max_retries,
                )
                if reason in {"commentsDisabled", "forbidden", "notFound"}:
                    raise SkipVideo(f"{reason}: {message}")
                if reason in {
                    "quotaExceeded",
                    "dailyLimitExceeded",
                    "rateLimitExceeded",
                }:
                    raise StopAll(f"{reason}: {message}")
            else:
                log.warning("HTTP ? en %s (intento %d/%d)", path, i + 1, max_retries)
            time.sleep(sleep_base * (i + 1))
        except requests.RequestException as e:
            log.warning(
                "Error de red en %s: %s (intento %d/%d)", path, e, i + 1, max_retries
            )
            time.sleep(sleep_base * (i + 1))
    raise RuntimeError(f"Fallo al llamar {path} tras {max_retries} intentos")


def list_search_videos(query, max_pages):
    """Devuelve lista de (videoId, title, publishedAt) para la query."""
    vids = []
    params = dict(
        part="snippet",
        q=query,
        type="video",
        maxResults=25,
        relevanceLanguage="es",
        safeSearch="none",
    )
    if PUBLISHED_AFTER:
        params["publishedAfter"] = PUBLISHED_AFTER
    page = 0
    while True:
        page += 1
        data = yt_get("search", params)
        for it in data.get("items", []):
            vid = it["id"]["videoId"]
            sn = it["snippet"]
            vids.append((vid, sn.get("title", ""), sn.get("publishedAt", "")))
        if "nextPageToken" in data and page < max_pages:
            params["pageToken"] = data["nextPageToken"]
        else:
            break
    return vids


def get_comment_counts(video_ids):
    """Devuelve dict {video_id: int(commentCount)}. Si falta, asume 0."""
    counts = {}
    # videos.list permite hasta 50 ids por llamada
    CHUNK = 50
    for i in range(0, len(video_ids), CHUNK):
        ids = video_ids[i : i + CHUNK]
        params = dict(part="statistics", id=",".join(ids))
        data = yt_get("videos", params)
        for it in data.get("items", []):
            vid = it.get("id")
            cc = 0
            try:
                cc = int(it.get("statistics", {}).get("commentCount", 0))
            except Exception:
                cc = 0
            counts[vid] = cc
        # Los que no volvieron en items quedan en 0
        for vid in ids:
            counts.setdefault(vid, 0)
    return counts


def select_videos_for_query(query, diet_seen_videos, limit=12, min_comments=150):
    """Selecciona hasta `limit` videos únicos (por dieta) con ≥ min_comments."""
    candidates = list_search_videos(query, MAX_SEARCH_PAGES)
    # Orden natural de YouTube ya viene por relevancia/recencia según query.
    # Filtra no vistos por la dieta:
    candidates = [
        (vid, title, pub)
        for (vid, title, pub) in candidates
        if vid not in diet_seen_videos
    ]
    if not candidates:
        return []

    counts = get_comment_counts([c[0] for c in candidates])
    filtered = [
        (vid, title, pub, counts.get(vid, 0))
        for (vid, title, pub) in candidates
        if counts.get(vid, 0) >= min_comments
    ]
    # Toma los primeros hasta `limit`
    selected = filtered[:limit]
    log.info(
        "Query '%s': %d candidatos, %d con ≥%d comentarios, seleccionados=%d",
        query,
        len(candidates),
        len(filtered),
        min_comments,
        len(selected),
    )
    return selected


def comments_for_video(video_id, max_pages=5, include_replies=True):
    """Genera comentarios (y replies opcional). Maneja errores 'saltables'."""
    params = dict(
        part="snippet,replies" if include_replies else "snippet",
        videoId=video_id,
        maxResults=100,
        textFormat="plainText",
        order="time",
    )
    page = 0
    while True:
        page += 1
        data = yt_get("commentThreads", params)
        for it in data.get("items", []):
            # Top-level
            top = it.get("snippet", {}).get("topLevelComment", {}).get("snippet", {})
            if top:
                yield dict(
                    id=it.get("id"),
                    texto=top.get("textDisplay", "") or "",
                    fecha=top.get("publishedAt"),
                    is_reply=False,
                )
            # Replies
            if include_replies:
                for rep in it.get("replies", {}).get("comments", []):
                    rsn = rep.get("snippet", {})
                    yield dict(
                        id=rep.get("id"),
                        texto=rsn.get("textDisplay", "") or "",
                        fecha=rsn.get("publishedAt"),
                        is_reply=True,
                    )
        if "nextPageToken" in data and page < max_pages:
            params["pageToken"] = data["nextPageToken"]
        else:
            break


def run():
    out = pjoin("data", "raw", "youtube_comments.csv")
    fieldnames = [
        "id",
        "fuente",
        "url",
        "fecha",
        "texto",
        "rating",
        "video_id",
        "video_title",
        "video_publishedAt",
        "video_commentCount",
        "busqueda",
        "dieta",
        "is_reply",
    ]

    # Deduplicación de comentarios (persistente entre ejecuciones)
    seen_ids = set()
    append = False
    if os.path.exists(out):
        append = True
        with open(out, "r", encoding="utf-8") as fr:
            try:
                for row in csv.DictReader(fr):
                    seen_ids.add(row["id"])
            except Exception:
                pass

    with open(out, "a" if append else "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not append:
            w.writeheader()

        for diet, queries in DIET_QUERIES.items():
            log.info("=== Dieta: %s ===", diet)
            diet_seen_videos = (
                set()
            )  # evita repetir videos entre queries de la misma dieta

            for q in queries:
                selected = select_videos_for_query(
                    q,
                    diet_seen_videos,
                    limit=MAX_VIDEOS_PER_QUERY,
                    min_comments=MIN_COMMENTS_PER_VIDEO,
                )
                if not selected:
                    log.info(
                        "Búsqueda '%s' no alcanzó el mínimo de videos con ≥%d comentarios.",
                        q,
                        MIN_COMMENTS_PER_VIDEO,
                    )
                    time.sleep(REQUEST_SLEEP)
                    continue

                for vid, vtitle, vpub, vcc in selected:
                    if vid in diet_seen_videos:
                        continue
                    diet_seen_videos.add(vid)
                    url = f"https://www.youtube.com/watch?v={vid}"

                    count_written = 0
                    try:
                        for c in comments_for_video(
                            vid, max_pages=MAX_COMMENT_PAGES, include_replies=True
                        ):
                            cid = c["id"]
                            if not cid or cid in seen_ids:
                                continue
                            seen_ids.add(cid)
                            w.writerow(
                                dict(
                                    id=cid,
                                    fuente="youtube",
                                    url=url,
                                    fecha=c.get("fecha"),
                                    texto=c.get("texto", ""),
                                    rating=None,
                                    video_id=vid,
                                    video_title=vtitle,
                                    video_publishedAt=vpub,
                                    video_commentCount=str(vcc),
                                    busqueda=q,
                                    dieta=diet,
                                    is_reply=str(c.get("is_reply", False)),
                                )
                            )
                            count_written += 1
                        log.info(
                            "Video %s (cc=%s) -> %d comentarios guardados",
                            vid,
                            vcc,
                            count_written,
                        )
                    except SkipVideo as sv:
                        log.info("Video %s saltado: %s", vid, sv)
                        continue
                    except StopAll as sa:
                        log.error("Deteniendo: %s", sa)
                        return
                    except RuntimeError as re:
                        log.warning(
                            "Video %s con errores persistentes: %s (saltando)", vid, re
                        )
                        continue

                time.sleep(REQUEST_SLEEP)

    log.info("Guardado %s", out)


if __name__ == "__main__":
    run()
