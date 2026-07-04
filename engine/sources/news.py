"""News-Schnipsel für die LLM-Analyse (concept.md Schicht 3, Teil 1).

RSS statt HTML-Scraping: stabiles XML, kein Layout-Risiko wie bei
Kicktipp-Scraping. Liefert die jüngsten Schlagzeilen, die einen der beiden
Teamnamen erwähnen – unstrukturiert als roher Text ans LLM, das selbst
beurteilt, ob etwas relevant ist (Verletzung, Sperre, Umbruch). Kein Anspruch
auf Vollständigkeit oder Team-Alias-Erkennung; findet sich nichts, bleibt die
LLM-Anpassung aus (siehe engine/llm.py) – besser keine Anpassung als eine auf
Basis von nichts.
"""

from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from xml.etree import ElementTree

import requests

from ..config import CACHE_DIR

FEEDS = {
    "kicker": "https://newsfeed.kicker.de/news/aktuell",
    "sportschau": "https://www.sportschau.de/fussball/index~rss2.xml",
}


def _fetch_feed(name: str, url: str, cache_dir: Path, cache_tag: str) -> list[dict]:
    cache_file = cache_dir / f"news_{name}_{cache_tag}.xml"
    if cache_file.exists():
        raw = cache_file.read_bytes()
    else:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        raw = resp.content
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_bytes(raw)

    # rohe Bytes statt resp.text: ElementTree liest die Kodierung aus der
    # XML-Deklaration selbst - manche Feeds (z.B. kicker.de) senden
    # "Content-Type: text/xml" ohne charset, worauf requests fälschlich
    # ISO-8859-1 statt des tatsächlichen UTF-8 rät (Mojibake bei Umlauten)
    root = ElementTree.fromstring(raw)
    items = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        description = (item.findtext("description") or "").strip()
        pub_date_raw = item.findtext("pubDate")
        try:
            pub_date = parsedate_to_datetime(pub_date_raw) if pub_date_raw else None
        except (TypeError, ValueError):
            pub_date = None
        if title:
            items.append(
                {"title": title, "description": description, "published": pub_date, "source": name}
            )
    return items


def fetch_snippets(
    home_name: str,
    away_name: str,
    max_age_days: int = 5,
    max_items: int = 5,
    cache_dir: Path = CACHE_DIR,
    cache_tag: str | None = None,
    now: datetime | None = None,
) -> list[dict]:
    """Bis zu `max_items` jüngste Schlagzeilen, die eines der beiden Teams
    erwähnen. Best-effort: [] statt Fehler, wenn ein Feed nicht erreichbar ist."""
    now = now or datetime.now(timezone.utc)
    cache_tag = cache_tag or now.date().isoformat()
    cutoff = now - timedelta(days=max_age_days)

    all_items = []
    for name, url in FEEDS.items():
        try:
            all_items += _fetch_feed(name, url, cache_dir, cache_tag)
        except (requests.RequestException, ElementTree.ParseError) as exc:
            print(f"News-Feed {name} nicht verfügbar: {exc}")

    names = (home_name, away_name)
    relevant = [
        item
        for item in all_items
        if any(n in f"{item['title']} {item['description']}" for n in names)
        and (item["published"] is None or item["published"] >= cutoff)
    ]
    relevant.sort(key=lambda i: i["published"] or cutoff, reverse=True)
    return relevant[:max_items]
