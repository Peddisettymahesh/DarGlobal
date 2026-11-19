"""Collect publicly available data from DarGlobal and Wasalt.

Both sites put their page bodies behind bot-protection (Incapsula / Cloudflare),
so this scraper only uses what they publish openly for crawlers: their XML
sitemaps (listed in each site's robots.txt). URL slugs on both sites are
descriptive, so they are parsed into structured records.

Usage:  python scraper/scrape.py [--out data]
"""
import argparse
import gzip
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import requests

UA = "Mozilla/5.0 (compatible; RealEstateChatbotResearch/1.0)"
NS = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}

DARGLOBAL_SITEMAP = "https://darglobal.co.uk/sitemap.xml"
WASALT_SITEMAPS = {
    "listings": "https://cdn.wasalt.sa/sitemap/product_sitemap_en_sa.xml.gz",
    "categories": "https://cdn.wasalt.sa/sitemap/category_sitemap_en_sa.xml.gz",
    "daily_rentals": "https://cdn.wasalt.sa/sitemap/rental_pdp_sitemap_en_sa.xml.gz",
    "daily_rental_areas": "https://cdn.wasalt.sa/sitemap/rental_srp_sitemap_en_sa.xml.gz",
    "static": "https://cdn.wasalt.sa/sitemap/static_sitemap_en_sa.xml.gz",
}


def fetch(url: str) -> bytes:
    for attempt in range(3):
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=60)
            r.raise_for_status()
            body = r.content
            if body[:2] == b"\x1f\x8b":
                body = gzip.decompress(body)
            return body
        except Exception as e:  # noqa: BLE001
            print(f"  retry {attempt + 1} for {url}: {e}", file=sys.stderr)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}")


def sitemap_entries(url: str):
    root = ET.fromstring(fetch(url))
    for u in root.findall("s:url", NS):
        loc = (u.findtext("s:loc", default="", namespaces=NS) or "").strip()
        lastmod = (u.findtext("s:lastmod", default="", namespaces=NS) or "").strip()
        if loc:
            yield loc, lastmod[:10]


def humanize(slug: str) -> str:
    s = re.sub(r"[-_]+", " ", slug).strip()
    return s[:1].upper() + s[1:]


# ---------------------------------------------------------------- DarGlobal
DG_COUNTRIES = {
    "uae": "UAE", "uk": "United Kingdom", "spain": "Spain", "oman": "Oman",
    "qatar": "Qatar", "saudi-arabia": "Saudi Arabia",
}
# Only hints that are evidenced by the site's own URLs (e.g. ".../aida-oman",
# ".../rayana/trump-international-qatar", "...-dubai", "...-jeddah").
DG_KNOWN_LOCATION_HINTS = {
    "aida": "Oman", "oman": "Oman", "jeddah": "Saudi Arabia", "dubai": "UAE",
    "rayana": "Qatar", "qatar": "Qatar",
}


def dg_section(path_parts):
    if not path_parts:
        return "home"
    first = path_parts[0].lower()
    if first in {"blog", "press", "insights", "partners", "projects", "one-of-one",
                 "rayana", "amaya", "trump-plaza"}:
        return first
    if first in {"privacy-policy", "terms-conditions", "terms-of-uses", "thank-you",
                 "success", "pay-online", "landing-page", "landingpage",
                 "new-home-page", "broker-contest-terms-conditions"}:
        return "utility"
    if first in {"about", "careers", "internships", "internship-application",
                 "faq", "get-in-touch", "become-a-broker", "become-an-agent",
                 "investor", "why-invest", "tokenization", "invest-crypto",
                 "dg-circle", "exclusive-membership", "insignia",
                 "darglobal-insignia-card", "development-management",
                 "hospitality", "commercial", "campaigns", "bookyourunit"}:
        return "company"
    return "project"


def scrape_darglobal():
    records, seen = [], set()
    for loc, lastmod in sitemap_entries(DARGLOBAL_SITEMAP):
        if loc in seen:
            continue
        seen.add(loc)
        path = re.sub(r"^https?://[^/]+", "", loc).strip("/")
        parts = [p for p in path.split("/") if p]
        section = dg_section(parts)
        slug = parts[-1] if parts else "home"
        rec = {
            "source": "DarGlobal", "url": loc, "lastmod": lastmod,
            "section": section, "title": humanize(slug) if parts else "DarGlobal home page",
            "path": "/" + path,
        }
        if section == "projects" and len(parts) >= 2:
            m = re.match(r"properties-in-(.+)", parts[1])
            if m:
                rec["country"] = DG_COUNTRIES.get(m.group(1), humanize(m.group(1)))
                rec["title"] = f"DarGlobal properties in {rec['country']}"
            if len(parts) >= 3:
                rec["property_type"] = humanize(parts[2])
                rec["title"] += f" - {rec['property_type']}"
        elif section in {"rayana", "amaya", "trump-plaza"}:
            rec["development"] = humanize(parts[0])
            rec["title"] = f"{humanize(parts[0])}: {humanize(slug)}"
        if "country" not in rec and section in {"project", "one-of-one", "rayana", "amaya", "trump-plaza"}:
            low = path.lower()
            for hint, country in DG_KNOWN_LOCATION_HINTS.items():
                if hint in low:
                    rec["country_hint"] = country
                    break
        records.append(rec)
    return records


# ---------------------------------------------------------------- Wasalt
LISTING_RE = re.compile(r"/property/(sale|rent)/(.+)-(\d+)$")
TYPE_RE = re.compile(r"^([a-z]+(?:-[a-z]+)*?)(?=-\d|-with-|-facing-|$)")


def parse_listing(loc, lastmod):
    m = LISTING_RE.search(loc)
    if not m:
        return None
    purpose, slug, pid = m.groups()
    t = TYPE_RE.match(slug)
    rec = {
        "source": "Wasalt", "url": loc, "lastmod": lastmod, "id": int(pid),
        "purpose": purpose, "type": (t.group(1).replace("-", " ") if t else "property"),
        "title": humanize(slug),
    }
    if (a := re.search(r"-(\d+)-sqm", "-" + slug)):
        rec["area_sqm_raw"] = int(a.group(1))
    if (b := re.search(r"with-(\d+)-bedroom", slug)):
        rec["bedrooms"] = int(b.group(1))
    if (f := re.search(r"facing-([a-z]+(?:-[a-z]+)?)(?:-on|$)", slug)):
        rec["facing"] = f.group(1).replace("-", " ")
    if (w := re.search(r"on-(\d+)m-width-street", slug)):
        rec["street_width_m"] = int(w.group(1))
    return rec


CAT_RE = re.compile(r"/en/(.+?)-for-(sale|rent|daily-rental)-in-(.+)$")


def parse_categories(entries, purpose_override=None):
    raw = []
    for loc, lastmod in entries:
        m = CAT_RE.search(loc)
        if not m:
            continue
        what, purpose, where = m.groups()
        price = None
        if (pm := re.search(r"-for-sar-(\w+)$", where)):
            price = pm.group(1).upper()
            where = where[: pm.start()]
        raw.append((loc, lastmod, what, purpose_override or purpose, where, price))

    # Work out which location slugs are cities: a city slug is a prefix of
    # other (city-district) slugs.
    wheres = {r[4] for r in raw}
    cities = {w for w in wheres if any(o.startswith(w + "-") for o in wheres if o != w)}

    out = []
    for loc, lastmod, what, purpose, where, price in raw:
        city = max((c for c in cities if where == c or where.startswith(c + "-")),
                   key=len, default=where)
        district = where[len(city) + 1:] if where != city else None
        beds = None
        if (bm := re.match(r"(\d+)-bedroom-(.+)", what)):
            beds, what = int(bm.group(1)), bm.group(2)
        rec = {
            "source": "Wasalt", "url": loc, "lastmod": lastmod,
            "purpose": purpose.replace("-", " "), "category": what.replace("-", " "),
            "city": humanize(city), "title": humanize(loc.rsplit("/", 1)[-1]),
        }
        if district:
            rec["district"] = humanize(district)
        if beds:
            rec["bedrooms"] = beds
        if price:
            rec["price_filter"] = f"SAR {price}"
        out.append(rec)
    return out


def scrape_wasalt():
    data = {}
    print("  listings ...")
    data["listings"] = [r for r in (parse_listing(l, d) for l, d in
                        sitemap_entries(WASALT_SITEMAPS["listings"])) if r]
    print("  categories ...")
    data["categories"] = parse_categories(sitemap_entries(WASALT_SITEMAPS["categories"]))
    print("  daily rentals ...")
    dr = []
    for loc, lastmod in sitemap_entries(WASALT_SITEMAPS["daily_rentals"]):
        m = re.search(r"/dailyrental/(.+)-(\d+)$", loc)
        if m:
            t = m.group(1).replace("-for-booking", "").replace("-", " ")
            dr.append({"source": "Wasalt", "url": loc, "lastmod": lastmod, "id": int(m.group(2)),
                       "purpose": "daily rental", "type": "unspecified" if t == "null" else t,
                       "title": humanize(m.group(1))})
    data["daily_rentals"] = dr
    print("  daily rental areas ...")
    data["daily_rental_areas"] = parse_categories(sitemap_entries(WASALT_SITEMAPS["daily_rental_areas"]))
    data["static"] = [{"source": "Wasalt", "url": l, "lastmod": d,
                       "title": humanize(l.rstrip("/").rsplit("/", 1)[-1])}
                      for l, d in sitemap_entries(WASALT_SITEMAPS["static"])]
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "data"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("DarGlobal sitemap ...")
    dg = scrape_darglobal()
    (out / "darglobal.json").write_text(json.dumps(dg, ensure_ascii=False, indent=0))
    print(f"  {len(dg)} pages", dict(Counter(r["section"] for r in dg)))

    print("Wasalt sitemaps ...")
    ws = scrape_wasalt()
    for k, v in ws.items():
        (out / f"wasalt_{k}.json").write_text(json.dumps(v, ensure_ascii=False, separators=(",", ":")))
        print(f"  {k}: {len(v)}")

    (out / "meta.json").write_text(json.dumps({
        "scraped_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sources": {"darglobal": DARGLOBAL_SITEMAP, **WASALT_SITEMAPS},
    }, indent=1))


if __name__ == "__main__":
    main()
