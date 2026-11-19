"""Load scraped data, build a TF-IDF index, and answer structured queries."""
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

PROPERTY_WORDS = {
    "apartment": ["apartment", "apartments", "flat", "flats"],
    "villa": ["villa", "villas"],
    "land": ["land", "lands", "plot", "plots"],
    "floor": ["floor", "floors"],
    "building": ["building", "buildings"],
    "rest house": ["rest house", "rest houses", "istiraha"],
    "chalet": ["chalet", "chalets"],
    "studio": ["studio", "studios"],
    "duplex": ["duplex"],
    "showroom": ["showroom", "showrooms"],
    "office": ["office", "offices"],
    "farm": ["farm", "farms"],
}


def _load(name):
    p = DATA_DIR / name
    return json.loads(p.read_text()) if p.exists() else []


class Knowledge:
    def __init__(self):
        self.darglobal = _load("darglobal.json")
        self.listings = _load("wasalt_listings.json")
        # Wasalt URL slugs drop the decimal point ("villa-32003-sqm" is 320.03 sqm).
        # 5+ digit areas are implausible for residential units, so treat them as
        # having two decimal places. Land/buildings can genuinely be that large.
        for r in self.listings:
            raw = r.get("area_sqm_raw")
            if raw is None:
                continue
            if raw >= 10000 and r["type"] not in ("land", "building", "sqm land sale", "sqm land rent"):
                r["area_sqm"] = round(raw / 100, 2)
                r["area_estimated"] = True
            else:
                r["area_sqm"] = raw
        self.categories = _load("wasalt_categories.json")
        self.daily = _load("wasalt_daily_rentals.json")
        self.daily_areas = _load("wasalt_daily_rental_areas.json")
        self.static = _load("wasalt_static.json")
        meta = DATA_DIR / "meta.json"
        self.meta = json.loads(meta.read_text()) if meta.exists() else {}

        self.cities = sorted({c["city"].lower() for c in self.categories + self.daily_areas},
                             key=len, reverse=True)
        self._build_index()
        self.summary = self._summary()

    # ------------------------------------------------------------ index
    def _doc_text(self, r):
        parts = [r.get("source", ""), r.get("title", "")]
        for k in ("section", "country", "country_hint", "property_type", "development",
                  "purpose", "type", "category", "city", "district", "price_filter", "facing"):
            if r.get(k):
                parts.append(str(r[k]))
        if r.get("bedrooms"):
            parts.append(f"{r['bedrooms']} bedrooms")
        return " ".join(parts)

    def _build_index(self):
        # Individual listings are served by the structured filter; the text
        # index covers DarGlobal pages and Wasalt area/category pages.
        self.docs = self.darglobal + self.categories + self.daily_areas + self.static
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        self.mat = self.vec.fit_transform([self._doc_text(d) for d in self.docs])

    def search(self, q, k=12, source=None):
        qv = self.vec.transform([q])
        scores = (self.mat @ qv.T).toarray().ravel()
        if source:
            mask = np.array([d["source"] == source for d in self.docs])
            scores = scores * mask
        idx = np.argsort(-scores)[: k]
        return [self.docs[i] for i in idx if scores[i] > 0.05]

    # ------------------------------------------------------------ summary
    def _summary(self):
        dg_sections = Counter(r["section"] for r in self.darglobal)
        dg_projects = [r["title"] for r in self.darglobal
                       if r["section"] in {"project", "one-of-one", "rayana", "amaya", "trump-plaza"}]
        dg_countries = sorted({r["country"] for r in self.darglobal if r.get("country")})
        dg_partners = [{"name": r["title"], "url": r["url"]} for r in self.darglobal
                       if r["section"] == "partners" and r["path"] != "/partners"]
        lst = Counter(f"{r['purpose']}:{r['type']}" for r in self.listings)
        cities = Counter(r["city"] for r in self.categories)
        daily = Counter(r["type"] for r in self.daily)
        return {
            "scraped_at": self.meta.get("scraped_at"),
            "darglobal": {
                "pages": len(self.darglobal),
                "pages_by_section": dict(dg_sections),
                "countries_with_project_pages": dg_countries,
                "project_and_development_pages": dg_projects,
                "brand_partner_pages": dg_partners,
            },
            "wasalt": {
                "listings_total": len(self.listings),
                "listings_for_sale": sum(1 for r in self.listings if r["purpose"] == "sale"),
                "listings_for_rent": sum(1 for r in self.listings if r["purpose"] == "rent"),
                "listings_by_purpose_and_type": dict(lst.most_common(20)),
                "daily_rental_units": len(self.daily),
                "daily_rental_units_by_type": dict(daily),
                "area_search_pages": len(self.categories),
                "daily_rental_area_pages": len(self.daily_areas),
                "top_cities_by_area_pages": dict(cities.most_common(25)),
            },
        }

    # ------------------------------------------------------------ structured
    def parse_filters(self, q):
        ql = " " + q.lower() + " "
        f = {}
        if re.search(r"\b(daily|per night|nightly|booking|holiday|vacation)\b", ql):
            f["purpose"] = "daily rental"
        elif re.search(r"\b(rent|rental|rentals|lease|renting)\b", ql):
            f["purpose"] = "rent"
        elif re.search(r"\b(sale|buy|buying|purchase|sell)\b", ql):
            f["purpose"] = "sale"
        for t, words in PROPERTY_WORDS.items():
            if any(re.search(rf"\b{re.escape(w)}\b", ql) for w in words):
                f["type"] = t
                break
        if (m := re.search(r"\b(\d+)\s*(?:-|\s)?(?:bed|beds|bedroom|bedrooms|br|bhk)\b", ql)):
            f["bedrooms"] = int(m.group(1))
        if (m := re.search(r"(?:over|above|more than|at least|>=?|minimum|min)\s*(\d+)\s*(?:sqm|m2|square)", ql)):
            f["min_sqm"] = int(m.group(1))
        if (m := re.search(r"(?:under|below|less than|at most|<=?|maximum|max)\s*(\d+)\s*(?:sqm|m2|square)", ql)):
            f["max_sqm"] = int(m.group(1))
        for c in self.cities:
            if re.search(rf"\b{re.escape(c)}\b", ql):
                f["city"] = c
                break
        if "city" not in f and "riyadh" in ql:
            f["city"] = "riyadh"
        if (m := re.search(r"\b(north|south|east|west)\w*[- ]facing|facing (north|south|east|west)", ql)):
            f["facing"] = m.group(1) or m.group(2)
        return f

    def listing_matches(self, f):
        if f.get("purpose") == "daily rental":
            rows = self.daily
            if f.get("type"):
                rows = [r for r in rows if f["type"] in r["type"]]
            return rows
        rows = self.listings
        if f.get("purpose"):
            rows = [r for r in rows if r["purpose"] == f["purpose"]]
        if f.get("type"):
            rows = [r for r in rows if f["type"] in r["type"]]
        if f.get("bedrooms"):
            rows = [r for r in rows if r.get("bedrooms") == f["bedrooms"]]
        if f.get("min_sqm"):
            rows = [r for r in rows if r.get("area_sqm", 0) >= f["min_sqm"]]
        if f.get("max_sqm"):
            rows = [r for r in rows if 0 < r.get("area_sqm", 0) <= f["max_sqm"]]
        if f.get("facing"):
            rows = [r for r in rows if f["facing"] in r.get("facing", "")]
        return rows

    def area_matches(self, f):
        pool = self.daily_areas if f.get("purpose") == "daily rental" else self.categories
        rows = pool
        if f.get("city"):
            rows = [r for r in rows if f["city"] in r["city"].lower()
                    or f["city"] in (r.get("district") or "").lower()]
        if f.get("purpose") and f["purpose"] != "daily rental":
            rows = [r for r in rows if r["purpose"] == f["purpose"]]
        if f.get("type"):
            rows = [r for r in rows if f["type"] in r["category"]]
        if f.get("bedrooms"):
            rows = [r for r in rows if r.get("bedrooms") == f["bedrooms"]]
        return rows

    # ------------------------------------------------------------ context
    def build_context(self, question, history_text=""):
        q = question if not history_text else history_text + " " + question
        f = self.parse_filters(question) or self.parse_filters(q)
        ctx = {"dataset_summary": self.summary, "detected_filters": f}
        sources = []

        wants_dg = bool(re.search(r"dar\s*global|darglobal|trump|aida|pagani|missoni|lamborghini|"
                                  r"aston martin|fendi|elie saab|w residences|astera|mulliner|"
                                  r"one of one|marbella|london|dubai|oman|qatar|spain", q.lower()))

        if f and not wants_dg:
            matches = self.listing_matches(f)
            listing_f = {k: v for k, v in f.items() if k != "city"}
            ctx["wasalt_listing_query"] = {
                "filters_applied": listing_f,
                "note": ("This count is NATIONWIDE across Saudi Arabia: listing URLs have no city, so the "
                         "city filter could not be applied. Never present it as a count for the city."
                         if f.get("city") else None),
                "matching_listings_nationwide": len(matches),
                "by_type": dict(Counter(f"{r['purpose']}:{r['type']}" for r in matches).most_common(8)),
                "bedroom_breakdown": dict(Counter(r.get("bedrooms") for r in matches
                                                  if r.get("bedrooms")).most_common(8)),
                "examples": [{k: r[k] for k in ("title", "purpose", "type", "bedrooms",
                                                "area_sqm", "area_estimated", "facing", "street_width_m",
                                                "url", "lastmod") if r.get(k)}
                             for r in sorted(matches, key=lambda r: -r["id"])[:8]],
            }
            areas = self.area_matches(f)
            if areas:
                ctx["wasalt_area_pages"] = {
                    "matching_pages": len(areas),
                    "by_city": dict(Counter(r["city"] for r in areas).most_common(12)),
                    "districts": sorted({r["district"] for r in areas if r.get("district")})[:40],
                    "examples": [{k: r[k] for k in ("title", "city", "district", "category",
                                                    "purpose", "bedrooms", "price_filter", "url")
                                  if r.get(k)} for r in areas[:10]],
                }
            sources += [e["url"] for e in ctx["wasalt_listing_query"]["examples"][:4]]
            if areas:
                sources += [a["url"] for a in areas[:3]]

        hits = self.search(q, k=14, source="DarGlobal" if wants_dg else None)
        ctx["related_pages"] = [{k: h[k] for k in ("source", "title", "section", "country",
                                                   "country_hint", "property_type", "city",
                                                   "district", "category", "purpose", "url")
                                 if h.get(k)} for h in hits]
        sources += [h["url"] for h in hits[:6]]
        seen, uniq = set(), []
        for s in sources:
            if s not in seen:
                seen.add(s)
                uniq.append(s)
        return ctx, uniq[:8]
