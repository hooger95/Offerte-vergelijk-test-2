import re
import math
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict

import fitz  # pymupdf
import pandas as pd
from pathlib import Path

# ===================== Hulpfuncties: parsing & normalisatie =====================

CURRENCY_SIGNS = ["€", "eur", "euro", "eur.", "€,"]
UNITS = {
    "st", "stuk", "stuks", "m", "m1", "m2", "m³", "m3", "mm", "cm", "km",
    "uur", "u", "kg", "g", "ton", "l", "liter", "set", "pkg", "doos"
}
LABOR_KEYWORDS = {"arbeid", "uren", "uur", "montage", "installatie", "uurtarief", "werkzaamheden"}
MATERIAL_KEYWORDS = {"materiaal", "materialen", "leveren", "artikel", "artikelnr", "product", "onderdeel"}
SUMMARY_KEYWORDS = {
    "subtotaal": "subtotal",
    "sub totaal": "subtotal",
    "btw": "vat",
    "totaal": "total",
    "korting": "discount",
    "st elpost": "allowance",
    "stelpost": "allowance",
    "meerwerk": "option",
    "optie": "option",
}
DECIMAL_SEPS = [",", "."]

def has_text_in_pdf(path: str) -> bool:
    with fitz.open(path) as doc:
        for page in doc:
            if page.get_text("text").strip():
                return True
    return False

def extract_lines(path: str) -> List[str]:
    """Extraheer tekstlijnen per pagina in leesvolgorde."""
    lines = []
    with fitz.open(path) as doc:
        for page in doc:
            # blocks houdt groepering beter dan raw text
            for b in page.get_text("blocks"):
                if len(b) >= 5:
                    text = b[4]
                    for ln in text.splitlines():
                        ln = ln.strip()
                        if ln:
                            lines.append(ln)
    return lines

def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())

def normalize_desc(s: str) -> str:
    s = s.lower()
    s = normalize_whitespace(s)
    # verwijder veelvoorkomende artikelprefixen/codes (heuristiek)
    s = re.sub(r"\b(art(ikel)?nr\.?|sku|code)\s*[:#-]?\s*\w+\b", "", s)
    # verwijder losse # en dubbele spaties
    s = s.replace("–", "-").replace("—", "-").replace("|", " ")
    s = normalize_whitespace(s)
    return s

def detect_category(desc_norm: str) -> str:
    tokens = set(desc_norm.split())
    if tokens & LABOR_KEYWORDS:
        return "Werkzaamheden"
    if tokens & MATERIAL_KEYWORDS:
        return "Materialen"
    return "Onbekend"

def clean_money(s: str) -> str:
    s = s.lower()
    for c in CURRENCY_SIGNS:
        s = s.replace(c.lower(), "")
    s = s.replace(" ", "")
    return s

def parse_number_any(s: str) -> Optional[float]:
    """
    Parseert zowel EU (1.234,56) als US (1,234.56) notatie.
    Werkt voor decimals en gehele getallen. Geeft None bij mislukken.
    """
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    s0 = clean_money(s)
    # Houd alleen cijfers, . en ,
    s1 = re.sub(r"[^0-9,.\-]", "", s0)
    if not s1 or s1 in {",", ".", "-"}:
        return None

    # Bepaal laatste scheidingsteken als decimal
    last_comma = s1.rfind(",")
    last_dot = s1.rfind(".")
    decimal_sep = None
    if last_comma == -1 and last_dot == -1:
        decimal_sep = None
    elif last_comma > last_dot:
        decimal_sep = ","
    else:
        decimal_sep = "."

    if decimal_sep:
        int_part = s1[:s1.rfind(decimal_sep)]
        frac_part = s1[s1.rfind(decimal_sep)+1:]
        int_digits = re.sub(r"[^0-9\-]", "", int_part)
        s_norm = f"{int_digits}.{re.sub(r'[^0-9]', '', frac_part)}"
    else:
        s_norm = re.sub(r"[^0-9\-]", "", s1)

    try:
        return float(s_norm)
    except Exception:
        return None

def find_money_tokens(line: str) -> List[Tuple[str, float]]:
    """
    Vind alle geldbedragen in een regel, retourneer [(raw, value_float)].
    """
    # Zoek tokens met € of typische geldpatronen
    candidates = re.findall(r"(?:€\s*)?[-]?\d{1,3}(?:[.\s]\d{3})*(?:[.,]\d{2})|(?:€\s*)?[-]?\d+[.,]\d{2}", line)
    out = []
    for raw in candidates:
        val = parse_number_any(raw)
        if val is not None:
            out.append((raw, val))
    # Unieke op value en raw
    seen = set()
    unique = []
    for raw, val in out:
        key = (raw, round(val, 4))
        if key not in seen:
            seen.add(key)
            unique.append((raw, val))
    return unique

def find_qty_and_unit(tokens: List[str]) -> Tuple[Optional[float], Optional[str]]:
    """
    Zoek (aantal, eenheid) heuristisch in tokens.
    Voorbeeld: ['Leveren', 'kabel', '100', 'm', '€1,20', '€120,00'] -> (100, 'm')
    """
    # Eerst: expliciete labels
    for i, t in enumerate(tokens):
        tl = t.lower().strip(",.")
        if tl in {"aantal", "qty"} and i+1 < len(tokens):
            q = parse_number_any(tokens[i+1])
            if q is not None:
                # unit mogelijk daarna
                unit = None
                if i+2 < len(tokens) and tokens[i+2].lower() in UNITS:
                    unit = tokens[i+2].lower()
                return q, unit

    # Anders: zoek patroon [getal] [unit]
    for i in range(len(tokens)-1):
        q = parse_number_any(tokens[i])
        u = tokens[i+1].lower().strip(",.")
        if q is not None and (u in UNITS or re.fullmatch(r"[a-z]{1,4}\.?$", u)):
            return q, u if u in UNITS else None

    # Als laatste: een los getal dat plausibel qty is (en geen prijs, dus staat eerder dan geldbedragen)
    money_idx = [i for i, t in enumerate(tokens) if find_money_tokens(t)]
    cutoff = min(money_idx) if money_idx else len(tokens)
    for i in range(min(cutoff, len(tokens))):
        q = parse_number_any(tokens[i])
        if q is not None and q > 0 and not float(q).is_integer() is False or q >= 1:
            return q, None

    return None, None

@dataclass
class Item:
    description_raw: str
    description_norm: str
    category: str
    qty: Optional[float]
    unit: Optional[str]
    unit_price: Optional[float]
    total_price: Optional[float]
    source: str

@dataclass
class SummaryLine:
    kind: str  # subtotal / vat / total / discount / allowance / option / other
    label: str
    amount: float
    source: str

def classify_line(line: str) -> str:
    ln = line.lower()
    ln = normalize_whitespace(ln)
    for k, v in SUMMARY_KEYWORDS.items():
        if k in ln:
            return v
    return "line"

def parse_item_from_line(line: str, source: str) -> Optional[Item]:
    tokens = normalize_whitespace(line).split()
    money = find_money_tokens(line)

    # Als geen geldbedrag in regel, is het zelden een itemregel
    if not money:
        return None

    # Vaak staat laatste geldbedrag = regel-totaal, het voorlaatste = stuksprijs
    unit_price = None
    total_price = None
    if len(money) >= 2:
        unit_price = money[-2][1]
        total_price = money[-1][1]
    else:
        # 1 bedrag: kan unit of total zijn. Als qty gevonden en realistisch -> interpreteer bedrag als unit en bereken total
        qty_guess, unit_guess = find_qty_and_unit(tokens)
        if qty_guess is not None and qty_guess > 0:
            unit_price = money[-1][1]
            total_price = round(unit_price * qty_guess, 2)
        else:
            total_price = money[-1][1]

    qty, unit = find_qty_and_unit(tokens)

    # Omschrijving = lijn minus prijsdelen aan het einde
    # Strip geld-tokens rechts
    desc_part = line
    for raw, _ in money[::-1]:
        # verwijder alleen laatste occurrences
        idx = desc_part.rfind(raw)
        if idx != -1:
            desc_part = desc_part[:idx]
    desc = normalize_whitespace(desc_part)

    # Voorkom dat de beschrijving leeg is
    if len(desc) < 2:
        # alternatief: neem de eerste helft van tokens als beschrijving
        half = max(1, len(tokens)//2)
        desc = " ".join(tokens[:half])

    desc_norm = normalize_desc(desc)
    category = detect_category(desc_norm)

    return Item(
        description_raw=desc,
        description_norm=desc_norm,
        category=category,
        qty=qty,
        unit=unit,
        unit_price=unit_price,
        total_price=total_price,
        source=source
    )

def parse_summary_from_line(line: str, source: str) -> Optional[SummaryLine]:
    kind = classify_line(line)
    if kind == "line":
        return None
    money = find_money_tokens(line)
    if not money:
        return None
    amount = money[-1][1]
    label = normalize_whitespace(line)
    return SummaryLine(kind=kind, label=label, amount=amount, source=source)

# ===================== Hoofdroutine: PDF -> DataFrames =====================

def pdf_to_items_and_summary(pdf_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not has_text_in_pdf(pdf_path):
        raise ValueError(f"Geen doorzoekbare tekst gevonden in PDF '{pdf_path}'. "
                         f"Voer eerst OCR uit (bijv. OCRmyPDF) of vraag om een tekst-PDF.")

    lines = extract_lines(pdf_path)
    items: List[Item] = []
    summaries: List[SummaryLine] = []

    for ln in lines:
        s = parse_summary_from_line(ln, source=Path(pdf_path).name)
        if s:
            summaries.append(s)
            continue
        it = parse_item_from_line(ln, source=Path(pdf_path).name)
        if it:
            items.append(it)

    df_items = pd.DataFrame([asdict(i) for i in items]) if items else pd.DataFrame(
        columns=["description_raw", "description_norm", "category", "qty", "unit", "unit_price", "total_price", "source"]
    )
    df_summary = pd.DataFrame([asdict(s) for s in summaries]) if summaries else pd.DataFrame(
        columns=["kind", "label", "amount", "source"]
    )
    return df_items, df_summary

# ===================== Vergelijking =====================

def compare_offers(df_old: pd.DataFrame, df_new: pd.DataFrame, price_tol: float = 0.01):
    """
    Vergelijk items op basis van genormaliseerde beschrijving.
    price_tol: tolerantie in euro voor prijsvergelijking (afrondverschillen).
    """
    # Kies relevante kolommen
    keep = ["description_norm", "description_raw", "category", "qty", "unit", "unit_price", "total_price", "source"]
    df_old = df_old[keep].copy() if not df_old.empty else pd.DataFrame(columns=keep)
    df_new = df_new[keep].copy() if not df_new.empty else pd.DataFrame(columns=keep)

    # De-duplicatie door sommeren op description_norm (soms splitst PDF een item over meerdere regels)
    agg = {
        "description_raw": "first",
        "category": "first",
        "qty": "sum",
        "unit": "first",
        "unit_price": "mean",
        "total_price": "sum",
        "source": "first",
    }
    old_ag = df_old.groupby("description_norm", dropna=False).agg(agg).reset_index()
    new_ag = df_new.groupby("description_norm", dropna=False).agg(agg).reset_index()

    # Full outer join
    merged = old_ag.merge(new_ag, on="description_norm", how="outer", suffixes=("_old", "_new"), indicator=True)

    added = merged[merged["_merge"] == "right_only"].copy()
    removed = merged[merged["_merge"] == "left_only"].copy()

    both = merged[merged["_merge"] == "both"].copy()

    # Gewijzigd: qty/unit/price verschillen
    def differs(a, b, tol=price_tol):
        if pd.isna(a) and pd.isna(b):
            return False
        if isinstance(a, float) and isinstance(b, float):
            return abs((a or 0) - (b or 0)) > tol
        return (str(a or "") != str(b or ""))

    changes = []
    for _, r in both.iterrows():
        diffs = {}
        for col in ["qty", "unit", "unit_price", "total_price", "category"]:
            a = r.get(f"{col}_old")
            n = r.get(f"{col}_new")
            # Voor qty en total/price numeriek vergelijken met tolerantie
            if col in {"qty", "unit_price", "total_price"}:
                a = float(a) if pd.notna(a) else None
                n = float(n) if pd.notna(n) else None
                if (a is None) != (n is None) or (a is not None and n is not None and abs(a - n) > price_tol):
                    diffs[col] = {"old": a, "new": n}
            else:
                if differs(a, n, tol=0.0):
                    diffs[col] = {"old": a, "new": n}
        if diffs:
            changes.append({
                "description_norm": r["description_norm"],
                "description_raw_old": r.get("description_raw_old"),
                "description_raw_new": r.get("description_raw_new"),
                **{f"{k}__old": v["old"] for k, v in diffs.items()},
                **{f"{k}__new": v["new"] for k, v in diffs.items()},
            })

    df_added = added[["description_norm", "description_raw_new", "category_new", "qty_new", "unit_new", "unit_price_new", "total_price_new"]].rename(
        columns=lambda c: c.replace("_new", "")
    )
    df_removed = removed[["description_norm", "description_raw_old", "category_old", "qty_old", "unit_old", "unit_price_old", "total_price_old"]].rename(
        columns=lambda c: c.replace("_old", "")
    )
    df_changed = pd.DataFrame(changes)

    return df_added, df_removed, df_changed, old_ag, new_ag

def summarize_totals(df_summary: pd.DataFrame) -> Dict[str, float]:
    out = {"subtotal": None, "vat": None, "total": None, "discount": 0.0}
    if df_summary.empty:
        return out
    # Pak laatste waarde van elk type (meestal onderaan)
    for kind in ["subtotal", "vat", "total", "discount"]:
        rows = df_summary[df_summary["kind"] == kind]
        if not rows.empty:
            val = float(rows.iloc[-1]["amount"])
            out[kind] = val
    return out

# ===================== Rapportage =====================

def write_excel_report(path_xlsx: str,
                       df_items_old: pd.DataFrame,
                       df_items_new: pd.DataFrame,
                       df_added: pd.DataFrame,
                       df_removed: pd.DataFrame,
                       df_changed: pd.DataFrame,
                       summary_old: Dict[str, float],
                       summary_new: Dict[str, float]):
    with pd.ExcelWriter(path_xlsx, engine="openpyxl") as xw:
        df_items_old.to_excel(xw, index=False, sheet_name="Items_old")
        df_items_new.to_excel(xw, index=False, sheet_name="Items_new")
        df_added.to_excel(xw, index=False, sheet_name="Diff_added")
        df_removed.to_excel(xw, index=False, sheet_name="Diff_removed")
        df_changed.to_excel(xw, index=False, sheet_name="Diff_price_changed")

        # Summary-sheet
        summ_rows = []
        for label, d in [("OLD", summary_old), ("NEW", summary_new)]:
            summ_rows.append({"Offerte": label, "Subtotaal": d.get("subtotal"), "Korting": d.get("discount"), "BTW": d.get("vat"), "Totaal": d.get("total")})
        df_summ = pd.DataFrame(summ_rows)
        df_summ.to_excel(xw, index=False, sheet_name="Summary")

def write_text_summary(path_txt: str,
                       df_added: pd.DataFrame,
                       df_removed: pd.DataFrame,
                       df_changed: pd.DataFrame,
                       summary_old: Dict[str, float],
                       summary_new: Dict[str, float],
                       offer_old: str,
                       offer_new: str):
    lines = []
    lines.append(f"Vergelijkingsrapport voor offertes")
    lines.append(f"Oud: {offer_old}")
    lines.append(f"Nieuw: {offer_new}")
    lines.append("-" * 80)
    lines.append("Samenvatting bedragen:")
    def fmt(d):
        def f(x):
            return "-" if x is None else f"€ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        return f"Subtotaal: {f(d.get('subtotal'))} | Korting: {f(d.get('discount'))} | BTW: {f(d.get('vat'))} | Totaal: {f(d.get('total'))}"
    lines.append(f"  Oud:  {fmt(summary_old)}")
    lines.append(f"  Nieuw:{fmt(summary_new)}")
    lines.append("")
    lines.append(f"Toegevoegd: {len(df_added)} | Verwijderd: {len(df_removed)} | Gewijzigd: {len(df_changed)}")
    lines.append("")

    # Top 10 prijsstijgingen/dalingen op totaalprijs
    if not df_changed.empty:
        tmp = []
        for _, r in df_changed.iterrows():
            old = r.get("total_price__old")
            new = r.get("total_price__new")
            if pd.notna(old) and pd.notna(new):
                tmp.append((r.get("description_norm"), new - old, old, new))
        tmp.sort(key=lambda x: abs(x[1]), reverse=True)
        lines.append("Top wijzigingen (totaalprijs):")
        for desc, dif, old, new in tmp[:10]:
            s_old = f"€ {old:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
            s_new = f"€ {new:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
            s_dif = f"€ {dif:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
            lines.append(f"  • {desc} | oud: {s_old} → nieuw: {s_new} (Δ {s_dif})")
        lines.append("")

    # Lijsten
    if not df_added.empty:
        lines.append("Toegevoegde posten:")
        for _, r in df_added.iterrows():
            desc = r.get("description_norm")
            tot = r.get("total_price")
            s_tot = "-" if pd.isna(tot) else f"€ {tot:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
            lines.append(f"  + {desc} (totaal: {s_tot})")
        lines.append("")
    if not df_removed.empty:
        lines.append("Verwijderde posten:")
        for _, r in df_removed.iterrows():
            desc = r.get("description_norm")
            tot = r.get("total_price")
            s_tot = "-" if pd.isna(tot) else f"€ {tot:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
            lines.append(f"  - {desc} (totaal: {s_tot})")
        lines.append("")
    with open(path_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# ===================== Main: voer vergelijking uit =====================

def compare_pdf_offers(pdf_old: str, pdf_new: str, out_prefix: str = "offerte_vergelijk", price_tol: float = 0.01):
    items_old, summary_old_df = pdf_to_items_and_summary(pdf_old)
    items_new, summary_new_df = pdf_to_items_and_summary(pdf_new)

    df_added, df_removed, df_changed, old_ag, new_ag = compare_offers(items_old, items_new, price_tol=price_tol)

    summ_old = summarize_totals(summary_old_df)
    summ_new = summarize_totals(summary_new_df)

    out_dir = Path("offerte_diff_output")
    out_dir.mkdir(parents=True, exist_ok=True)

    xlsx_path = out_dir / f"{out_prefix}.xlsx"
    txt_path = out_dir / f"{out_prefix}.txt"

    write_excel_report(str(xlsx_path), old_ag, new_ag, df_added, df_removed, df_changed, summ_old, summ_new)
    write_text_summary(str(txt_path), df_added, df_removed, df_changed, summ_old, summ_new, Path(pdf_old).name, Path(pdf_new).name)

    return str(xlsx_path), str(txt_path)

# ---------------------------- Uitvoeren (pas aan) -----------------------------

if __name__ == "__main__":
    # Voorbeeld – vervang door jouw bestandsnamen:
    OLD = "offerte_oud.pdf"
    NEW = "offerte_nieuw.pdf"

    xlsx, txt = compare_pdf_offers(OLD, NEW, out_prefix="vergelijk_demo", price_tol=0.02)
    print("Klaar.")
    print("Excel-rapport:", xlsx)
    print("Tekstrapport:", txt)

