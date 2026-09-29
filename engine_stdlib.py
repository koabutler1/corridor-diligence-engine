"""Stdlib-only market engine for the Static Web Apps managed function.

Zero third-party dependencies (the SWA free-tier build budget rules out pandas).
Definitions match the browser engine in app/index.html.
"""
from __future__ import annotations

import csv
import io
import re

FIELDS = {
    "submarket":    {"required": True,  "cands": ["Submarket", "Submarket Name", "Geography Name", "Market", "Market Name"]},
    "period":       {"required": True,  "cands": ["Period", "Quarter", "Date", "As Of"]},
    "inventory_sf": {"required": True,  "cands": ["Inventory SF", "Inventory", "Existing SF", "Total Inventory SF"]},
    "vacant_sf":    {"required": False, "cands": ["Vacant SF", "Total Vacant SF"]},
    "vacancy_rate": {"required": False, "cands": ["Vacancy Rate", "Vacancy %", "Total Vacancy"]},
    "net_absorption": {"required": True, "cands": ["Net Absorption SF", "Net Absorption", "Absorption SF"]},
    "asking_rent":  {"required": False, "cands": ["Market Asking Rent/SF", "Asking Rent", "Market Asking Rent", "Asking Rent/SF", "Rent"]},
    "under_construction_sf": {"required": False, "cands": ["Under Construction SF", "UC SF", "Under Construction"]},
    "class_a_vac":  {"required": False, "cands": ["Class A Vacancy %", "Class A Vacancy Rate", "4-5 Star Vacancy Rate"]},
    "class_b_vac":  {"required": False, "cands": ["Class B Vacancy %", "Class B Vacancy Rate", "1-3 Star Vacancy Rate"]},
}


def resolve_columns(headers):
    actual = {h.strip().lower(): h for h in headers}
    resolved, missing = {}, []
    for f, spec in FIELDS.items():
        found = None
        for c in spec["cands"]:
            if c.strip().lower() in actual:
                found = actual[c.strip().lower()]
                break
        resolved[f] = found
        if found is None and spec["required"]:
            missing.append(f)
    return resolved, missing


def norm_period(s):
    s = str(s).strip().upper()
    m = re.search(r"(\d{4})\s*Q([1-4])", s)
    if m: return m.group(1) + "Q" + m.group(2)
    m = re.search(r"Q([1-4])\s*(\d{4})", s)
    if m: return m.group(2) + "Q" + m.group(1)
    m = re.match(r"^([1-4])Q(\d{2})$", s)
    if m: return "20" + m.group(2) + "Q" + m.group(1)
    m = re.match(r"^(\d{4})-(\d{2})", s)
    if m: return m.group(1) + "Q" + str((int(m.group(2)) + 2) // 3)
    return None


def q_back(p, n):
    y, q = int(p[:4]), int(p[5:])
    q -= n
    while q < 1:
        q += 4
        y -= 1
    return f"{y}Q{q}"


def _num(v):
    try:
        return float(re.sub(r"[$,%\s]", "", str(v)))
    except (ValueError, TypeError):
        return None


def screen_request(csv_bytes, selection, window_start, window_end, mapping=None):
    """Same contract as api/service.py: CSV bytes + selection -> screen dict."""
    try:
        text = csv_bytes.decode("utf-8-sig", errors="replace")
        reader = csv.DictReader(io.StringIO(text))
        headers = reader.fieldnames or []
        records = list(reader)
    except Exception as e:  # noqa: BLE001
        return {"error": f"Could not read the file as CSV: {e}"}
    if not records:
        return {"error": "That file has no data rows."}

    if mapping:
        resolved = {f: (c if c in headers else None) for f, c in mapping.items()}
        missing = [f for f, spec in FIELDS.items() if spec["required"] and not resolved.get(f)]
    else:
        resolved, missing = resolve_columns(headers)
    if missing:
        return {"error": "Missing required column(s): " + ", ".join(missing), "headers": headers}
    if not (resolved.get("vacant_sf") or resolved.get("vacancy_rate")):
        return {"error": "Need either a Vacant SF column or a Vacancy Rate column."}
    if not selection:
        return {"error": "Select at least one submarket."}

    win = {"start": norm_period(window_start) or "2015Q1",
           "end": norm_period(window_end) or "2019Q4"}

    rows = []
    for r in records:
        p = norm_period(r.get(resolved["period"], ""))
        sm = str(r.get(resolved["submarket"], "")).strip()
        if not p or sm not in selection:
            continue
        g = lambda f: _num(r.get(resolved[f], "")) if resolved.get(f) else None  # noqa: E731
        rows.append({"sm": sm, "p": p, "inv": g("inventory_sf"), "abs": g("net_absorption"),
                     "vsf": g("vacant_sf"), "vr": g("vacancy_rate"), "rent": g("asking_rent"),
                     "uc": g("under_construction_sf"), "va": g("class_a_vac"), "vb": g("class_b_vac")})
    if not rows:
        return {"error": "No rows matched the selected submarkets (check the period format)."}

    periods = sorted({r["p"] for r in rows})
    latest = periods[-1]
    prior = q_back(latest, 4)
    L = [r for r in rows if r["p"] == latest]

    def vac(r):
        if resolved.get("vacant_sf"):
            return r["vsf"] or 0.0
        if r["inv"] is not None and r["vr"] is not None:
            x = r["vr"] / 100.0 if r["vr"] > 1 else r["vr"]
            return r["inv"] * x
        return 0.0

    tot_inv = sum(r["inv"] or 0 for r in L)
    tot_vac = sum(vac(r) for r in L)
    tot_uc = sum(r["uc"] or 0 for r in L) if resolved.get("under_construction_sf") else None

    in_win = [r for r in rows if win["start"] <= r["p"] <= win["end"]]
    q_tot = {}
    for r in in_win:
        q_tot[r["p"]] = q_tot.get(r["p"], 0.0) + (r["abs"] or 0.0)
    monthly = (sum(q_tot.values()) / len(q_tot)) / 3.0 if q_tot else None
    ok_base = monthly is not None and monthly > 0

    def w_avg(rows_, key):
        s = w = 0.0
        for r in rows_:
            if r[key] is not None and r["inv"] and r["inv"] > 0:
                s += r[key] * r["inv"]
                w += r["inv"]
        return s / w if w else None

    rent_now = w_avg(L, "rent")
    rent_prior = w_avg([r for r in rows if r["p"] == prior], "rent")
    v_a = w_avg(L, "va")
    v_b = w_avg(L, "vb")
    if v_a is not None: v_a = round(v_a, 1)
    if v_b is not None: v_b = round(v_b, 1)

    def metric(name, unit, value, inputs, note=""):
        return {"name": name, "unit": unit, "value": value, "inputs": inputs, "note": note}

    m = {}
    m["months_of_supply"] = metric("Months of supply", "months",
        round(tot_vac / monthly, 1) if ok_base else None,
        {"as_of": latest, "total_vacant_sf": round(tot_vac),
         "monthly_absorption_baseline": round(monthly) if ok_base else None,
         "baseline_window": f"{win['start']}–{win['end']}", "window_quarters": len(q_tot)},
        "" if ok_base else "No absorption rows found inside the baseline window.")
    m["forward_months_of_supply"] = metric("Forward months of supply", "months",
        round(tot_uc / monthly, 1) if (ok_base and tot_uc is not None) else None,
        {"as_of": latest, "under_construction_sf": round(tot_uc) if tot_uc is not None else None,
         "monthly_absorption_baseline": round(monthly) if ok_base else None},
        "" if resolved.get("under_construction_sf") else "No under-construction column resolved.")
    m["vacancy_rate"] = metric("Overall vacancy", "%",
        round(100 * tot_vac / tot_inv, 1) if tot_inv else None,
        {"as_of": latest, "total_vacant_sf": round(tot_vac), "inventory_sf": round(tot_inv)})
    m["vacancy_by_class"] = metric("Vacancy A / B", "%",
        f"{v_a} / {v_b}" if (v_a is not None and v_b is not None) else None,
        {"as_of": latest, "class_A": v_a, "class_B": v_b,
         "method": "inventory-weighted average of submarket class rates"},
        ("Class B tighter than Class A" if v_b < v_a else "Class A tighter than Class B")
        if (v_a is not None and v_b is not None) else "Class-slice columns not present in this export.")
    m["rent_growth"] = metric("Rent growth", "% (yoy)",
        round(100 * (rent_now - rent_prior) / rent_prior, 1)
        if (rent_now is not None and rent_prior and rent_prior > 0) else None,
        {"as_of": latest, "prior_period": prior,
         "current": round(rent_now, 2) if rent_now is not None else None,
         "prior": round(rent_prior, 2) if rent_prior is not None else None,
         "method": "inventory-weighted asking rent"},
        "" if (rent_now is not None and rent_prior and rent_prior > 0)
        else f"Need asking-rent rows at both {latest} and {prior}.")

    return {"metrics": m, "latest": latest, "submarkets": len(selection),
            "rows": len(rows), "engine": "python-function"}
