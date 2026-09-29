"""Phase 2 LLM core — the AI boundary, implemented.

Division of labor:
  - The MODEL does exactly two jobs: extract quantitative claims from broker
    prose, and draft the narrative from already-computed figures.
  - CODE does the judging: reconciliation verdicts are |claim - engine| vs
    fixed thresholds. The model never decides whether a claim is right.

Provider-agnostic via environment variables (SWA "Environment variables" /
App Service application settings; Key Vault reference in production):
  LLM_PROVIDER  "anthropic" or "azure"
  LLM_KEY       the API key (never in code, never in the repo)
  LLM_MODEL     e.g. "claude-sonnet-4-6" or an Azure deployment name
  LLM_ENDPOINT  required for azure: the chat-completions URL of the deployment

No key configured -> {"configured": false, ...}: the socket exists; the app
stays deterministic-only until a key is configured.

stdlib only (urllib) — no SDK, keeps the SWA free-tier build trivial.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request

# The judge: thresholds in code, not model opinion. Tune to taste.
THRESHOLDS = {
    "vacancy_rate": 1.0,            # percentage points
    "rent_growth": 2.0,             # percentage points
    "months_of_supply": 2.0,        # months
    "institutional_ownership": 5.0, # percentage points
    "_default": 1.0,
}

EXTRACT_SYSTEM = (
    "You extract quantitative market claims from commercial real estate broker "
    "reports. Return ONLY a JSON array, no prose, no markdown fences. Each item: "
    '{"metric": one of ["vacancy_rate","rent_growth","months_of_supply",'
    '"institutional_ownership"], "value": number (percent as the number, e.g. '
    '5.5 for 5.5%), "unit": string, "quote": the short phrase from the text '
    "making the claim}. Only include claims mapping to those four metrics. "
    "If none, return []."
)

MAP_SYSTEM = (
    "You map named US industrial corridors to market names. Given a "
    "corridor name and a list of available market names, return ONLY a JSON "
    "array of the market names that belong to that corridor, chosen STRICTLY "
    "from the provided list (exact strings, no edits). Be inclusive of the "
    "metros the corridor's highway(s) actually connect. If nothing plausibly "
    "matches, return []."
)

DECK_SYSTEM = (
    "You draft slide content for an institutional real estate research deck. "
    "Use ONLY the figures provided; never invent a number. Voice: institutional, "
    "hedged. Return ONLY JSON, no fences: {\"headline\": string (<=12 words), "
    "\"bullets\": [4-5 strings, each <=22 words, each grounded in a provided "
    "figure], \"outlook\": string (<=45 words)}."
)

NARRATIVE_SYSTEM = (
    "You draft submarket commentary for an institutional real estate research "
    "team. Voice: institutional, confident but hedged — analysis 'suggests', "
    "data is 'consistent with'. No exclamation points. HARD RULE: use ONLY the "
    "figures provided. Do not invent, adjust, or estimate any number. One "
    "paragraph, 90-130 words."
)

CHAT_SYSTEM = (
    "You are Corridor Chat, the conversational layer of the Corridor Diligence "
    "Engine, for an institutional real-estate research team. Voice: "
    "institutional, concise, hedged. HARD RULES: (1) Use ONLY the numbers in "
    "the provided engine_facts and reconciled_claims — never compute, invent, "
    "extrapolate, or estimate a figure. If asked for a number that is not "
    "provided, say the engine does not compute it and point to what is "
    "available. (2) If the user pastes broker or third-party text containing "
    "quantitative claims, extract each claim and compare it to the matching "
    "engine fact, stating the difference and whether it exceeds the fixed "
    "thresholds (vacancy 1.0pp, rent growth 2.0pp, months of supply 2.0, "
    "institutional ownership 5.0pp); note that the Reconciler panel is the "
    "code-adjudicated path and your comparison is advisory. (3) You may "
    "explain methodology using the summary provided. (4) Keep replies under "
    "180 words unless asked for depth. (5) Never present yourself as having "
    "computed anything — the deterministic engine computed it."
)

METHOD_SUMMARY = (
    "Methodology summary: months of supply = total vacant SF / monthly "
    "absorption baseline (avg quarterly net absorption over the configurable "
    "pre-COVID window, / 3). Forward MoS = under-construction SF / same "
    "baseline. Excess-vacancy model (per market) = (excess vacancy above the market's "
    "window-average equilibrium + pipeline SF) / baseline; split into excess "
    "months (standing vacancy, clears on demand) and pipeline months (burns "
    "off on delivery schedule). Vacancy is corridor-aggregated; rent growth "
    "is inventory-weighted yoy. National percentile = share of markets in "
    "the loaded file with lower simple MoS. Verdict flags are "
    "distribution-based (bottom/top quartile). Forecast rows are excluded "
    "via the As Of cap. All metrics are arithmetic and auditable; AI only "
    "extracts claims and drafts prose."
)

LONGREAD_SYSTEM = (
    "You draft the opening read of a one-page institutional research note on an "
    "industrial corridor. Voice: institutional, hedged. Three paragraphs, "
    "230-300 words total. HARD RULES: use ONLY the figures provided in the "
    "metrics object; never invent, adjust, or estimate a number. If broker "
    "claims are provided, you may cite them with attribution — e.g. 'the broker "
    "excerpt cites 5.5% vacancy' — and you MUST note the reconciliation flag "
    "when you do (matches the engine, or diverges by the stated amount); cite "
    "only claims from the provided list. No headers, no bullets, prose only."
)


def config():
    return {
        "provider": os.environ.get("LLM_PROVIDER", "").strip().lower(),
        "key": os.environ.get("LLM_KEY", "").strip(),
        "model": os.environ.get("LLM_MODEL", "").strip(),
        "endpoint": os.environ.get("LLM_ENDPOINT", "").strip(),
    }


def is_configured(cfg=None) -> bool:
    c = cfg or config()
    if c["provider"] == "anthropic":
        return bool(c["key"] and c["model"])
    if c["provider"] == "azure":
        return bool(c["key"] and c["endpoint"])
    return False


def call_llm(system: str, user: str, max_tokens: int = 1000, cfg=None,
             _transport=None) -> str:
    """Single model call. _transport is injectable for tests (no network here)."""
    c = cfg or config()
    if c["provider"] == "anthropic":
        url = "https://api.anthropic.com/v1/messages"
        headers = {"Content-Type": "application/json", "x-api-key": c["key"],
                   "anthropic-version": "2023-06-01"}
        body = {"model": c["model"] or "claude-sonnet-4-6",
                "max_tokens": max_tokens, "system": system,
                "messages": [{"role": "user", "content": user}]}
        raw = (_transport or _http)(url, headers, body)
        data = json.loads(raw)
        return "\n".join(b.get("text", "") for b in data.get("content", [])
                         if b.get("type") == "text")
    if c["provider"] == "azure":
        # OpenAI-style chat completions (Azure OpenAI / AI Foundry deployments)
        headers = {"Content-Type": "application/json", "api-key": c["key"]}
        body = {"messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
                "max_tokens": max_tokens}
        if c["model"]:
            body["model"] = c["model"]
        raw = (_transport or _http)(c["endpoint"], headers, body)
        data = json.loads(raw)
        return data["choices"][0]["message"]["content"]
    raise RuntimeError("LLM not configured")


def _http(url: str, headers: dict, body: dict) -> str:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode()


# ---- AI job #1: extraction (model) + reconciliation (code) -----------------

def parse_claims(raw: str) -> list[dict]:
    """Strict parse of the model's JSON; drop anything malformed."""
    clean = re.sub(r"```json|```", "", raw).strip()
    try:
        items = json.loads(clean)
    except json.JSONDecodeError:
        return []
    out = []
    if not isinstance(items, list):
        return out
    for it in items:
        try:
            metric = str(it["metric"])
            value = float(it["value"])
        except (KeyError, TypeError, ValueError):
            continue
        out.append({"metric": metric, "value": value,
                    "unit": str(it.get("unit", "")),
                    "quote": str(it.get("quote", ""))[:200]})
    return out


def reconcile(claims: list[dict], engine_values: dict) -> list[dict]:
    """DETERMINISTIC verdicts: |claim - engine| vs threshold. No AI here."""
    out = []
    for c in claims:
        eng = engine_values.get(c["metric"])
        if not isinstance(eng, (int, float)):
            out.append({**c, "engine": None, "diff": None,
                        "flag": "NO_ENGINE_VALUE"})
            continue
        diff = round(c["value"] - float(eng), 2)
        t = THRESHOLDS.get(c["metric"], THRESHOLDS["_default"])
        out.append({**c, "engine": eng, "diff": diff,
                    "flag": "DIVERGES" if abs(diff) > t else "MATCH"})
    return out


def call_llm_messages(system: str, messages: list, max_tokens: int, cfg=None,
                      _transport=None) -> str:
    c = cfg or config()
    if c["provider"] == "anthropic":
        url = c["endpoint"] or "https://api.anthropic.com/v1/messages"
        headers = {"x-api-key": c["key"], "anthropic-version": "2023-06-01",
                   "content-type": "application/json"}
        body = {"model": c["model"] or "claude-sonnet-4-6",
                "max_tokens": max_tokens, "system": system,
                "messages": messages}
        raw = (_transport or _http)(url, headers, body)
        data = json.loads(raw)
        return "".join(b.get("text", "") for b in data.get("content", [])
                       if b.get("type") == "text")
    else:  # azure openai-compatible
        url = c["endpoint"]
        headers = {"api-key": c["key"], "content-type": "application/json"}
        body = {"messages": [{"role": "system", "content": system}] + messages,
                "max_tokens": max_tokens}
        raw = (_transport or _http)(url, headers, body)
        data = json.loads(raw)
        return data["choices"][0]["message"]["content"]


def run_chat(messages: list, metrics: dict | None, claims: list | None,
             corridor: str, window: str, cfg=None, _transport=None) -> dict:
    if not is_configured(cfg):
        return {"configured": False,
                "note": "No LLM endpoint configured — Corridor Chat is dormant."}
    ctx = {"corridor": corridor, "baseline_window": window,
           "engine_facts": metrics or "NO SCREEN HAS BEEN RUN YET",
           "reconciled_claims": claims or []}
    system = (CHAT_SYSTEM + " " + METHOD_SUMMARY +
              " Context (the ONLY numbers you may use): " + json.dumps(ctx))
    reply = call_llm_messages(system, messages, 700, cfg, _transport)
    return {"configured": True, "reply": reply.strip()}


def run_reconcile(broker_text: str, engine_values: dict, cfg=None,
                  _transport=None) -> dict:
    if not is_configured(cfg):
        return {"configured": False,
                "note": "No LLM endpoint configured. Set LLM_PROVIDER / LLM_KEY "
                        "(+ LLM_MODEL or LLM_ENDPOINT) in the app settings; the "
                        "module activates without a redeploy."}
    raw = call_llm(EXTRACT_SYSTEM,
                   "Extract the claims from this report excerpt:\n\n" + broker_text,
                   800, cfg, _transport)
    claims = parse_claims(raw)
    return {"configured": True, "claims": reconcile(claims, engine_values),
            "extraction_model": (cfg or config())["model"] or "(default)"}


def run_map_corridor(corridor: str, available_markets: list[str], cfg=None,
                     _transport=None) -> dict:
    """AI PROPOSES corridor membership; the analyst confirms via checkboxes.
    Output is validated against the provided list — the model cannot invent
    a market, only select from what the file contains."""
    if not is_configured(cfg):
        return {"configured": False,
                "note": "No LLM endpoint configured — pick markets manually or "
                        "use the built-in corridor catalog."}
    prompt = ("Corridor: " + corridor + chr(10) + "Available markets:" + chr(10)
              + chr(10).join(available_markets[:600]))
    raw = call_llm(MAP_SYSTEM, prompt, 1200, cfg, _transport)
    clean = re.sub(r"```json|```", "", raw).strip()
    try:
        picks = json.loads(clean)
    except json.JSONDecodeError:
        return {"configured": True, "markets": [],
                "error": "Model returned non-JSON; try again."}
    allowed = set(available_markets)
    markets = [m for m in picks if isinstance(m, str) and m in allowed]
    return {"configured": True, "markets": markets,
            "note": "AI-proposed membership — review and edit before screening; "
                    "corridor definition is a definitional judgment call."}


def run_deck_bullets(corridor: str, window: str, metrics: dict, cfg=None,
                     _transport=None) -> dict:
    """Slide copy drafted from computed figures only; falls back to template."""
    if not is_configured(cfg):
        return {"configured": False}
    raw = call_llm(DECK_SYSTEM,
                   f"Corridor: {corridor}. Baseline window: {window}. Figures "
                   f"(the ONLY numbers you may use): {json.dumps(metrics)}",
                   700, cfg, _transport)
    clean = re.sub(r"```json|```", "", raw).strip()
    try:
        out = json.loads(clean)
        assert isinstance(out.get("bullets"), list)
    except (json.JSONDecodeError, AssertionError):
        return {"configured": True, "error": "Model returned non-JSON."}
    return {"configured": True, "headline": str(out.get("headline", ""))[:120],
            "bullets": [str(b)[:200] for b in out["bullets"][:5]],
            "outlook": str(out.get("outlook", ""))[:400]}


# ---- AI job #2: narrative from computed figures only ------------------------

def run_narrative(corridor: str, window: str, metrics: dict, cfg=None,
                  _transport=None, length: str = "short",
                  claims: list | None = None) -> dict:
    if not is_configured(cfg):
        return {"configured": False,
                "note": "No LLM endpoint configured — template narrative remains "
                        "in effect."}
    facts = json.dumps(metrics)
    if length == "long":
        user = (f"Corridor: {corridor}. Baseline window: {window}. Computed "
                f"figures (the ONLY numbers you may use): {facts}.")
        if claims:
            user += (" Reconciled broker claims you may cite (with their flags): "
                     + json.dumps(claims[:12]))
        user += " Draft the three-paragraph opening read."
        text = call_llm(LONGREAD_SYSTEM, user, 900, cfg, _transport)
    else:
        text = call_llm(
            NARRATIVE_SYSTEM,
            f"Corridor: {corridor}. Baseline window: {window}. Computed figures "
            f"(the ONLY numbers you may use): {facts}. Draft the submarket paragraph.",
            500, cfg, _transport)
    return {"configured": True, "narrative": text.strip(),
            "disclaimer": "Figures deterministic; prose model-assisted; review before use."}
