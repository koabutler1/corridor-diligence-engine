# Corridor Diligence Engine

A screening tool for industrial real estate corridors. Load a market-level quarterly time series, group markets into a corridor, and get deterministic supply and demand metrics: vacancy, rent growth, months of supply, forward months of supply, and an excess-vacancy model that splits supply into standing vacancy and pipeline.

An optional LLM layer extracts claims from broker text and drafts commentary, but **it never computes a number**. Every metric is arithmetic done by the engine, and broker claims are judged by code against fixed thresholds.

> This public build runs on a **synthetic, procedurally generated dataset only**. It contains no licensed market data.

## What it does

- **Corridor builder:** pick markets by hand, from a preset corridor list (I-81, I-35, I-10, etc.), or let the model propose a mapping
- **Core metrics:** vacancy, inventory-weighted rent growth, months of supply against a configurable pre-COVID absorption baseline, forward months of supply from the construction pipeline
- **Excess-vacancy model:** equilibrium vacancy = each market's average over the baseline window; months of supply = (excess vacant SF + under-construction SF) / monthly baseline absorption
- **National benchmarking:** the corridor's percentile against every market in the file, with quartile-based TIGHT / LOOSE flags
- **Claims reconciler:** paste broker text, the model extracts claims, code compares them to engine values
- **Exports:** Excel workbook, PowerPoint deck, and a one-page HTML note, all generated in the browser

## Structure

```
app/index.html              front end (single file, no build step)
api/screen/                 Azure Function: stdlib-only Python metric engine
api/ai/                     Azure Function: LLM claim extraction, chat, narrative
staticwebapp.config.json    Azure Static Web Apps routing and auth
```

## Running it

Deployed as an Azure Static Web App (`app_location: app`, `api_location: api`). To enable the AI features, set these application settings (never commit them):

| Setting | Value |
|---|---|
| `LLM_PROVIDER` | `anthropic` or `azure` |
| `LLM_KEY` | API key |
| `LLM_MODEL` | model or deployment name |
| `LLM_ENDPOINT` | required for Azure |

Without a key, the app runs deterministic-only.

`staticwebapp.config.json` requires a Microsoft sign-in for every route, which keeps the AI endpoint from being used anonymously. Loosen it if you want an open demo with AI turned off.

## Author

Koa Butler, Finance & MIS, Rochester Institute of Technology
