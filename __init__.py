"""SWA managed function: POST /api/screen — thin shell over engine_stdlib."""
import json
import azure.functions as func
from . import engine_stdlib as eng


def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        body = req.get_json()
    except ValueError:
        return func.HttpResponse(json.dumps({"error": "Body must be JSON."}),
                                 status_code=400, mimetype="application/json")
    out = eng.screen_request(
        str(body.get("csv", "")).encode(),
        body.get("selection", []),
        body.get("window_start", "2015 Q1"),
        body.get("window_end", "2019 Q4"),
        body.get("mapping") or None)
    return func.HttpResponse(json.dumps(out), mimetype="application/json")
