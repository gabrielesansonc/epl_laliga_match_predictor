"""HTTP API wrapper for match inference."""

from html import escape
import re

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import PlainTextResponse, Response

from match_inference import predict_match_probability


app = FastAPI(title="Match Prediction API")


def _parse_match_text(message: str) -> tuple[str, str]:
    """Parse a free-text message into home and away teams."""
    cleaned = message.strip()
    if not cleaned:
        raise ValueError("Send a message like 'barcelona, madrid'.")

    for separator in [",", " vs ", " v ", " - "]:
        if separator in cleaned.lower():
            if separator.strip() in {",", "-"}:
                parts = [part.strip() for part in cleaned.split(separator, maxsplit=1)]
            else:
                parts = re.split(separator, cleaned, maxsplit=1, flags=re.IGNORECASE)
                parts = [part.strip() for part in parts]
            if len(parts) == 2 and all(parts):
                return parts[0], parts[1]

    raise ValueError("Could not parse teams. Use 'home team, away team'.")


def _twiml_message(message: str) -> str:
    """Build a minimal TwiML response."""
    return f'<?xml version="1.0" encoding="UTF-8"?><Response><Message>{escape(message)}</Message></Response>'


@app.get("/", response_class=PlainTextResponse)
def healthcheck() -> str:
    return "Match Prediction API is running."


@app.get("/predict", response_class=PlainTextResponse)
def predict_match(home_team: str, away_team: str) -> str:
    try:
        return predict_match_probability(home_team, away_team)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/whatsapp/webhook")
def whatsapp_webhook(Body: str = Form(...)) -> Response:
    try:
        home_team, away_team = _parse_match_text(Body)
        reply = predict_match_probability(home_team, away_team)
    except ValueError as exc:
        reply = str(exc)
    except Exception:
        reply = "Something went wrong while generating the prediction."

    return Response(content=_twiml_message(reply), media_type="application/xml")
