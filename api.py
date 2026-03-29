"""HTTP API wrapper for match inference."""

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

from match_inference import predict_match_probability


app = FastAPI(title="Match Prediction API")


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
