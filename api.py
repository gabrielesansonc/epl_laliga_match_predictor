"""HTTP API — match inference + web frontend."""

import re
import time
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

from match_inference import predict_match_probability
from src.data.extract import align_team_names, load_upcoming_fixtures
from src.features.engineering import (
    FEATURE_COLUMNS,
    INV_LABEL_MAP,
    LABEL_MAP,
    assign_season,
    build_features,
    select_raw_columns,
)
from src.models.predict import load_model, predict


# ── Paths ─────────────────────────────────────────────────────────────────────

_ROOT = Path(__file__).resolve().parent
MODEL_PATH     = _ROOT / "models" / "rf_match_predict_model.pkl"
PROCESSED_PATH = _ROOT / "data" / "processed" / "matches_features.csv"
HISTORIC_CSV   = _ROOT / "data" / "raw" / "EPL_LaLiga_Data.csv"
STATIC_DIR     = _ROOT / "static"

RECENT_SOURCES = {
    "LaLiga_2023_24": "https://www.football-data.co.uk/mmz4281/2324/SP1.csv",
    "LaLiga_2024_25": "https://www.football-data.co.uk/mmz4281/2425/SP1.csv",
    "LaLiga_2025_26": "https://www.football-data.co.uk/mmz4281/2526/SP1.csv",
    "EPL_2023_24":    "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
    "EPL_2024_25":    "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
    "EPL_2025_26":    "https://www.football-data.co.uk/mmz4281/2526/E0.csv",
}

OUTCOME_LABELS = {"H": "Home Win", "D": "Draw", "A": "Away Win"}
LEAGUE_LABELS  = {"SP1": "La Liga", "E0": "Premier League", "UCL": "Champions League"}
_CACHE_TTL     = 3600  # seconds

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Match Prediction API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

_cache: dict = {}


def _cached(key: str, fn, ttl: int = _CACHE_TTL):
    if key in _cache:
        data, ts = _cache[key]
        if time.time() - ts < ttl:
            return data
    data = fn()
    _cache[key] = (data, time.time())
    return data


# ── Data helpers ──────────────────────────────────────────────────────────────

def _load_recent_history() -> pd.DataFrame:
    frames = [pd.read_csv(url) for url in RECENT_SOURCES.values()]
    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = select_raw_columns(df)
    df["Season"] = df["Date"].apply(assign_season)
    return df


def _load_model():
    return _cached("model", lambda: load_model(MODEL_PATH), ttl=86400)


def _build_insights(row: pd.Series) -> list[dict]:
    home, away = row["HomeTeam"], row["AwayTeam"]
    lines: list[dict] = []

    h_pts = row.get("HomeRollingPoints10")
    a_pts = row.get("AwayRollingPoints10")
    if pd.notna(h_pts):
        if h_pts >= 21:
            lines.append({"icon": "↑", "text": f"{home} strong form — {int(h_pts)} pts last 10", "team": "H"})
        elif h_pts <= 8:
            lines.append({"icon": "↓", "text": f"{home} struggling — {int(h_pts)} pts last 10", "team": "H"})
    if pd.notna(a_pts):
        if a_pts >= 21:
            lines.append({"icon": "↑", "text": f"{away} strong form — {int(a_pts)} pts last 10", "team": "A"})
        elif a_pts <= 8:
            lines.append({"icon": "↓", "text": f"{away} struggling — {int(a_pts)} pts last 10", "team": "A"})

    h_gls = row.get("HomeRollingGoals20")
    h_rcv = row.get("HomeRollingReceivedGoals20")
    a_gls = row.get("AwayRollingGoals20")
    a_rcv = row.get("AwayRollingReceivedGoals20")
    if pd.notna(h_gls) and pd.notna(h_rcv) and (h_gls - h_rcv) >= 10:
        lines.append({"icon": "→", "text": f"{home} +{int(h_gls - h_rcv)} GD last 20", "team": "H"})
    if pd.notna(a_gls) and pd.notna(a_rcv) and (a_gls - a_rcv) >= 10:
        lines.append({"icon": "→", "text": f"{away} +{int(a_gls - a_rcv)} GD last 20", "team": "A"})

    h2h = row.get("HomeTeamPointsLast2Encounters")
    if pd.notna(h2h):
        if h2h >= 5:
            lines.append({"icon": "H2H", "text": f"{home} dominant in recent H2H", "team": "H"})
        elif h2h <= 1:
            lines.append({"icon": "H2H", "text": f"{away} dominant in recent H2H", "team": "A"})

    h_rank = row.get("HomeRank")
    a_rank = row.get("AwayRank")
    if pd.notna(h_rank) and h_rank <= 4:
        lines.append({"icon": f"#{int(h_rank)}", "text": f"{home} top-4 contention", "team": "H"})
    if pd.notna(a_rank) and a_rank <= 4:
        lines.append({"icon": f"#{int(a_rank)}", "text": f"{away} top-4 contention", "team": "A"})

    if not lines and pd.notna(h_pts) and pd.notna(a_pts):
        if abs(h_pts - a_pts) >= 6:
            better = home if h_pts > a_pts else away
            team_type = "H" if better == home else "A"
            lines.append({"icon": "→", "text": f"{better} in stronger recent form", "team": team_type})
        else:
            lines.append({"icon": "=", "text": "Teams evenly matched on recent form", "team": "D"})

    return lines[:3]


def _build_fixtures(league: str) -> list[dict]:
    history  = _cached("history", _load_recent_history)
    upcoming = load_upcoming_fixtures()

    valid_names  = history["HomeTeam"].dropna().unique().tolist()
    upcoming     = align_team_names(upcoming.copy(), valid_names)
    upcoming_keys = upcoming[["Div", "Date", "HomeTeam", "AwayTeam"]].copy()

    combined     = pd.concat([history, upcoming], ignore_index=True)
    combined     = build_features(combined, drop_na=False)
    upcoming_full = combined.merge(
        upcoming_keys.assign(IsUpcoming=True),
        on=["Div", "Date", "HomeTeam", "AwayTeam"],
        how="inner",
    )

    if upcoming_full.empty:
        return []

    preds = predict(_load_model(), upcoming_full)

    insight_cols = [
        "HomeRollingPoints10", "AwayRollingPoints10",
        "HomeRollingGoals20", "AwayRollingGoals20",
        "HomeRollingReceivedGoals20", "AwayRollingReceivedGoals20",
        "HomeTeamPointsLast2Encounters", "HomeRank", "AwayRank",
    ]
    extra = [c for c in insight_cols if c in upcoming_full.columns]

    result = pd.concat(
        [upcoming_full[["Div", "Date", "HomeTeam", "AwayTeam"] + extra].reset_index(drop=True),
         preds.reset_index(drop=True)],
        axis=1,
    ).sort_values("Date").reset_index(drop=True)

    if league != "ALL":
        result = result[result["Div"] == league]

    fixtures = []
    for _, row in result.iterrows():
        d = pd.Timestamp(row["Date"])
        fixtures.append({
            "home_team": row["HomeTeam"],
            "away_team": row["AwayTeam"],
            "date":      f"{d.strftime('%a')} {d.day} {d.strftime('%b')}",
            "league":    LEAGUE_LABELS.get(row["Div"], row["Div"]),
            "div":       row["Div"],
            "prob_h":    round(float(row["Prob_H"]), 4),
            "prob_d":    round(float(row["Prob_D"]), 4),
            "prob_a":    round(float(row["Prob_A"]), 4),
            "predicted": row["PredictedResult"],
            "insights":  _build_insights(row),
        })
    return fixtures


def _get_validation_df() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_PATH, parse_dates=["Date"])
    df["Season"] = df["Date"].apply(assign_season)

    X = df[FEATURE_COLUMNS].astype("float32")
    y = df["FTR"].map(LABEL_MAP)
    _, X_test, _, _ = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

    test_df = df.loc[X_test.index, ["Div", "Date", "Season", "HomeTeam", "AwayTeam", "FTR"]].copy()
    preds   = predict(_load_model(), X_test)

    result = pd.concat([test_df.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)
    result["Correct"]   = result["PredictedResult"] == result["FTR"]
    result["TrueLabel"] = result["FTR"].map(LABEL_MAP)
    return result


def _get_betting_baseline() -> pd.DataFrame:
    df = pd.read_csv(HISTORIC_CSV, low_memory=False)
    df["Date"]   = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df["Season"] = df["Date"].apply(assign_season)

    cols = ["Season", "Div", "FTR", "B365H", "B365D", "B365A"]
    df   = df[[c for c in cols if c in df.columns]].dropna()

    raw     = df[["B365H", "B365D", "B365A"]].apply(lambda x: 1 / x)
    ip      = raw.div(raw.sum(axis=1), axis=0)
    ip.columns = ["IP_H", "IP_D", "IP_A"]
    df["Pred"]    = np.array(["H", "D", "A"])[ip.values.argmax(axis=1)]
    df["Correct"] = df["Pred"] == df["FTR"]
    return df


def _get_betting_sim(val: pd.DataFrame) -> pd.DataFrame:
    raw = pd.read_csv(HISTORIC_CSV, low_memory=False)
    raw["Date"] = pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce")
    odds_cols   = ["Div", "Date", "HomeTeam", "AwayTeam", "B365H", "B365D", "B365A"]
    raw = raw[[c for c in odds_cols if c in raw.columns]].dropna()

    merged = val.merge(raw, on=["Div", "Date", "HomeTeam", "AwayTeam"], how="inner")
    inv    = merged[["B365H", "B365D", "B365A"]].apply(lambda s: 1 / s)
    inv    = inv.div(inv.sum(axis=1), axis=0)
    inv.columns = ["IP_H", "IP_D", "IP_A"]
    merged = pd.concat([merged, inv], axis=1)

    merged["OI_H"] = merged["Prob_H"] - merged["IP_H"]
    merged["OI_D"] = merged["Prob_D"] - merged["IP_D"]
    merged["OI_A"] = merged["Prob_A"] - merged["IP_A"]

    best           = merged[["OI_H", "OI_D", "OI_A"]].values.argmax(axis=1)
    merged["Bet"]  = np.array(["H", "D", "A"])[best]
    merged["BetOdds"] = 0.0
    for outcome, col in [("H", "B365H"), ("D", "B365D"), ("A", "B365A")]:
        mask = merged["Bet"] == outcome
        merged.loc[mask, "BetOdds"] = merged.loc[mask, col]

    merged["Won"] = merged["Bet"] == merged["FTR"]
    merged["PnL"] = np.where(merged["Won"], merged["BetOdds"] - 1.0, -1.0)
    merged = merged.sort_values("Date").reset_index(drop=True)
    merged["MatchNum"] = np.arange(1, len(merged) + 1)
    merged["CumPnL"]   = merged["PnL"].cumsum()
    return merged


def _build_recent_results(league: str) -> list[dict]:
    """Build list of last 10 completed matches with predictions vs actual results."""
    history = _cached("history", _load_recent_history)
    featured = build_features(history.copy(), drop_na=False)

    # Filter to completed matches
    completed = featured[featured["FTR"].notna() & featured["FTHG"].notna() & featured["FTAG"].notna()].copy()

    if league != "ALL":
        completed = completed[completed["Div"] == league]

    # Last 10 per Div, sorted by date descending
    recent = (
        completed.sort_values("Date", ascending=False)
        .groupby("Div", group_keys=False)
        .head(10)
        .sort_values(["Div", "Date"], ascending=[True, False])
        .reset_index(drop=True)
    )

    if recent.empty:
        return []

    preds = predict(_load_model(), recent)
    result = pd.concat([recent.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)

    out = []
    for _, row in result.iterrows():
        d = pd.Timestamp(row["Date"])
        ftr = row["FTR"]
        predicted = row["PredictedResult"]
        out.append({
            "date":       f"{d.strftime('%a')} {d.day} {d.strftime('%b %Y')}",
            "league":     LEAGUE_LABELS.get(row["Div"], row["Div"]),
            "home_team":  row["HomeTeam"],
            "away_team":  row["AwayTeam"],
            "home_goals": int(row["FTHG"]),
            "away_goals": int(row["FTAG"]),
            "result":     ftr,
            "predicted":  predicted,
            "correct":    bool(predicted == ftr),
            "prob_h":     round(float(row["Prob_H"]), 4),
            "prob_d":     round(float(row["Prob_D"]), 4),
            "prob_a":     round(float(row["Prob_A"]), 4),
        })
    return out


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_class=PlainTextResponse)
def healthcheck() -> str:
    return "Match Prediction API is running."


@app.get("/predict", response_class=PlainTextResponse)
def predict_text(home_team: str, away_team: str) -> str:
    try:
        return predict_match_probability(home_team, away_team)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/fixtures")
def get_fixtures(league: str = Query("SP1", pattern="^(SP1|E0)$")):
    try:
        fixtures = _cached(f"fixtures_{league}", lambda: _build_fixtures(league))
        return JSONResponse({"league": league, "count": len(fixtures), "fixtures": fixtures})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/predict")
def predict_json(home_team: str, away_team: str):
    try:
        result_str   = predict_match_probability(home_team, away_team)
        parts        = result_str.split(", ")
        matched_home = parts[0].split(":")[0].strip()
        prob_h       = float(parts[0].split(":")[1].strip())
        prob_d       = float(parts[1].split(":")[1].strip())
        matched_away = parts[2].split(":")[0].strip()
        prob_a       = float(parts[2].split(":")[1].strip())
        probs        = {"H": prob_h, "D": prob_d, "A": prob_a}
        predicted    = max(probs, key=probs.get)
        return JSONResponse({
            "home_team": matched_home,
            "away_team": matched_away,
            "prob_h":    round(prob_h, 4),
            "prob_d":    round(prob_d, 4),
            "prob_a":    round(prob_a, 4),
            "predicted": predicted,
        })
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/recent_results")
def get_recent_results(league: str = Query("ALL", pattern="^(ALL|SP1|E0)$")):
    try:
        results = _cached(f"recent_{league}", lambda: _build_recent_results(league))
        return JSONResponse({"league": league, "count": len(results), "results": results})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/validation")
def get_validation(league: str = Query("All", pattern="^(All|SP1|E0)$")):
    try:
        val = _cached("validation_df", _get_validation_df, ttl=86400)
        if league != "All":
            val = val[val["Div"] == league]
        if val.empty:
            raise HTTPException(status_code=404, detail="No validation data for this league.")

        overall_acc = float(val["Correct"].mean())

        # Season accuracy
        season_acc = (
            val.groupby("Season")["Correct"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "accuracy", "count": "matches"})
            .reset_index().sort_values("Season")
        )

        # Betting baseline
        betting = _cached("betting_baseline", _get_betting_baseline, ttl=86400)
        if league != "All":
            betting = betting[betting["Div"] == league]
        betting_overall = float(betting["Correct"].mean())
        season_bet = (
            betting.groupby("Season")["Correct"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "accuracy", "count": "matches"})
            .reset_index().sort_values("Season")
        )

        # Drift
        val_sorted = val.sort_values("Date").reset_index(drop=True)
        val_sorted["RollingAcc"] = val_sorted["Correct"].rolling(100, min_periods=40).mean()
        drift = [
            {"idx": int(i), "date": d.strftime("%b %Y"), "acc": round(float(a), 4)}
            for i, (d, a) in enumerate(zip(val_sorted["Date"], val_sorted["RollingAcc"]))
            if pd.notna(a)
        ]

        # ROC
        classes    = [0, 1, 2]
        y_true_bin = label_binarize(val["TrueLabel"], classes=classes)
        y_score    = val[[f"Prob_{INV_LABEL_MAP[c]}" for c in classes]].values
        roc: dict = {}
        for i, cls in enumerate(classes):
            name = INV_LABEL_MAP[cls]
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            roc[name] = {
                "fpr":   [round(float(x), 4) for x in fpr],
                "tpr":   [round(float(x), 4) for x in tpr],
                "auc":   round(float(auc(fpr, tpr)), 3),
                "label": OUTCOME_LABELS[name],
            }

        # By outcome
        class_acc  = (
            val.groupby("FTR")["Correct"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "accuracy", "count": "matches"})
            .reset_index()
        )
        by_outcome = {
            r["FTR"]: {"accuracy": round(float(r["accuracy"]), 4), "matches": int(r["matches"])}
            for _, r in class_acc.iterrows()
        }

        # Calibration
        calibration: dict = {}
        for outcome in ["H", "D", "A"]:
            y_true = (val["FTR"] == outcome).astype(int).values
            y_prob = val[f"Prob_{outcome}"].values
            frac, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
            calibration[outcome] = {
                "mean_pred": [round(float(x), 4) for x in mean_pred],
                "fraction":  [round(float(x), 4) for x in frac],
                "label":     OUTCOME_LABELS[outcome],
            }

        # Betting simulation
        sim_df = _get_betting_sim(val)
        if league != "All" and not sim_df.empty:
            sim_df = sim_df[sim_df["Div"] == league].sort_values("Date").reset_index(drop=True)
            if not sim_df.empty:
                sim_df["MatchNum"] = np.arange(1, len(sim_df) + 1)
                sim_df["CumPnL"]   = sim_df["PnL"].cumsum()

        if not sim_df.empty:
            total_pnl = float(sim_df["CumPnL"].iloc[-1])
            betting_sim: dict | None = {
                "total_pnl": round(total_pnl, 2),
                "roi":       round(total_pnl / float(len(sim_df)), 4),
                "win_rate":  round(float(sim_df["Won"].mean()), 4),
                "avg_odds":  round(float(sim_df["BetOdds"].mean()), 2),
                "points":    [
                    {"match": int(r["MatchNum"]), "cum_pnl": round(float(r["CumPnL"]), 2)}
                    for _, r in sim_df.iterrows()
                ],
            }
        else:
            betting_sim = None

        return JSONResponse({
            "metrics": {
                "accuracy":   round(overall_acc, 4),
                "matches":    len(val),
                "seasons":    int(val["Season"].dropna().nunique()),
                "date_range": f"{val['Date'].min().year}–{val['Date'].max().year}",
            },
            "overall_accuracy":  round(overall_acc, 4),
            "betting_overall":   round(betting_overall, 4),
            "season_accuracy":   season_acc.to_dict(orient="records"),
            "betting_baseline":  season_bet.to_dict(orient="records"),
            "drift":             drift,
            "roc":               roc,
            "by_outcome":        by_outcome,
            "calibration":       calibration,
            "betting_sim":       betting_sim,
        })
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/whatsapp/webhook")
def whatsapp_webhook(Body: str = Form(...)) -> Response:
    def _parse(msg: str) -> tuple[str, str]:
        cleaned = msg.strip()
        if not cleaned:
            raise ValueError("Send a message like 'barcelona, madrid'.")
        for sep in [",", " vs ", " v ", " - "]:
            if sep in cleaned.lower():
                if sep.strip() in {",", "-"}:
                    parts = [p.strip() for p in cleaned.split(sep, maxsplit=1)]
                else:
                    parts = [p.strip() for p in re.split(sep, cleaned, maxsplit=1, flags=re.IGNORECASE)]
                if len(parts) == 2 and all(parts):
                    return parts[0], parts[1]
        raise ValueError("Could not parse teams. Use 'home team, away team'.")

    try:
        home, away = _parse(Body)
        reply = predict_match_probability(home, away)
    except ValueError as exc:
        reply = str(exc)
    except Exception:
        reply = "Something went wrong while generating the prediction."

    twiml = f'<?xml version="1.0" encoding="UTF-8"?><Response><Message>{escape(reply)}</Message></Response>'
    return Response(content=twiml, media_type="application/xml")


# ── Static + frontend ─────────────────────────────────────────────────────────

@app.get("/")
def serve_frontend():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return PlainTextResponse("Frontend not available. Build static/ first.", status_code=404)


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static_files")
