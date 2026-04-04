"""Single-function match inference workflow."""

from datetime import datetime
from pathlib import Path

import pandas as pd

from src.data.extract import align_team_names, load_upcoming_fixtures
from src.features.engineering import assign_season, build_features, select_raw_columns
from src.models.predict import load_model, predict


MODEL_PATH = Path(__file__).resolve().parent / "models" / "rf_match_predict_model.pkl"

DATA_SOURCES = {
    "LaLiga_2023_24": "https://www.football-data.co.uk/mmz4281/2324/SP1.csv",
    "LaLiga_2024_25": "https://www.football-data.co.uk/mmz4281/2425/SP1.csv",
    "LaLiga_2025_26": "https://www.football-data.co.uk/mmz4281/2526/SP1.csv",
    "EPL_2023_24": "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
    "EPL_2024_25": "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
    "EPL_2025_26": "https://www.football-data.co.uk/mmz4281/2526/E0.csv",
}


def predict_match_probability(home_team: str, away_team: str) -> str:
    """Run the full workflow for one match and return formatted probabilities."""
    history_frames = [pd.read_csv(url) for url in DATA_SOURCES.values()]
    history = pd.concat(history_frames, ignore_index=True)
    history["Date"] = pd.to_datetime(history["Date"], dayfirst=True, errors="coerce")
    history = select_raw_columns(history)
    history["Season"] = history["Date"].apply(assign_season)

    latest_season_by_div = history.groupby("Div")["Season"].max().to_dict()
    current_teams_by_div = {}
    for div, season in latest_season_by_div.items():
        season_slice = history[(history["Div"] == div) & (history["Season"] == season)]
        current_teams_by_div[div] = (
            set(season_slice["HomeTeam"].dropna()) | set(season_slice["AwayTeam"].dropna())
        )

    valid_names = sorted({team for teams in current_teams_by_div.values() for team in teams})
    requested_match = pd.DataFrame([{"HomeTeam": home_team, "AwayTeam": away_team}])
    aligned_match = align_team_names(requested_match, valid_names)
    matched_home = aligned_match.at[0, "HomeTeam"]
    matched_away = aligned_match.at[0, "AwayTeam"]

    candidate_divisions = [
        div
        for div, teams in current_teams_by_div.items()
        if matched_home in teams and matched_away in teams
    ]
    if len(candidate_divisions) > 1:
        raise ValueError(
            f"Multiple leagues matched for {matched_home} vs {matched_away}: {candidate_divisions}"
        )

    if len(candidate_divisions) == 1:
        match_division = candidate_divisions[0]
    else:
        # Cross-league match (e.g. UCL): teams belong to different divisions.
        # Check both teams are known individually.
        all_known = {team for teams in current_teams_by_div.values() for team in teams}
        unknown = [t for t in (matched_home, matched_away) if t not in all_known]
        if unknown:
            raise ValueError(
                f"Unknown team(s): {unknown}. Check spelling or use a supported team."
            )
        match_division = "UCL"
    match_date = pd.Timestamp(datetime.now().date())
    match_to_score = pd.DataFrame(
        [{
            "Div": match_division,
            "Date": match_date,
            "HomeTeam": matched_home,
            "AwayTeam": matched_away,
        }]
    )

    # Load all upcoming league fixtures so that HomeRank/AwayRank are computed
    # against the full 20-team pool on match_date, not just the 2 teams in this
    # single fixture (which would give artificially low rank values).
    upcoming = load_upcoming_fixtures()
    upcoming = align_team_names(upcoming.copy(), valid_names)
    # Only add fixtures from the relevant division(s); exclude the target match
    # itself since it is already in match_to_score.
    if match_division == "UCL":
        home_div = next(d for d, t in current_teams_by_div.items() if matched_home in t)
        away_div = next(d for d, t in current_teams_by_div.items() if matched_away in t)
        divs_to_pad = {home_div, away_div}
    else:
        divs_to_pad = {match_division}

    padding = upcoming[
        upcoming["Div"].isin(divs_to_pad)
        & ~(
            (upcoming["HomeTeam"] == matched_home)
            & (upcoming["AwayTeam"] == matched_away)
        )
    ].copy()

    combined = pd.concat([history, padding, match_to_score], ignore_index=True)
    engineered = build_features(combined, drop_na=False)
    match_features = engineered[
        (engineered["Div"] == match_division)
        & (engineered["Date"] == match_date)
        & (engineered["HomeTeam"] == matched_home)
        & (engineered["AwayTeam"] == matched_away)
    ].tail(1)
    if match_features.empty:
        raise ValueError("Feature engineering did not produce a row for the requested match.")

    model = load_model(MODEL_PATH)
    probabilities = predict(model, match_features).iloc[0]

    return (
        f"{matched_home}: {float(probabilities['Prob_H']):.4f}, "
        f"ties: {float(probabilities['Prob_D']):.4f}, "
        f"{matched_away}: {float(probabilities['Prob_A']):.4f}"
    )
