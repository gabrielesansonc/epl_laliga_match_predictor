"""Streamlit dashboard — upcoming match outcome predictions."""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.extract import (
    align_team_names,
    load_upcoming_fixtures,
    RAW_COLUMNS,
)
from src.features.engineering import (
    select_raw_columns,
    build_features,
    FEATURE_COLUMNS,
    INV_LABEL_MAP,
)
from src.models.predict import load_model, predict

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------

MODEL_PATH = Path(__file__).resolve().parent / "models" / "rf_match_predict_model.pkl"

DATA_SOURCES = {
    "LaLiga_2023_24": "https://www.football-data.co.uk/mmz4281/2324/SP1.csv",
    "LaLiga_2024_25": "https://www.football-data.co.uk/mmz4281/2425/SP1.csv",
    "LaLiga_2025_26": "https://www.football-data.co.uk/mmz4281/2526/SP1.csv",
    "EPL_2023_24":    "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
    "EPL_2024_25":    "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
    "EPL_2025_26":    "https://www.football-data.co.uk/mmz4281/2526/E0.csv",
}

LEAGUE_LABELS = {"SP1": "La Liga", "E0": "Premier League"}


# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_match_history() -> pd.DataFrame:
    """Download and merge recent historical seasons for rolling-window context."""
    frames = [pd.read_csv(url) for url in DATA_SOURCES.values()]
    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    return select_raw_columns(df)


@st.cache_data(show_spinner=False)
def load_fixtures() -> pd.DataFrame:
    return load_upcoming_fixtures()


@st.cache_resource(show_spinner=False)
def load_cached_model():
    return load_model(MODEL_PATH)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def build_predictions() -> pd.DataFrame:
    history = load_match_history()
    upcoming = load_fixtures()

    valid_names = history["HomeTeam"].dropna().unique().tolist()
    upcoming = align_team_names(upcoming.copy(), valid_names)
    upcoming_keys = upcoming[["Div", "Date", "HomeTeam", "AwayTeam"]].copy()

    combined = pd.concat([history, upcoming], ignore_index=True)
    combined = build_features(combined, drop_na=False)

    upcoming_full = combined.merge(
        upcoming_keys.assign(IsUpcoming=True),
        on=["Div", "Date", "HomeTeam", "AwayTeam"],
        how="inner",
    )

    model = load_cached_model()
    preds = predict(model, upcoming_full)

    output = pd.concat(
        [upcoming_full[["Div", "Date", "HomeTeam", "AwayTeam"]], preds],
        axis=1,
    ).sort_values("Date").reset_index(drop=True)
    return output


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _donut_chart(prob_h: float, prob_d: float, prob_a: float) -> go.Figure:
    fig = go.Figure(data=[go.Pie(
        labels=["Away Win", "Draw", "Home Win"],
        values=[prob_a, prob_d, prob_h],
        hole=0.4,
        marker_colors=["#FF6F6F", "#808080", "#6F7BFF"],
        textinfo="text+percent",
        textposition="outside",
        insidetextorientation="radial",
        direction="clockwise",
        sort=False,
    )])
    fig.update_traces(
        hoverinfo="label+percent",
        textfont_size=12,
        marker=dict(line=dict(color="#000000", width=0.4)),
    )
    fig.update_layout(width=325, height=350, legend=dict(font=dict(size=10)))
    return fig


def _render_league(df: pd.DataFrame, div: str) -> None:
    label = LEAGUE_LABELS.get(div, div)
    st.title(f"{label} Predictions")

    league_df = df[df["Div"] == div]
    if league_df.empty:
        st.info("No upcoming fixtures found.")
        return

    for _, row in league_df.iterrows():
        st.markdown(
            f"<span style='color:#6F7BFF;font-weight:bold'>{row['HomeTeam']}</span>"
            f" vs "
            f"<span style='color:#FF6F6F;font-weight:bold'>{row['AwayTeam']}</span>",
            unsafe_allow_html=True,
        )
        fig = _donut_chart(row["Prob_H"], row["Prob_D"], row["Prob_A"])
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Match Outcome Predictions",
        page_icon="⚽",
        layout="wide",
    )

    with st.spinner("Loading fixtures and scoring matches..."):
        predictions = build_predictions()

    for div in ["SP1", "E0"]:
        _render_league(predictions, div)


if __name__ == "__main__":
    main()
