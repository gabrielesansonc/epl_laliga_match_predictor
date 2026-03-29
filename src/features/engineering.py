"""
Feature engineering for match prediction.

All transforms respect temporal ordering — rolling windows and cumulative
statistics are always computed over past matches only (shift(1) before
any aggregation) to prevent data leakage.
"""

import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RAW_COLUMNS = [
    "Div", "Season", "Date", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR",
    "HTHG", "HTAG", "HTR",
    "HS", "AS", "HST", "AST",
    "HF", "AF", "HC", "AC",
    "HY", "AY", "HR", "AR",
]

FEATURE_COLUMNS = [
    "RollingPointsDif10",
    "HomeTeamPointsLast2Encounters",
    "HomeRollingPoints38",
    "AwayRollingPoints38",
    "RollingPointsDiff38",
    "HomeRollingShotsTarget20",
    "AwayRollingShotsTarget20",
    "HomeRollingShots20",
    "AwayRollingShots20",
    "HomeRollingYellowCards20",
    "AwayRollingYellowCards20",
    "HomeRollingRedCards20",
    "AwayRollingRedCards20",
    "HomeRollingFouls20",
    "AwayRollingFouls20",
    "HomeRollingCorners20",
    "AwayRollingCorners20",
    "HomeRollingGoals20",
    "AwayRollingGoals20",
    "HomeRollingReceivedGoals20",
    "AwayRollingReceivedGoals20",
    "RollingShotsDif20",
    "RollingShotsTargetDif20",
    "RollingYellowCardsDif20",
    "RollingRedCardsDif20",
    "RollingFoulsDif20",
    "RollingCornersDif20",
    "HomeRank",
    "AwayRank",
    "RankDif",
]

LABEL_MAP = {"H": 0, "D": 1, "A": 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assign_season(date: pd.Timestamp) -> str:
    """Convert a date into a football season string (e.g. '2024/2025').

    Seasons run from August to May. A match in January 2025 belongs to
    the 2024/2025 season.
    """
    year = date.year
    if date.month >= 8:
        return f"{year}/{year + 1}"
    return f"{year - 1}/{year}"


def select_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the 22 columns needed for feature engineering."""
    available = [c for c in RAW_COLUMNS if c in df.columns]
    return df[available].copy()


# ---------------------------------------------------------------------------
# Rolling statistics
# ---------------------------------------------------------------------------

def _add_rolling_stat(
    df: pd.DataFrame,
    home_col: str,
    away_col: str,
    output_name: str,
    window: int,
) -> pd.DataFrame:
    """Compute rolling sums for a home and away stat column.

    Groups by team identity (home or away), shifts by one match to exclude
    the current match, then takes a rolling sum over `window` matches.

    Args:
        df: Match DataFrame sorted by Date.
        home_col: Column name for the home-team stat.
        away_col: Column name for the away-team stat.
        output_name: Base name for the output columns
                     (produces Home<output_name> and Away<output_name>).
        window: Rolling window size (number of past matches).

    Returns:
        DataFrame with two new columns appended.
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df = df.sort_values("Date")

    df[f"Home{output_name}"] = (
        df.groupby("HomeTeam")[home_col]
        .transform(lambda s: s.shift().rolling(window=window, min_periods=1).sum())
    )
    df[f"Away{output_name}"] = (
        df.groupby("AwayTeam")[away_col]
        .transform(lambda s: s.shift().rolling(window=window, min_periods=1).sum())
    )
    return df


# ---------------------------------------------------------------------------
# Head-to-head feature
# ---------------------------------------------------------------------------

def _add_last2_encounters_points(df: pd.DataFrame) -> pd.DataFrame:
    """Add HomeTeamPointsLast2Encounters: points the home team earned in
    the two most recent head-to-head meetings (from the home team's POV).

    Fully vectorised — no row-by-row iteration.
    """
    df = df.sort_values("Date").reset_index(drop=True)

    # Build a long-form table where each row is one team's result in a match,
    # tagged with opponent and date.
    home_side = df[["Date", "HomeTeam", "AwayTeam", "FTR"]].copy()
    home_side.columns = ["Date", "Team", "Opponent", "FTR"]
    home_side["Points"] = home_side["FTR"].map({"H": 3, "D": 1, "A": 0}).fillna(0)
    home_side["IsHome"] = True

    away_side = df[["Date", "AwayTeam", "HomeTeam", "FTR"]].copy()
    away_side.columns = ["Date", "Team", "Opponent", "FTR"]
    away_side["Points"] = away_side["FTR"].map({"A": 3, "D": 1, "H": 0}).fillna(0)
    away_side["IsHome"] = False

    h2h = pd.concat([home_side, away_side], ignore_index=True)
    h2h = h2h.sort_values("Date")

    # Create a canonical pair key so we can group home+away encounters together.
    h2h["Pair"] = h2h.apply(
        lambda r: tuple(sorted([r["Team"], r["Opponent"]])), axis=1
    )

    # For each (Team, Opponent) encounter, accumulate a running sum of the
    # last 2 meetings' points *before* the current date.
    h2h["CumH2HPoints"] = (
        h2h.groupby(["Team", "Opponent"])["Points"]
        .transform(lambda s: s.shift().rolling(window=2, min_periods=1).sum())
        .fillna(0)
    )

    # Pull only the home-team rows and merge back.
    home_h2h = h2h[h2h["IsHome"]].copy()
    home_h2h = home_h2h.rename(columns={
        "Team": "HomeTeam",
        "Opponent": "AwayTeam",
        "CumH2HPoints": "HomeTeamPointsLast2Encounters",
    })[["Date", "HomeTeam", "AwayTeam", "HomeTeamPointsLast2Encounters"]]

    df = df.merge(home_h2h, on=["Date", "HomeTeam", "AwayTeam"], how="left")
    df["HomeTeamPointsLast2Encounters"] = df["HomeTeamPointsLast2Encounters"].fillna(0)
    return df


# ---------------------------------------------------------------------------
# League rankings
# ---------------------------------------------------------------------------

def _add_pre_match_rankings(df: pd.DataFrame) -> pd.DataFrame:
    """Add HomeRank and AwayRank: each team's league position *before* the match.

    Position is derived from cumulative points (shifted to exclude current match)
    ranked within each Div / Season / Date snapshot.
    """
    df = df.sort_values(["Div", "Season", "Date"]).copy()

    home = df[["Div", "Season", "Date", "HomeTeam", "HomeTeamPoints"]].copy()
    home.columns = ["Div", "Season", "Date", "Team", "Points"]
    away = df[["Div", "Season", "Date", "AwayTeam", "AwayTeamPoints"]].copy()
    away.columns = ["Div", "Season", "Date", "Team", "Points"]

    long = pd.concat([home, away], ignore_index=True)
    long = long.sort_values(["Div", "Season", "Team", "Date"])

    long["CumulativePoints"] = (
        long.groupby(["Div", "Season", "Team"])["Points"]
        .cumsum()
        .shift(1)
        .fillna(0)
    )
    long["Rank"] = (
        long.groupby(["Div", "Season", "Date"])["CumulativePoints"]
        .rank(ascending=False, method="min")
    )

    home_rank = long[["Div", "Season", "Date", "Team", "Rank"]].copy()
    home_rank.columns = ["Div", "Season", "Date", "HomeTeam", "HomeRank"]
    away_rank = long[["Div", "Season", "Date", "Team", "Rank"]].copy()
    away_rank.columns = ["Div", "Season", "Date", "AwayTeam", "AwayRank"]

    df = df.merge(home_rank, on=["Div", "Season", "Date", "HomeTeam"], how="left")
    df = df.merge(away_rank, on=["Div", "Season", "Date", "AwayTeam"], how="left")
    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
    """Run the full feature engineering pipeline.

    Steps:
        1. Compute per-match points for home and away teams.
        2. Add rolling windows for all match statistics.
        3. Add head-to-head cumulative points (last 2 meetings).
        4. Compute differential features (home minus away).
        5. Add pre-match league rankings.
        6. Optionally drop rows with NaN features (early-season matches
           that lack enough history for rolling windows).

    Args:
        df: DataFrame containing at least the columns in RAW_COLUMNS.
            FTR must be populated for historical matches; upcoming matches
            may have NaN for FTR and all match stats.
        drop_na: If True, drop rows where any feature column is NaN.
                 Set to False when engineering features for upcoming matches
                 that will be scored (not trained on).

    Returns:
        DataFrame with all FEATURE_COLUMNS appended.
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df["Season"] = df["Date"].apply(assign_season)

    # Points per match (used for rolling and ranking)
    df["HomeTeamPoints"] = df["FTR"].map({"H": 3, "A": 0, "D": 1})
    df["AwayTeamPoints"] = df["FTR"].map({"H": 0, "A": 3, "D": 1})

    # Rolling windows
    df = _add_rolling_stat(df, "HomeTeamPoints", "AwayTeamPoints", "RollingPoints10", 10)
    df = _add_rolling_stat(df, "HomeTeamPoints", "AwayTeamPoints", "RollingPoints38", 38)
    df = _add_rolling_stat(df, "HST",  "AST",  "RollingShotsTarget20", 20)
    df = _add_rolling_stat(df, "HS",   "AS",   "RollingShots20",       20)
    df = _add_rolling_stat(df, "HY",   "AY",   "RollingYellowCards20", 20)
    df = _add_rolling_stat(df, "HR",   "AR",   "RollingRedCards20",    20)
    df = _add_rolling_stat(df, "HF",   "AF",   "RollingFouls20",       20)
    df = _add_rolling_stat(df, "HC",   "AC",   "RollingCorners20",     20)
    df = _add_rolling_stat(df, "FTHG", "FTAG", "RollingGoals20",       20)
    df = _add_rolling_stat(df, "FTAG", "FTHG", "RollingReceivedGoals20", 20)

    # Head-to-head
    df = _add_last2_encounters_points(df)

    # Differentials (home minus away)
    df["RollingPointsDif10"]       = df["HomeRollingPoints10"]       - df["AwayRollingPoints10"]
    df["RollingPointsDiff38"]      = df["HomeRollingPoints38"]       - df["AwayRollingPoints38"]
    df["RollingShotsDif20"]        = df["HomeRollingShots20"]        - df["AwayRollingShots20"]
    df["RollingShotsTargetDif20"]  = df["HomeRollingShotsTarget20"]  - df["AwayRollingShotsTarget20"]
    df["RollingYellowCardsDif20"]  = df["HomeRollingYellowCards20"]  - df["AwayRollingYellowCards20"]
    df["RollingRedCardsDif20"]     = df["HomeRollingRedCards20"]     - df["AwayRollingRedCards20"]
    df["RollingFoulsDif20"]        = df["HomeRollingFouls20"]        - df["AwayRollingFouls20"]
    df["RollingCornersDif20"]      = df["HomeRollingCorners20"]      - df["AwayRollingCorners20"]
    df["RollingGoalsDif20"]        = df["HomeRollingGoals20"]        - df["AwayRollingGoals20"]
    df["RollingReceivedGoalsDif20"] = df["HomeRollingReceivedGoals20"] - df["AwayRollingReceivedGoals20"]

    # Rankings
    df = _add_pre_match_rankings(df)
    df["RankDif"] = df["HomeRank"] - df["AwayRank"]

    if drop_na:
        df = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)

    return df
