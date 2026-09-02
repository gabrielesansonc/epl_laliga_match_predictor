"""
Data extraction utilities.

Downloads historical match data and upcoming fixtures from
football-data.co.uk.

NOTE (Sep 2026): Upcoming fixtures previously came from scraping TalkSport's
per-league fixtures pages (talksport.com/football/{league}/fixtures). Those
pages were removed from talksport.com's own site — even TalkSport's internal
"See all" links now 301-redirect into a dead URL — so that source no longer
works and can't be patched around. load_upcoming_fixtures() now reads
football-data.co.uk's own rolling fixtures.csv feed instead: same trusted
domain already used for historical results, structured CSV instead of
scraped/versioned HTML, so it isn't exposed to the next frontend redesign.
One tradeoff: fixtures.csv only carries a short rolling window (roughly the
next few days across all leagues), so during international breaks a league
can legitimately show zero upcoming fixtures for a bit — that's the real
fixture calendar, not a bug.
"""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LEAGUE_CODES = ["E0", "SP1"]

RAW_COLUMNS = [
    "Div", "Season", "Date", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR",
    "HTHG", "HTAG", "HTR",
    "HS", "AS", "HST", "AST",
    "HF", "AF", "HC", "AC",
    "HY", "AY", "HR", "AR",
]

FIXTURES_CSV_URL = "https://www.football-data.co.uk/fixtures.csv"

TEAM_NAME_FIXES = {
    # La Liga
    "Athletic Club": "Ath Bilbao",
    "Athletic": "Ath Bilbao",
    "Atlético": "Atl Madrid",
    "Atletico Madrid": "Atl Madrid",
    "Atletico": "Atl Madrid",
    "Rayo": "Vallecano",
    "Rayo Vallecano": "Vallecano",
    "Espanyol": "Espanol",
    "Real Sociedad": "Sociedad",
    "Alavés": "Alaves",
    # EPL
    "Spurs": "Tottenham",
    "Man Utd": "Man United",
    "Nottm Forest": "Nott'm Forest",
}


# ---------------------------------------------------------------------------
# Historical data
# ---------------------------------------------------------------------------

def _season_url(season_start: int, league_code: str) -> str:
    yy_start = str(season_start)[-2:]
    yy_end = str(season_start + 1)[-2:]
    return f"https://www.football-data.co.uk/mmz4281/{yy_start}{yy_end}/{league_code}.csv"


def load_historical_data(
    start_season: int = 2005,
    end_season: int = 2024,
    league_codes: list[str] = None,
) -> pd.DataFrame:
    """Download and concatenate historical match data from football-data.co.uk.

    Args:
        start_season: First season start year (e.g. 2005 for 2005/2006).
        end_season: Last season start year (inclusive).
        league_codes: List of league codes to download (default: E0, SP1).

    Returns:
        Raw DataFrame with all matches concatenated.
    """
    if league_codes is None:
        league_codes = LEAGUE_CODES

    frames = []
    for season in range(start_season, end_season + 1):
        for code in league_codes:
            url = _season_url(season, code)
            try:
                df = pd.read_csv(url)
                df["Season"] = f"{season}/{season + 1}"
                frames.append(df)
                print(f"  Loaded {season}/{season + 1} {code} ({len(df)} rows)")
            except Exception as e:
                print(f"  Skipped {season}/{season + 1} {code}: {e}")

    combined = pd.concat(frames, ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"], dayfirst=True, errors="coerce")
    combined = combined.sort_values("Date").reset_index(drop=True)
    print(f"\nTotal: {len(combined)} matches loaded.")
    return combined


# ---------------------------------------------------------------------------
# Upcoming fixtures
# ---------------------------------------------------------------------------

# football-data.co.uk publishes Date/Time in UK local time.
_UK_TZ = ZoneInfo("Europe/London")


def _kickoff_utc(date_str: str, time_str: str) -> datetime | None:
    """Combine a 'DD/MM/YYYY' date and 'HH:MM' kickoff time (UK local) into
    an aware UTC datetime. Returns None if either field is missing/unparsable.
    """
    if not isinstance(date_str, str) or not isinstance(time_str, str):
        return None
    try:
        naive = datetime.strptime(f"{date_str.strip()} {time_str.strip()}", "%d/%m/%Y %H:%M")
    except ValueError:
        return None
    return naive.replace(tzinfo=_UK_TZ).astimezone(timezone.utc)


def load_upcoming_fixtures(league_codes: list[str] = None) -> pd.DataFrame:
    """Load upcoming, not-yet-played fixtures from football-data.co.uk.

    football-data.co.uk publishes a rolling ``fixtures.csv`` covering all
    leagues for the current window (typically the next several days). It's
    the same trusted domain already used for historical results, and a
    structured CSV instead of scraped/versioned HTML.

    The feed isn't guaranteed to drop a row the instant a match kicks off,
    so rows are filtered against the current UTC time (using each match's
    UK-local kickoff) to make sure only genuinely unplayed fixtures are
    returned. Rows with no parsable kickoff time are kept only if their
    date is still today or later, since we can't otherwise confirm they
    haven't already been played.

    Args:
        league_codes: League codes to keep. Defaults to LEAGUE_CODES (E0, SP1).

    Returns:
        DataFrame with Div, Date, HomeTeam, AwayTeam columns (match stats
        are left as NaN; they get filled during feature engineering).
    """
    if league_codes is None:
        league_codes = LEAGUE_CODES

    raw = pd.read_csv(FIXTURES_CSV_URL, encoding="utf-8-sig")
    raw = raw[raw["Div"].isin(league_codes)].copy()

    now_utc = datetime.now(timezone.utc)
    today_uk = datetime.now(_UK_TZ).date()

    keep = []
    for _, row in raw.iterrows():
        kickoff = _kickoff_utc(row.get("Date"), row.get("Time"))
        if kickoff is not None:
            keep.append(kickoff > now_utc)
            continue
        # No usable kickoff time — fall back to a date-only check so we
        # don't accidentally keep something clearly in the past.
        date_only = pd.to_datetime(row.get("Date"), dayfirst=True, errors="coerce")
        keep.append(bool(pd.notna(date_only) and date_only.date() >= today_uk))

    # Build an explicit boolean Series (not a plain list) for the mask: when
    # a league has zero rows (e.g. E0 during an international break), an
    # empty Python list is ambiguous to pandas' __getitem__ and gets read as
    # "select these zero columns" rather than "keep these zero rows", which
    # would silently drop every column instead of just filtering to nothing.
    keep_mask = pd.Series(keep, index=raw.index, dtype=bool)
    upcoming = raw.loc[keep_mask, ["Div", "Date", "HomeTeam", "AwayTeam"]].reset_index(drop=True)
    upcoming["Date"] = pd.to_datetime(upcoming["Date"], dayfirst=True, errors="coerce")
    upcoming = upcoming.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return upcoming


def align_team_names(
    upcoming: pd.DataFrame,
    valid_names: list[str],
    manual_fixes: dict[str, str] = None,
) -> pd.DataFrame:
    """Standardise scraped team names against the historical team name list.

    Applies manual substitutions first, then fuzzy-matches any remaining
    names that don't appear in the reference list.

    Args:
        upcoming: DataFrame with HomeTeam / AwayTeam columns.
        valid_names: Reference list of canonical team names.
        manual_fixes: Optional extra name substitutions.

    Returns:
        DataFrame with corrected team names.
    """
    from fuzzywuzzy import process

    fixes = {**TEAM_NAME_FIXES, **(manual_fixes or {})}
    upcoming = upcoming.copy()
    upcoming["HomeTeam"] = upcoming["HomeTeam"].replace(fixes)
    upcoming["AwayTeam"] = upcoming["AwayTeam"].replace(fixes)

    valid_set = set(valid_names)

    def best_match(name: str) -> str:
        if name in valid_set:
            return name
        match, _ = process.extractOne(name, valid_names)
        return match

    upcoming["HomeTeam"] = upcoming["HomeTeam"].apply(best_match)
    upcoming["AwayTeam"] = upcoming["AwayTeam"].apply(best_match)
    return upcoming
