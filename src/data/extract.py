"""
Data extraction utilities.

Historical match data comes from football-data.co.uk. Upcoming fixtures come
from the openfootball/football.json full-season schedules.

NOTE (Sep 2026): fixtures have now moved sources twice.

  1. Originally scraped from TalkSport's per-league fixtures pages. Those
     pages were removed from talksport.com — even TalkSport's own internal
     "See all" links 301-redirect into a dead URL — so the scrape returned
     an empty list forever and /api/fixtures silently showed nothing.
  2. Then football-data.co.uk's fixtures.csv. Correct, but it is a rolling
     ~7-day feed: it answers "what is on this week", not "what is next".
     During an international break it legitimately returns almost nothing,
     so the page looked broken even though it was working.

Neither matched the original intent, which was "the next N fixtures,
whenever they fall". openfootball publishes the complete season schedule
(all 380 matches) as JSON on raw.githubusercontent.com, so the next N are
always available regardless of breaks.

Caveats worth knowing:

  * openfootball's ``score`` field lags real results by over a week, so it
    is NOT used to decide whether a match has been played. Fixtures are
    filtered on kickoff time against the current UTC instant instead, which
    stays correct however stale the results tracking is.
  * Schedules are revised (postponements, TV moves) and a cached upstream
    file will not reflect that immediately.
"""

import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from functools import lru_cache
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

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

# openfootball publishes one JSON file per league per season, containing the
# complete fixture list for that season.
OPENFOOTBALL_BASE = "https://raw.githubusercontent.com/openfootball/football.json/master"
OPENFOOTBALL_FILES = {"E0": "en.1.json", "SP1": "es.1.json"}

# How many upcoming fixtures to return per league by default. A full EPL /
# La Liga matchday is exactly 10 matches (20 teams), so this is one round.
DEFAULT_FIXTURE_LIMIT = 10

# Kickoff times in openfootball are league-local, not UTC. Stored as names and
# resolved lazily: ZoneInfo reads the IANA database from the operating system,
# which slim container images routinely ship without. Building the ZoneInfo
# objects at import time meant a missing tz database raised
# ZoneInfoNotFoundError while this module was still being imported, taking down
# the whole API — including every endpoint that has nothing to do with
# kickoff times. requirements.txt now pins tzdata so the data is always
# present; this resolver is the belt-and-braces half.
LEAGUE_TIMEZONE_NAMES = {"E0": "Europe/London", "SP1": "Europe/Madrid"}


@lru_cache(maxsize=None)
def league_timezone(league_code: str):
    """Return the tzinfo for a league's local kickoff times.

    Falls back to UTC if the platform has no IANA time zone database. That
    shifts displayed kickoff times by at most an hour or two rather than
    failing the request, which is the right trade for a fixture list.
    """
    name = LEAGUE_TIMEZONE_NAMES.get(league_code)
    if name is None:
        return timezone.utc
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, KeyError, ValueError):
        print(f"  No time zone data for {name}; falling back to UTC.")
        return timezone.utc

# openfootball uses full club names ("Manchester City FC"); the model is
# trained on football-data.co.uk's short names ("Man City"). These are mapped
# explicitly rather than fuzzy-matched: fuzzy matching scores
# "Coventry City FC" and "Hull City AFC" onto "Man City" at 86, and
# "RCD Espanyol de Barcelona" onto "Barcelona" at 90 — confidently wrong
# rather than merely unmatched.
OPENFOOTBALL_TEAM_NAMES = {
    # Premier League
    "AFC Bournemouth": "Bournemouth",
    "Arsenal FC": "Arsenal",
    "Aston Villa FC": "Aston Villa",
    "Brentford FC": "Brentford",
    "Brighton & Hove Albion FC": "Brighton",
    "Burnley FC": "Burnley",
    "Chelsea FC": "Chelsea",
    "Coventry City FC": "Coventry",
    "Crystal Palace FC": "Crystal Palace",
    "Everton FC": "Everton",
    "Fulham FC": "Fulham",
    "Hull City AFC": "Hull",
    "Ipswich Town FC": "Ipswich",
    "Leeds United FC": "Leeds",
    "Leicester City FC": "Leicester",
    "Liverpool FC": "Liverpool",
    "Manchester City FC": "Man City",
    "Manchester United FC": "Man United",
    "Newcastle United FC": "Newcastle",
    "Nottingham Forest FC": "Nott'm Forest",
    "Sheffield United FC": "Sheffield United",
    "Southampton FC": "Southampton",
    "Sunderland AFC": "Sunderland",
    "Tottenham Hotspur FC": "Tottenham",
    "West Ham United FC": "West Ham",
    "Wolverhampton Wanderers FC": "Wolves",
    # La Liga
    "Athletic Club": "Ath Bilbao",
    "CA Osasuna": "Osasuna",
    "CD Leganés": "Leganes",
    "Club Atlético de Madrid": "Ath Madrid",
    "Deportivo Alavés": "Alaves",
    "Elche CF": "Elche",
    "FC Barcelona": "Barcelona",
    "Getafe CF": "Getafe",
    "Girona FC": "Girona",
    "Granada CF": "Granada",
    "Levante UD": "Levante",
    "Málaga CF": "Malaga",
    "RC Celta de Vigo": "Celta",
    "RC Deportivo La Coruña": "La Coruna",
    "RCD Espanyol de Barcelona": "Espanol",
    "RCD Mallorca": "Mallorca",
    "Rayo Vallecano de Madrid": "Vallecano",
    "Real Betis Balompié": "Betis",
    "Real Madrid CF": "Real Madrid",
    "Real Racing Club de Santander": "Santander",
    "Real Sociedad de Fútbol": "Sociedad",
    "Real Valladolid CF": "Valladolid",
    "Sevilla FC": "Sevilla",
    "UD Almería": "Almeria",
    "UD Las Palmas": "Las Palmas",
    "Valencia CF": "Valencia",
    "Villarreal CF": "Villarreal",
}

TEAM_NAME_FIXES = {
    # La Liga
    "Athletic Club": "Ath Bilbao",
    "Athletic": "Ath Bilbao",
    "Atlético": "Ath Madrid",
    "Atletico Madrid": "Ath Madrid",
    "Atletico": "Ath Madrid",
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

def _season_slug(today: datetime = None) -> str:
    """Return the openfootball season directory for a date, e.g. "2026-27".

    European seasons run August-May, so anything from July onwards belongs to
    the season starting that calendar year.
    """
    today = today or datetime.now(timezone.utc)
    start = today.year if today.month >= 7 else today.year - 1
    return f"{start}-{str(start + 1)[-2:]}"


def _fetch_season_schedule(league_code: str, season: str) -> list[dict]:
    """Download one league-season schedule from openfootball.

    Returns the raw ``matches`` list, or an empty list if that season file
    does not exist yet (openfootball publishes a new season shortly before
    it starts, so the current slug can 404 during the summer).
    """
    filename = OPENFOOTBALL_FILES.get(league_code)
    if filename is None:
        return []
    url = f"{OPENFOOTBALL_BASE}/{season}/{filename}"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return json.loads(resp.read()).get("matches", [])
    except (urllib.error.HTTPError, urllib.error.URLError, ValueError) as e:
        print(f"  Could not load {league_code} schedule for {season}: {e}")
        return []


def _kickoff_utc(date_str: str, time_str: str, tz: ZoneInfo) -> datetime | None:
    """Combine an ISO 'YYYY-MM-DD' date and 'HH:MM' kickoff (league-local)
    into an aware UTC datetime. Returns None if the date is unparsable.

    A missing or malformed time is treated as an end-of-day kickoff so that a
    match is not dropped from its own matchday just because openfootball has
    not published the exact time yet.
    """
    if not isinstance(date_str, str):
        return None
    try:
        day = datetime.strptime(date_str.strip(), "%Y-%m-%d")
    except ValueError:
        return None
    hour, minute = 23, 59
    if isinstance(time_str, str) and time_str.strip():
        try:
            parsed = datetime.strptime(time_str.strip(), "%H:%M")
            hour, minute = parsed.hour, parsed.minute
        except ValueError:
            pass
    local = day.replace(hour=hour, minute=minute, tzinfo=tz)
    return local.astimezone(timezone.utc)


def load_upcoming_fixtures(
    league_codes: list[str] = None,
    limit: int = DEFAULT_FIXTURE_LIMIT,
    season: str = None,
) -> pd.DataFrame:
    """Load the next ``limit`` unplayed fixtures per league.

    Reads the complete season schedule from openfootball rather than a rolling
    weekly feed, so the next fixtures are always available even during an
    international break.

    Whether a match has been played is decided by comparing its kickoff to the
    current UTC time. openfootball's own ``score`` field is deliberately
    ignored: it lags real results by over a week, so trusting it would surface
    already-played matches as upcoming.

    Args:
        league_codes: League codes to load. Defaults to LEAGUE_CODES (E0, SP1).
        limit: Maximum fixtures to return per league. None/0 returns all
            remaining fixtures in the season.
        season: openfootball season slug (e.g. "2026-27"). Defaults to the
            current season, falling back to the previous one if unpublished.

    Returns:
        DataFrame with Div, Date, HomeTeam, AwayTeam columns, sorted by date
        (match stats are left absent; they get filled during feature
        engineering). Team names are mapped to football-data.co.uk's short
        names so they line up with the historical data.
    """
    if league_codes is None:
        league_codes = LEAGUE_CODES

    now_utc = datetime.now(timezone.utc)
    seasons = [season] if season else [_season_slug(now_utc), _season_slug(now_utc.replace(year=now_utc.year - 1))]

    rows: list[dict] = []
    for code in league_codes:
        tz = league_timezone(code)

        matches: list[dict] = []
        for candidate in seasons:
            matches = _fetch_season_schedule(code, candidate)
            if matches:
                break

        upcoming = []
        for match in matches:
            kickoff = _kickoff_utc(match.get("date"), match.get("time"), tz)
            if kickoff is None or kickoff <= now_utc:
                continue
            home = match.get("team1")
            away = match.get("team2")
            if not home or not away:
                continue
            upcoming.append({
                "Div": code,
                "Date": pd.Timestamp(kickoff.astimezone(tz).date()),
                "HomeTeam": OPENFOOTBALL_TEAM_NAMES.get(home, home),
                "AwayTeam": OPENFOOTBALL_TEAM_NAMES.get(away, away),
                "_kickoff": kickoff,
            })

        upcoming.sort(key=lambda r: r["_kickoff"])
        if limit:
            upcoming = upcoming[:limit]
        rows.extend(upcoming)

    columns = ["Div", "Date", "HomeTeam", "AwayTeam"]
    if not rows:
        # Always hand back the full column set, even with nothing to show:
        # callers index these columns directly and would raise a KeyError.
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(rows).sort_values("_kickoff").reset_index(drop=True)
    return df[columns]


def align_team_names(
    upcoming: pd.DataFrame,
    valid_names: list[str],
    manual_fixes: dict[str, str] = None,
    min_score: int = 85,
) -> pd.DataFrame:
    """Standardise fixture team names against the historical team name list.

    Applies explicit substitutions first, then fuzzy-matches any remaining
    names that don't appear in the reference list.

    A fuzzy match is only accepted at or above ``min_score``. Below that the
    original name is kept unchanged. This matters because the reference list
    is built from recent seasons only, so a newly promoted club is absent
    from it — and an unguarded fuzzy match sends "Sunderland" to "Man City"
    (48) or "Malaga" to "Getafe" (43). Keeping the real name yields a fixture
    with no rolling history (blank features) instead of one wearing another
    club's form, which is wrong but far less visibly so.

    Args:
        upcoming: DataFrame with HomeTeam / AwayTeam columns.
        valid_names: Reference list of canonical team names.
        manual_fixes: Optional extra name substitutions.
        min_score: Minimum fuzzy score to accept a match (0-100).

    Returns:
        DataFrame with corrected team names.
    """
    from fuzzywuzzy import process

    fixes = {**OPENFOOTBALL_TEAM_NAMES, **TEAM_NAME_FIXES, **(manual_fixes or {})}
    upcoming = upcoming.copy()
    upcoming["HomeTeam"] = upcoming["HomeTeam"].replace(fixes)
    upcoming["AwayTeam"] = upcoming["AwayTeam"].replace(fixes)

    valid_set = set(valid_names)

    def best_match(name: str) -> str:
        if name in valid_set or not valid_names:
            return name
        match, score = process.extractOne(name, valid_names)
        return match if score >= min_score else name

    upcoming["HomeTeam"] = upcoming["HomeTeam"].apply(best_match)
    upcoming["AwayTeam"] = upcoming["AwayTeam"].apply(best_match)
    return upcoming
