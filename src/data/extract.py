"""
Data extraction utilities.

Downloads historical match data from football-data.co.uk and scrapes
upcoming fixtures from TalkSport.
"""

from datetime import datetime, timedelta

import pandas as pd
import requests
from bs4 import BeautifulSoup


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

FIXTURE_PAGES = {
    "SP1": "https://talksport.com/football/primera-division/fixtures",
    "E0": "https://talksport.com/football/premier-league/fixtures",
}

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

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/131.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
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

def _parse_date_text(text: str, reference: datetime) -> str | None:
    """Convert a TalkSport date header like 'Saturday 4th Apr' to 'YYYY-MM-DD'.

    Tries the reference year first; if the resulting date is more than 30 days in
    the past it adds a year (handles fixtures that wrap across a calendar year).
    """
    import re
    # Remove weekday prefix (Monday, Tue, Saturday, …)
    clean = re.sub(r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\w*\s*", "", text, flags=re.IGNORECASE)
    # Remove ordinal suffixes: 4th → 4, 21st → 21
    clean = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", clean).strip()
    for year in (reference.year, reference.year + 1):
        try:
            dt = datetime.strptime(f"{clean} {year}", "%d %b %Y")
            if dt.date() >= (reference - timedelta(days=30)).date():
                return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _parse_fixtures_page(
    soup: BeautifulSoup, reference: datetime
) -> list[tuple[str | None, str, str, bool]]:
    """Extract (date_iso, home, away, has_score) from a TalkSport fixtures page.

    Page structure: all fixture containers are direct children of a single
    ``FixturesAndResultsContainer`` wrapper.  The *first* container of each
    matchday embeds a ``FixturesHeader__title`` element with a human-readable
    date ("Saturday 4th Apr").  Subsequent containers for that day carry no
    header — the last seen date is reused.

    A match is considered upcoming when its ``FixtureDetails__time`` element
    contains a kick-off time (HH:MM); if that element is absent or holds a
    score the match is treated as already played.
    """
    import re

    wrapper = soup.find(True, class_=lambda c: c and "FixturesAndResultsContainer" in c)
    if wrapper is None:
        return []

    results: list[tuple[str | None, str, str, bool]] = []
    seen: set[tuple[str, str]] = set()
    current_date: str | None = None

    for container in wrapper.find_all(
        True,
        class_=lambda c: c and "CompetitionFixtures" in c and "container" in c.lower(),
        recursive=False,
    ):
        # Update current_date whenever this container opens a new matchday
        header_el = container.find(
            True, class_=lambda c: c and "FixturesHeader" in c and "title" in c
        )
        if header_el:
            parsed = _parse_date_text(header_el.text.strip(), reference)
            if parsed:
                current_date = parsed

        # Extract team names
        teams = container.find_all(
            True,
            class_=lambda c: c and "__team" in c and "Wrapper" not in c
                             and "FixtureDetails" in c,
        )
        if len(teams) < 2:
            continue
        home = teams[0].text.strip()
        away = teams[1].text.strip()

        key = (home, away)
        if key in seen:
            continue
        seen.add(key)

        # Determine played vs upcoming via the __time element:
        # upcoming → "14:15" / "19:00"; played → score text or absent
        time_el = container.find(
            True, class_=lambda c: c and "FixtureDetails" in c and "__time" in c
        )
        if time_el:
            time_text = time_el.text.strip()
            has_score = not bool(re.match(r"^\d{1,2}:\d{2}$", time_text))
        else:
            has_score = True  # no time element → treat as already played

        results.append((current_date, home, away, has_score))

    return results


def _scrape_fixtures(
    base_url: str, div_code: str, max_matches: int = 10,
) -> pd.DataFrame:
    """Return a DataFrame of the next ``max_matches`` upcoming fixtures, sorted by date.

    The TalkSport fixtures page lists all upcoming matchdays on a single page,
    so a single fetch is normally sufficient.  A second fetch (next month URL)
    is attempted as a fallback if fewer than ``max_matches`` unplayed matches
    are found.
    """
    now = datetime.now()
    all_upcoming: list[tuple[str, str, str]] = []  # (date_iso, home, away)
    seen: set[tuple[str, str]] = set()

    for month_offset in range(0, 3):
        month = now.month + month_offset
        year = now.year + (month - 1) // 12
        month = (month - 1) % 12 + 1

        url = base_url if month_offset == 0 else f"{base_url}/{year}-{month:02d}"

        resp = requests.get(url, headers=REQUEST_HEADERS, timeout=15)
        if resp.status_code != 200:
            continue

        soup = BeautifulSoup(resp.content, "html.parser")
        for date_iso, home, away, has_score in _parse_fixtures_page(soup, now):
            if has_score:
                continue
            key = (home, away)
            if key in seen:
                continue
            seen.add(key)
            all_upcoming.append((date_iso or now.strftime("%Y-%m-%d"), home, away))

        if len(all_upcoming) >= max_matches:
            break

    if not all_upcoming:
        return pd.DataFrame(columns=["Div", "Date", "HomeTeam", "AwayTeam"])

    # Sort chronologically and keep only the next max_matches
    all_upcoming.sort(key=lambda x: x[0])
    all_upcoming = all_upcoming[:max_matches]

    rows = [
        {
            "Div": div_code,
            "Date": datetime.fromisoformat(d).strftime("%d/%m/%Y"),
            "HomeTeam": home,
            "AwayTeam": away,
        }
        for d, home, away in all_upcoming
    ]
    return pd.DataFrame(rows)


def load_upcoming_fixtures(fixture_pages: dict[str, str] = None) -> pd.DataFrame:
    """Scrape upcoming fixtures from TalkSport.

    Returns a minimal DataFrame with Div, Date, HomeTeam, AwayTeam columns
    (match stats are left as NaN; they get filled during feature engineering).

    Args:
        fixture_pages: Mapping of league code → URL. Defaults to FIXTURE_PAGES.

    Returns:
        DataFrame of upcoming matches.
    """
    if fixture_pages is None:
        fixture_pages = FIXTURE_PAGES

    frames = [_scrape_fixtures(url, code) for code, url in fixture_pages.items()]
    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    return df


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
