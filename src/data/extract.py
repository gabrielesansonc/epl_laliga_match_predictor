"""
Data extraction utilities.

Downloads historical match data from football-data.co.uk and scrapes
upcoming fixtures from TalkSport.
"""

from datetime import datetime

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

def _parse_fixtures_page(soup: BeautifulSoup) -> list[tuple[str, str, bool]]:
    """Extract (home, away, has_score) tuples from a TalkSport fixtures page."""
    containers = soup.find_all(
        True,
        class_=lambda c: c and "CompetitionFixtures" in c and "container" in c,
    )
    results = []
    for container in containers:
        teams = container.find_all(
            True,
            class_=lambda c: c and "__team" in c and "Wrapper" not in c
                             and "FixtureDetails" in c,
        )
        if len(teams) < 2:
            continue
        home = teams[0].text.strip()
        away = teams[1].text.strip()
        score_el = container.find(
            True, class_=lambda c: c and "Score" in c and "score" in c,
        )
        has_score = bool(score_el and score_el.text.strip())
        results.append((home, away, has_score))
    return results


def _scrape_fixtures(
    base_url: str, div_code: str, max_matches: int = 10,
) -> pd.DataFrame:
    """Scrape upcoming (unplayed) fixtures from TalkSport.

    Checks the current month first; if all matches are played, advances
    to the next month to find the next matchday.
    """
    now = datetime.now()

    for month_offset in range(0, 3):
        # Build URL: base for current month, base/YYYY-MM for future months
        target = datetime(now.year, now.month, 1)
        month = target.month + month_offset
        year = target.year + (month - 1) // 12
        month = (month - 1) % 12 + 1

        if month_offset == 0:
            url = base_url
        else:
            url = f"{base_url}/{year}-{month:02d}"

        resp = requests.get(url, headers=REQUEST_HEADERS, timeout=15)
        if resp.status_code != 200:
            continue

        soup = BeautifulSoup(resp.content, "html.parser")
        fixtures = _parse_fixtures_page(soup)

        # Keep only unplayed fixtures
        upcoming = [(h, a) for h, a, has_score in fixtures if not has_score]

        if upcoming:
            home = [h for h, _ in upcoming[:max_matches]]
            away = [a for _, a in upcoming[:max_matches]]
            today = now.strftime("%d/%m/%Y")
            return pd.DataFrame({
                "Div": div_code,
                "Date": today,
                "HomeTeam": home,
                "AwayTeam": away,
            })

    # Fallback: return empty if no upcoming fixtures found
    return pd.DataFrame(columns=["Div", "Date", "HomeTeam", "AwayTeam"])


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
