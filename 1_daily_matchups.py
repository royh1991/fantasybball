import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import os


df = pd.read_csv("data/static/nba_schedule.csv")
df['Date'] = df['Date'].str.replace(r'([ap])$', '', regex=True)
df['Date'] = pd.to_datetime(df['Date'], format='%a %b %d %Y')

# Constants
TODAY_DT = datetime.today().strftime('%Y-%m-%d')
league_id = '40204'  # Example league ID
season_id = 2025

# API Headers and Cookies
headers = {
    "User-Agent": "Mozilla/5.0",
    "x-fantasy-filter": "{}"
}
cookies = {
    'swid': '{DCE1F96F-85B2-451D-A1F9-6F85B2D51DCD}',
    'espn_s2': '{AECfCrJpGmvxK%2BEefmEJFFV2DjG45aVHS7Iiag58LLpJtF8ILBTMjaBtkyUXFAwjIIn9NLKqORc4%2FDFs9LjqNl3jh9MJ%2BaKoPjjKWiw9sHzub3AoJkVDBsGa8d9M7tAhQOZtEHZKT42TdKMJEtWr%2F5vlurNDR7ET%2FJ5RbVrSQS2PIjs1dpN2y6k4ZFlZ0%2B0EWp22AuU%2BECtPQeU8LieW9WHthZZlwPGnSjjR9UbS%2Bo7Ep%2FAjJOI4CRshu20fk1j8EvHJVwsB35F%2FQ7wHvshRP4oN}'
}


# Create the week mapping with date ranges
week_mapping = [
    ("2024-10-22", "2024-10-27", 'Matchup 1'),
    ("2024-10-28", "2024-11-03", 'Week 2'),
    ("2024-11-04", "2024-11-10", 'Week 3'),
    ("2024-11-11", "2024-11-17", 'Week 4'),
    ("2024-11-18", "2024-11-24", 'Week 5'),
    ("2024-11-25", "2024-12-01", 'Week 6'),
    ("2024-12-02", "2024-12-08", 'Week 7'),
    ("2024-12-09", "2024-12-15", 'Week 8'),
    ("2024-12-16", "2024-12-22", 'Week 9'),
    ("2024-12-23", "2024-12-29", 'Week 10'),
    ("2024-12-30", "2025-01-05", 'Week 11'),
    ("2025-01-06", "2025-01-12", 'Week 12'),
    ("2025-01-13", "2025-01-19", 'Week 13'),
    ("2025-01-20", "2025-01-26", 'Week 14'),
    ("2025-01-27", "2025-02-02", 'Week 15'),
    ("2025-02-03", "2025-02-09", 'Week 16'),
    ("2025-02-10", "2025-03-02", "Playoff Round 1"),
    ("2025-03-03", "2025-03-16", "Playoff Round 2"),
    ("2025-03-17", "2025-03-30", "Playoff Round 3")
]

# Function to assign week and start/end dates
def assign_week_info(game_date):
    for start, end, week in week_mapping:
        if pd.to_datetime(start) <= game_date <= pd.to_datetime(end):
            return pd.Series([week, pd.to_datetime(start), pd.to_datetime(end)])
    return pd.Series(["No Match", pd.NaT, pd.NaT])

def extract_game_start_date(matchup_period_id):
    """Convert matchup period ID to a readable date."""
    season_start_date = datetime(2025, 10, 15)  # Example NBA season start date
    return (season_start_date + timedelta(days=matchup_period_id - 1)).strftime('%Y-%m-%d')

def get_matchup_data(league_id, season_id):
    """Fetch matchup data for the given league and season."""
    url = f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/seasons/{season_id}/segments/0/leagues/{league_id}?view=mMatchup"
    response = requests.get(url, headers=headers, cookies=cookies)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")
        return None

def extract_matchup_details(matchup_data):
    """Extract home team, away team, player details, and game start date."""
    matchups = []

    for game in matchup_data.get('schedule', []):
        home_team_id = game.get('home', {}).get('teamId', 'Unknown')
        away_team_id = game.get('away', {}).get('teamId', 'Unknown')
        matchup_period_id = game.get('matchupPeriodId', 1)
        game_start_date = extract_game_start_date(matchup_period_id)

        # Process both home and away rosters
        for team_key in ['home', 'away']:
            team = game.get(team_key, {})
            team_id = team.get('teamId', 'Unknown')
            team_role = 'Home' if team_key == 'home' else 'Away'  # Identify if it's the home or away team

            for entry in team.get('rosterForCurrentScoringPeriod', {}).get('entries', []):
                player = entry.get('playerPoolEntry', {}).get('player', {})
                player_name = player.get('fullName', 'Unknown Player')
                pro_team_id = player.get('proTeamId', 'Unknown')  # Get the player's pro team ID
                injured = player.get('injured', False)  # Get injury status
                injury_status = player.get('injuryStatus', 'Unknown')  # Get detailed injury status

                # Add the player's matchup information
                matchups.append({
                    'Team Role': team_role,
                    'Team ID': team_id,
                    'Opponent ID': away_team_id if team_role == 'Home' else home_team_id,
                    'Player Name': player_name,
                    'Pro Team ID': pro_team_id,
                    'Injured': injured,
                    'Injury Status': injury_status,
                    'Game Start Date': game_start_date
                })

    return pd.DataFrame(matchups)

def main():
    matchup_data = get_matchup_data(league_id, season_id)
    # Extract matchup details
    df_matchups = extract_matchup_details(matchup_data)

    pro_team_mapping = {
        1: "Atlanta Hawks", 
        2: "Boston Celtics", 
        3: "New Orleans Pelicans",  # Corrected
        4: "Chicago Bulls", 
        5: "Cleveland Cavaliers", 
        6: "Dallas Mavericks", 
        7: "Denver Nuggets", 
        8: "Detroit Pistons", 
        9: "Golden State Warriors", 
        10: "Houston Rockets", 
        11: "Indiana Pacers", 
        12: "Los Angeles Clippers", 
        13: "Los Angeles Lakers", 
        14: "Miami Heat", 
        15: "Milwaukee Bucks", 
        16: "Minnesota Timberwolves", 
        17: "Brooklyn Nets",  # Corrected
        18: "New York Knicks", 
        19: "Orlando Magic", 
        20: "Philadelphia 76ers", 
        21: "Phoenix Suns", 
        22: "Portland Trail Blazers", 
        23: "Sacramento Kings", 
        24: "San Antonio Spurs", 
        25: "Oklahoma City Thunder", 
        26: "Utah Jazz", 
        27: "Toronto Raptors", 
        28: "Washington Wizards", 
        29: "Memphis Grizzlies", 
        30: "Charlotte Hornets"
    }
    # Update Pro Team Name in the DataFrame
    df_matchups['Pro Team'] = df_matchups['Pro Team ID'].map(pro_team_mapping).fillna('Unknown')
    df_matchups = df_matchups.drop(columns=['Pro Team ID'])  # Optional: Remove ID column
    del df_matchups['Game Start Date'] 
    # Perform the first merge based on the home team
    # Merge on home team
    home_matches = pd.merge(
        df_matchups, df,
        left_on='Pro Team', 
        right_on='Home/Neutral',
        how='left'
    )

    # Merge on visitor team
    away_matches = pd.merge(
        df_matchups, df,
        left_on='Pro Team', 
        right_on='Visitor/Neutral',
        how='left'
    )

    merged_final = pd.concat([home_matches, away_matches], ignore_index=True)
    # Keep all original columns from matchup_df plus 'Date'
    merged_final_cleaned = merged_final[
        ['Team Role', 'Team ID', 'Opponent ID', 'Player Name', 
        'Pro Team', 'Date', 'Injury Status', 'Injured']
    ].drop_duplicates()

    # Rename 'Date' to 'gameDate'
    merged_final_cleaned = merged_final_cleaned.rename(columns={'Date': 'gameDate'})

    # Display the final cleaned DataFrame
    print(merged_final_cleaned)

    merged_final_cleaned[['week', 'week_start_dt', 'week_end_dt']] = merged_final_cleaned['gameDate'].apply(assign_week_info)

        # Save the matchup data to a CSV file
    cwd = os.getcwd()
    matchup_save_path = os.path.join(cwd, "data/daily_matchups")
    os.makedirs(matchup_save_path, exist_ok=True)
    merged_final_cleaned[merged_final_cleaned.gameDate==TODAY_DT].to_csv(f"{matchup_save_path}/games_played_{TODAY_DT}.csv")

    print("Matchup data saved successfully.")

if __name__ == "__main__":
    main()

