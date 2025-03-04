from nba_api.stats.endpoints import playergamelog
import pandas as pd
import time
from datetime import datetime
import os
# Set the date
TODAY_DATE = datetime.today().strftime('%Y-%m-%d')

# File paths
daily_matchups_path = f"/Users/rhu/fantasybasketball2/data/daily_matchups/games_played_{TODAY_DATE}.csv"
player_stats_path = "/Users/rhu/fantasybasketball2/data/static/player_stats_historical/active_players_historical_game_logs.csv"

# Season to fetch
season = '2024-25'

# Load the subset of players who played today
played_today_df = pd.read_csv(daily_matchups_path)
played_today_names = played_today_df['Player Name'].str.lower()  # Normalize names to lowercase for matching

# Load the player stats historical data
historical_stats_df = pd.read_csv(player_stats_path)
historical_stats_df['player_name'] = historical_stats_df['player_name'].str.lower()  # Normalize names to lowercase for matching

# Map the `PLAYER_ID` for players who played today
players_today_df = historical_stats_df[historical_stats_df['player_name'].isin(played_today_names)]
player_ids_today = players_today_df['Player_ID'].unique().tolist()

# Initialize an empty list to store game logs
game_logs_today = []

# Fetch game logs for players who played today
for player_id in player_ids_today:
    try:
        # Fetch the player's game logs for this season
        time.sleep(1)  # Rate-limit requests to avoid being blocked
        game_log = playergamelog.PlayerGameLog(player_id=player_id, season=season)
        game_log_df = game_log.get_data_frames()[0]
        
        # Filter for games played today
        game_log_df['GAME_DATE'] = pd.to_datetime(game_log_df['GAME_DATE'], format='%b %d, %Y').dt.strftime('%Y-%m-%d')
        game_log_today = game_log_df[game_log_df['GAME_DATE'] == TODAY_DATE]
        
        # Append the results if there are any games played today
        if not game_log_today.empty:
            game_logs_today.append(game_log_today)
            print(f"Fetched data for player ID {player_id} for today.")
    except Exception as e:
        # Handle errors gracefully
        print(f"Error fetching data for player ID {player_id}: {e}")

# Combine all game logs into a single DataFrame
if game_logs_today:
    games_played_today_df = pd.concat(game_logs_today, ignore_index=True)
else:
    games_played_today_df = pd.DataFrame()  # Create an empty DataFrame if no games

# Display the filtered game logs for today
print(games_played_today_df)

cwd = os.getcwd()
gamelog_save_path = os.path.join(cwd, "data/daily_gamelogs")
os.makedirs(gamelog_save_path, exist_ok=True)

# Save to CSV for analysis
games_played_today_df.to_csv(f"{gamelog_save_path}/player_daily_gamelogs_{TODAY_DATE}.csv")
