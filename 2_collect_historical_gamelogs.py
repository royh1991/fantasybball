from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players
import pandas as pd
import time


# Fetch the player data
player_list = players.get_players()

# Convert the list of dictionaries into a DataFrame
player_df = pd.DataFrame(player_list)

# Filter for only active players
active_players_df = player_df[player_df['is_active'] == True]

# List of seasons to fetch (last 5 seasons including the current one)
seasons = ['2019-20', '2020-21', '2021-22', '2022-23', '2023-24']

# Initialize an empty list to store game logs
all_game_logs = []

# Loop through each active player
for _, player in active_players_df.iterrows():
    player_id = player['id']
    player_name = player['full_name']  # Optional: for tracking progress
    
    for season in seasons:
        try:
            # Fetch the player's game logs for the specific season
            time.sleep(1)
            game_log = playergamelog.PlayerGameLog(player_id=player_id, season=season)
            game_log_df = game_log.get_data_frames()[0]
            
            # Add the player's ID and name to the DataFrame for context
            game_log_df['player_id'] = player_id
            game_log_df['player_name'] = player_name
            
            # Append to the list of all game logs
            all_game_logs.append(game_log_df)
            print(f"Fetched data for {player_name} in season {season}.")
        except Exception as e:
            # Skip errors (e.g., rookies without data for older seasons)
            print(f"Skipped {player_name} for season {season}: {e}")

# Combine all game logs into a single DataFrame
career_game_logs_df = pd.concat(all_game_logs, ignore_index=True)

# Display the combined game logs
print(career_game_logs_df)

# Save to CSV for analysis
career_game_logs_df.to_csv("active_players_career_game_logs.csv", index=False)
