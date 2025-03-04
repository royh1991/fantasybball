from glob import glob
import pandas as pd
import os

# Correct path to your player stats directory
data_path = '/Users/rhu/fantasybasketball2/data/static/player_stats_historical'
            
def combine_csv_files(path, prefix="df_"):
    """
    Combine all CSV files with a given prefix into one DataFrame.
    Each file should have the same header, which is retained only once.
    """
    # Use glob to match the full file name pattern
    csv_files = glob(os.path.join(path, f"{prefix}_*.csv"))

    # Debugging: Print the list of found files
    print(f"Found {len(csv_files)} files: {csv_files}")

    if not csv_files:
        print("No files found.")
        return None

    # Load and combine all CSVs, skipping header after the first file
    combined_df = pd.concat(
        (pd.read_csv(f) for f in csv_files), 
        ignore_index=True
    )

    print(f"Successfully combined {len(csv_files)} files.")
    return combined_df

# Example usage: Combine all Jayson Tatum game logs CSVs
combined_data = combine_csv_files(data_path, prefix="df")

# Check if combined_data is not None before calling .head()
if combined_data is not None:
    print(combined_data.head())
else:
    print("No data to display.")

combined_data = combined_data[combined_data.GS.isin(['1','0'])]
combined_data['DATE'] = pd.to_datetime(combined_data['DATE'], errors='coerce')

# Extract the season from the DATE column
combined_data['season'] = combined_data['DATE'].apply(lambda x: x.year if x.month >= 10 else x.year - 1)

# List of columns to convert to integers
counting_stat_cols = [
    'FG', 'FGA', '3P', '3PA', 'FT', 'FTA', 'ORB', 'DRB', 'TRB',
    'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS'
]

# Convert columns to numeric, forcing errors to NaN (useful for missing or invalid data)
combined_data[counting_stat_cols] = combined_data[counting_stat_cols].apply(pd.to_numeric, errors='coerce')

# Fill NaN values with 0 before converting to integer (optional, based on your needs)
combined_data[counting_stat_cols] = combined_data[counting_stat_cols].fillna(0).astype(int)


players_summary = pd.read_csv('/Users/rhu/fantasybasketball2/data/static/all_active_players_2024-10-23.csv')

# Perform a left join on the 'Name' column to get the 'Position'
combined_data_with_position = pd.merge(
    combined_data,            # Your main DataFrame
    players_summary[['Player Name', 'Position']],  # Select only the relevant columns from players_summary
    how='left',               # Left join to keep all rows from combined_data
    left_on='NAME',           # Column in combined_data to join on
    right_on='Player Name'           # Column in players_summary to join on
)

# Drop the redundant 'Name' column (optional, since you already have 'NAME')
combined_data_with_position = combined_data_with_position.drop(columns=['Player Name'])

# Display the result to verify
print(combined_data_with_position.head())


# Optionally, save the cleaned DataFrame to a new CSV
combined_data_with_position.to_csv("/Users/rhu/fantasybasketball2/data/intermediate/combined_data_cleaned.csv", index=False)
