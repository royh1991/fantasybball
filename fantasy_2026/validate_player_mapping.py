"""
Validate Player Mapping

Checks if all players in daily matchups can be properly mapped to NBA API names.
"""

import pandas as pd
import os

def validate_mapping():
    """Validate that all players can be mapped."""

    print("="*80)
    print("VALIDATING PLAYER MAPPING")
    print("="*80)

    # Load mapping
    mapping = pd.read_csv('/Users/rhu/fantasybasketball2/fantasy_2026/data/mappings/player_mapping_latest.csv')
    print(f"\nLoaded {len(mapping)} player mappings")

    # Check for special characters
    special_char_names = mapping[mapping['espn_name'].str.contains(r'[^\x00-\x7F]', regex=True, na=False)]
    if len(special_char_names) > 0:
        print(f"\nFound {len(special_char_names)} names with special characters:")
        for _, row in special_char_names.iterrows():
            print(f"  '{row['espn_name']}' -> '{row['nba_api_name']}'")

    # Load daily matchups
    daily_matchups_dir = '/Users/rhu/fantasybasketball2/data/daily_matchups'
    daily_matchups_files = [f for f in os.listdir(daily_matchups_dir) if f.endswith('.csv')]

    daily_matchups_list = []
    for file in daily_matchups_files:
        df = pd.read_csv(os.path.join(daily_matchups_dir, file))
        daily_matchups_list.append(df)

    daily_matchups = pd.concat(daily_matchups_list, ignore_index=True)

    unique_players = daily_matchups['Player Name'].unique()
    print(f"\nFound {len(unique_players)} unique players in daily matchups")

    # Check mapping coverage
    mapped_players = []
    unmapped_players = []

    for player_name in unique_players:
        match = mapping[mapping['espn_name'].str.lower() == player_name.lower()]
        if len(match) > 0:
            mapped_players.append((player_name, match.iloc[0]['nba_api_name']))
        else:
            unmapped_players.append(player_name)

    print(f"\nMapping Results:")
    print(f"  Mapped: {len(mapped_players)}/{len(unique_players)} ({len(mapped_players)/len(unique_players)*100:.1f}%)")
    print(f"  Unmapped: {len(unmapped_players)}/{len(unique_players)} ({len(unmapped_players)/len(unique_players)*100:.1f}%)")

    if len(unmapped_players) > 0:
        print(f"\nUNMAPPED PLAYERS:")
        for player in unmapped_players:
            print(f"  - {repr(player)}")

    # Load historical data to check model coverage
    historical = pd.read_csv('/Users/rhu/fantasybasketball2/fantasy_2026/data/historical_gamelogs/historical_gamelogs_latest.csv')
    historical_players = set(historical['PLAYER_NAME'].unique())

    # Check how many mapped players have historical data
    players_with_models = []
    players_without_models = []

    for player_name, nba_name in mapped_players:
        if nba_name in historical_players:
            players_with_models.append((player_name, nba_name))
        else:
            players_without_models.append((player_name, nba_name))

    print(f"\nModel Coverage (for mapped players):")
    print(f"  With historical data: {len(players_with_models)}/{len(mapped_players)} ({len(players_with_models)/len(mapped_players)*100:.1f}%)")
    print(f"  Without historical data: {len(players_without_models)}/{len(mapped_players)} ({len(players_without_models)/len(mapped_players)*100:.1f}%)")

    if len(players_without_models) > 0:
        print(f"\nMapped but NO HISTORICAL DATA:")
        for espn_name, nba_name in players_without_models[:10]:
            print(f"  - {espn_name} -> {nba_name}")
        if len(players_without_models) > 10:
            print(f"  ... and {len(players_without_models) - 10} more")

    # Save validation report
    validation_report = {
        'total_players_in_games': len(unique_players),
        'mapped': len(mapped_players),
        'unmapped': len(unmapped_players),
        'with_models': len(players_with_models),
        'without_models': len(players_without_models),
        'unmapped_players': unmapped_players,
        'players_without_models': [(e, n) for e, n in players_without_models]
    }

    df_report = pd.DataFrame({
        'player_name': unique_players,
        'mapped': [player in [p for p, _ in mapped_players] for player in unique_players],
        'has_model': [mapping[mapping['espn_name'].str.lower() == player.lower()].iloc[0]['nba_api_name']
                     in historical_players
                     if len(mapping[mapping['espn_name'].str.lower() == player.lower()]) > 0
                     else False
                     for player in unique_players]
    })

    df_report.to_csv('/Users/rhu/fantasybasketball2/fantasy_2026/player_mapping_validation.csv', index=False)
    print(f"\nSaved validation report to: player_mapping_validation.csv")

    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)

    return validation_report


if __name__ == "__main__":
    validate_mapping()
