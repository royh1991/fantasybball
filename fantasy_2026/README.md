# Fantasy 2026 - Data Collection Pipeline

Complete data collection system for the 2025-26 fantasy basketball season.

## Overview

This pipeline extracts current fantasy league rosters from ESPN, collects 4 seasons of historical NBA game logs for all rostered players, and fetches current week matchup data - all with intelligent deduplication and fuzzy name matching.

## Features

- **ESPN Roster Extraction**: Snapshots of all team rosters with player details
- **Historical Data Collection**: 4 seasons of game logs (2025-26, 2024-25, 2023-24, 2022-23)
- **Matchup Data Extraction**: Current week matchups with box scores and player performance
- **Smart Deduplication**: Avoids re-fetching players already in database
- **Fuzzy Name Matching**: Handles spelling differences between ESPN and NBA API
- **Roster Tracking**: Tracks `currently_rostered` flag to handle player drops/adds
- **Rate Limiting**: Respects NBA API limits (600ms between requests)

## Quick Start

### Prerequisites

```bash
# Install required packages
pip install pandas pyyaml nba-api fuzzywuzzy python-Levenshtein

# Ensure espn-api is available in parent directory
# Should be at: /Users/rhu/fantasybasketball2/espn-api
```

### Set ESPN Credentials

```bash
export ESPN_S2="your_espn_s2_cookie"
export ESPN_SWID="{your_swid_cookie}"
```

**Finding your credentials:**
1. Log in to ESPN Fantasy Basketball at fantasy.espn.com
2. Open browser DevTools (F12)
3. Go to Application > Cookies > espn.com
4. Copy `espn_s2` and `SWID` values

### Run the Pipeline

```bash
# Run everything (recommended for first time)
python run_all.py

# Or run individual steps
python run_all.py --step 1  # Extract rosters only
python run_all.py --step 2  # Collect historical data only
python run_all.py --step 3  # Get current week matchups only
python run_all.py --step 4  # Create player mapping only
```

## Pipeline Steps

### Step 1: Extract Rosters (`1_extract_rosters.py`)

Connects to ESPN Fantasy API and extracts:
- All team rosters with player names, positions, pro teams
- Injury status and acquisition type
- Season averages and totals (if available)
- Extraction timestamp for tracking changes over time

**Output:**
- `data/roster_snapshots/roster_snapshot_YYYYMMDD_HHMMSS.csv`
- `data/roster_snapshots/roster_latest.csv`

**Key Fields:**
```csv
extraction_date, fantasy_team_name, player_name, position, pro_team,
currently_rostered, injury_status, player_id_espn
```

### Step 2: Collect Historical Data (`2_collect_historical_data.py`)

For each rostered player:
1. Checks if player already exists in existing game logs
2. Uses fuzzy matching to map ESPN names to NBA API
3. Fetches 4 seasons of game-by-game data (current + 3 historical)
4. Combines with existing data and deduplicates

**Smart Fetching Logic:**
- If player exists in database → Only fetch current season (2025-26)
- If brand new player → Fetch all 4 seasons

**Output:**
- `data/historical_gamelogs/historical_gamelogs_YYYYMMDD_HHMMSS.csv`
- `data/historical_gamelogs/historical_gamelogs_latest.csv`

**Deduplication:**
- Loads existing data from: `/Users/rhu/fantasybasketball2/data/static/active_players_historical_game_logs.csv`
- Skips players already in database with current season data
- Removes duplicate game entries (same player + game ID)

### Step 3: Get Current Week Matchups (`3_get_matchups.py`)

Extracts current week matchup data from ESPN:
1. Current week's matchup pairings (which teams are playing each other)
2. Current scores and projections (for points leagues) or category records (for category leagues)
3. Detailed box scores with player-level performance data
4. Active lineup information (who's starting, who's benched)

**Output:**
- `data/matchups/matchups_YYYYMMDD_HHMMSS.csv` - Team matchup data
- `data/matchups/matchups_latest.csv`
- `data/matchups/box_scores_YYYYMMDD_HHMMSS.csv` - Player performance data
- `data/matchups/box_scores_latest.csv`

**Matchup Data Fields:**
```csv
week, matchup_period, home_team_name, away_team_name, home_score, away_score,
home_projected, away_projected, scoring_type
```

**Box Score Data Fields:**
```csv
week, matchup, team_name, team_side, player_name, player_id, position,
slot_position, pro_team, points, projected_points, stat_*
```

### Step 4: Create Player Mapping (`4_create_player_mapping.py`)

Creates a canonical mapping between ESPN and NBA API player names:
1. Loads unique players from roster snapshot
2. Loads unique players from historical game logs
3. Matches players using normalized names (removes accents, etc.)
4. Generates mapping file for easy data joins

**Why This Is Important:**
- ESPN uses "Luka Doncic" (no accents)
- NBA API uses "Luka Dončić" (with accents)
- Mapping file links them via normalized name "luka doncic"

**Output:**
- `data/mappings/player_mapping_YYYYMMDD_HHMMSS.csv`
- `data/mappings/player_mapping_latest.csv`

**Usage Example:**
```python
import pandas as pd

# Load data
mapping = pd.read_csv('data/mappings/player_mapping_latest.csv')
roster = pd.read_csv('data/roster_snapshots/roster_latest.csv')
gamelogs = pd.read_csv('data/historical_gamelogs/historical_gamelogs_latest.csv')

# Join roster to mapping (links ESPN names to NBA API names)
roster_mapped = roster.merge(mapping, left_on='player_name', right_on='espn_name')

# Join to game logs
full_data = roster_mapped.merge(gamelogs, left_on='nba_api_name', right_on='PLAYER_NAME')

# Now you have: roster info + game log data, all properly linked!
```

## Configuration

Edit `config/league_config.yaml` to customize:

```yaml
league:
  id: 40204
  season: 2026  # 2025-26 NBA season

data_collection:
  historical_seasons: 3
  nba_seasons:
    - "2024-25"
    - "2023-24"
    - "2022-23"
  rate_limit_ms: 600  # Delay between NBA API calls

matching:
  fuzzy_threshold: 85  # Name matching strictness (0-100)
```

## Directory Structure

```
fantasy_2026/
├── config/
│   └── league_config.yaml          # League settings
├── scripts/
│   ├── 1_extract_rosters.py        # ESPN roster extraction
│   ├── 2_collect_historical_data.py # NBA API game log collection
│   ├── 3_get_matchups.py           # ESPN matchup data extraction
│   └── 4_create_player_mapping.py  # Player name mapping
├── data/
│   ├── roster_snapshots/           # Timestamped roster CSVs
│   ├── historical_gamelogs/        # Combined game logs
│   ├── matchups/                   # Current week matchup data
│   └── mappings/                   # Player name mappings
├── logs/                           # Execution logs (future)
├── run_all.py                      # Main orchestration script
└── README.md                       # This file
```

## Output Data Schemas

### Roster Snapshot

| Column | Description |
|--------|-------------|
| `extraction_date` | Date of extraction (YYYY-MM-DD) |
| `extraction_timestamp` | Full timestamp (ISO format) |
| `season` | NBA season (e.g., 2026) |
| `fantasy_team_id` | ESPN team ID |
| `fantasy_team_name` | Team name |
| `fantasy_team_abbrev` | Team abbreviation |
| `player_name` | Player full name |
| `player_id_espn` | ESPN player ID |
| `position` | Primary position |
| `eligible_positions` | All eligible positions |
| `pro_team` | NBA team abbreviation |
| `currently_rostered` | Boolean (always True in snapshots) |
| `injury_status` | Current injury status |
| `season_avg_pts` | Season average fantasy points |

### Historical Game Logs

Standard NBA API format with additions:
- `PLAYER_NAME`: Matched NBA API player name
- `season`: Season string (e.g., "2023-24")
- All standard box score stats: `PTS`, `REB`, `AST`, `STL`, `BLK`, `FGM`, `FGA`, etc.

### Matchup Data

| Column | Description |
|--------|-------------|
| `week` | Fantasy week number |
| `matchup_period` | ESPN matchup period ID |
| `matchup_num` | Matchup number for the week |
| `home_team_id` | Home team's ESPN ID |
| `home_team_name` | Home team name |
| `away_team_id` | Away team's ESPN ID |
| `away_team_name` | Away team name |
| `scoring_type` | 'points' or 'category' |
| `home_score` | Home team score (points leagues) |
| `away_score` | Away team score (points leagues) |
| `home_projected` | Home team projected score |
| `away_projected` | Away team projected score |
| `home_wins` | Home team category wins (category leagues) |
| `home_losses` | Home team category losses |
| `home_ties` | Home team category ties |

### Box Score Data

| Column | Description |
|--------|-------------|
| `week` | Fantasy week number |
| `matchup` | Matchup description (e.g., "Team A vs Team B") |
| `team_id` | Fantasy team's ESPN ID |
| `team_name` | Fantasy team name |
| `team_side` | 'home' or 'away' |
| `player_name` | Player full name |
| `player_id` | ESPN player ID |
| `position` | Player's primary position |
| `slot_position` | Roster slot (e.g., "PG", "UTIL", "Bench") |
| `pro_team` | NBA team abbreviation |
| `points` | Fantasy points scored (if available) |
| `projected_points` | Projected fantasy points |
| `stat_*` | Individual stat categories (varies by league type) |

### Player Mapping

| Column | Description |
|--------|-------------|
| `espn_name` | Player name from ESPN (e.g., "Luka Doncic") |
| `player_id_espn` | ESPN player ID |
| `nba_api_name` | Player name from NBA API (e.g., "Luka Dončić") |
| `nba_api_id` | NBA API player ID |
| `normalized_name` | Normalized name for matching (e.g., "luka doncic") |
| `position` | Primary position |
| `matched` | Boolean indicating if mapping was successful |

## Handling Edge Cases

### Rookies
- No historical data available (will skip with message)
- Only current season data will be collected

### Name Mismatches
- Fuzzy matching threshold of 85% handles most variations
- Examples handled:
  - "Nic Claxton" ↔ "Nicolas Claxton"
  - "AJ Green" ↔ "A.J. Green"
  - "PJ Washington" ↔ "P.J. Washington"

### Players Not Found
- Logged with message: "Could not find NBA player ID"
- May be due to:
  - Misspelling in ESPN data
  - Player not in NBA API database
  - International/two-way players

### Rate Limiting
- 600ms delay between NBA API requests (configurable)
- Typical run time: 5-10 minutes for 150 players across 3 seasons
- If rate limited, increase `rate_limit_ms` in config

## Tracking Roster Changes Over Time

To track player drops/adds:

1. **Run pipeline weekly** to create new snapshots
2. **Compare snapshots** to identify changes:

```python
import pandas as pd

# Load two snapshots
week1 = pd.read_csv('data/roster_snapshots/roster_snapshot_20251022_120000.csv')
week2 = pd.read_csv('data/roster_snapshots/roster_snapshot_20251029_120000.csv')

# Find dropped players
week1_players = set(week1['player_name'])
week2_players = set(week2['player_name'])

dropped = week1_players - week2_players
added = week2_players - week1_players

print(f"Dropped: {dropped}")
print(f"Added: {added}")
```

3. **Update historical data** for new players by running Step 2 again

## Integration with Other Systems

### With `fantasy_modeling` System

```python
from fantasy_2026.scripts.data_loader import load_latest_data

# Load roster and game logs
roster_df, gamelogs_df = load_latest_data()

# Use with fantasy_modeling Bayesian simulator
from fantasy_modeling.simulation.game_simulator import GameSimulator

simulator = GameSimulator()
# ... fit models using gamelogs_df
```

### With Parent Directory Data

- **Reads from**: `/Users/rhu/fantasybasketball2/data/static/active_players_historical_game_logs.csv`
- **Writes to**: `fantasy_2026/data/historical_gamelogs/`
- **Combines**: New and existing data, removing duplicates

## Troubleshooting

### "ESPN credentials not found"
- Ensure `ESPN_S2` and `ESPN_SWID` environment variables are set
- Check they're properly formatted (S2 is long string, SWID is in braces)

### "No roster snapshot found"
- Run Step 1 first: `python run_all.py --step 1`
- Check `data/roster_snapshots/roster_latest.csv` exists

### "Import errors" (nba_api, fuzzywuzzy, etc.)
- Install missing packages: `pip install nba-api fuzzywuzzy python-Levenshtein pandas pyyaml`

### "espn_api.basketball could not be imported"
- Verify `espn-api` exists in parent directory
- Clone if missing: `git clone https://github.com/cwendt94/espn-api ../espn-api`

### Slow execution
- Increase rate limit in config: `rate_limit_ms: 1000`
- This is normal for large rosters (5-10 minutes)

### Player not found in NBA API
- Check spelling in ESPN roster
- Verify player is in NBA database (may not be for international/two-way players)
- Adjust fuzzy threshold: `fuzzy_threshold: 80` (lower = more lenient)

## Example Usage

```bash
# First time setup
export ESPN_S2="your_cookie_here"
export ESPN_SWID="{your_swid_here}"

# Run full pipeline
python run_all.py

# Output:
# ============================================================
# FANTASY 2026 DATA COLLECTION PIPELINE
# ============================================================
#
# ENVIRONMENT SETUP
# ✓ ESPN_S2: AEBp...
# ✓ ESPN_SWID: {ABC...}
# ✓ All dependencies satisfied
#
# STEP: Extract ESPN Fantasy Rosters
# ✓ Connected to ESPN league 40204
# ✓ Saved roster snapshot: data/roster_snapshots/roster_latest.csv
#
# STEP: Collect Historical Game Logs
# ✓ Loaded 150 unique players
# [1/150] Stephen Curry
#   ✓ Already have historical data, skipping
# [2/150] LeBron James
#   Fetching 2024-25... ✓ 45 games
#   Fetching 2023-24... ✓ 71 games
#   Fetching 2022-23... ✓ 55 games
#   ✓ Total: 171 games across 3 seasons
# ...
#
# ✓ PIPELINE COMPLETED SUCCESSFULLY
```

## Data Quality Checks

After running, verify data quality:

```python
import pandas as pd

# Check roster
roster = pd.read_csv('data/roster_snapshots/roster_latest.csv')
print(f"Total players: {len(roster)}")
print(f"Unique teams: {roster['fantasy_team_name'].nunique()}")
print(f"Players by position:\n{roster['position'].value_counts()}")

# Check game logs
logs = pd.read_csv('data/historical_gamelogs/historical_gamelogs_latest.csv')
print(f"\nTotal game logs: {len(logs)}")
print(f"Unique players: {logs['PLAYER_NAME'].nunique()}")
print(f"Seasons: {sorted(logs['season'].unique())}")
print(f"Date range: {logs['GAME_DATE'].min()} to {logs['GAME_DATE'].max()}")
```

## Maintenance

### Weekly Workflow

```bash
# Week 1
python run_all.py
# -> Creates roster_snapshot_20251022_120000.csv

# Week 2 (after roster changes)
python run_all.py --step 1  # Just get new roster
python run_all.py --step 2  # Fetch data for any new players
# -> Creates roster_snapshot_20251029_120000.csv
```

### Season End Archival

```bash
# Archive all snapshots
mkdir -p archive/2025-26
cp -r data/roster_snapshots archive/2025-26/
cp -r data/historical_gamelogs archive/2025-26/
```

## Performance

- **Roster extraction**: ~10-30 seconds
- **Historical data collection**:
  - 0 new players: ~5 seconds (all cached)
  - 50 new players: ~3-5 minutes
  - 150 new players: ~8-12 minutes

## Future Enhancements

Potential additions:
- [ ] Daily game log updates (not just historical)
- [ ] Injury status tracking over time
- [ ] Trade detection and logging
- [ ] Waiver wire activity tracking
- [ ] Integration with fantasy_modeling for automatic projections
- [ ] Web dashboard for roster visualization

## Support

For issues or questions:
1. Check this README
2. Review `config/league_config.yaml` settings
3. Enable verbose logging in scripts
4. Check NBA API status: https://github.com/swar/nba_api

## License

Part of the fantasy_basketball project. See parent directory for license.
