# Setup Instructions for H-Scoring

## Step 1: Create Conda Environment

```bash
cd /Users/rhu/fantasybasketball2/h_scoring

# Option A: Use the setup script
./setup_env.sh

# Option B: Create manually
conda env create -f environment.yml
```

## Step 2: Activate Environment

```bash
conda activate h_scoring
```

## Step 3: Test Data Collector (Quick Test - 20 players)

```bash
python test_data_collector.py
```

This will:
- Fetch data for 20 players (takes ~1-2 minutes)
- Save raw CSV and JSON files to `data/` directory
- Show you a preview of the data

**Output files:**
- `data/league_weekly_data_TIMESTAMP.csv` - Raw weekly statistics
- `data/player_variances_TIMESTAMP.json` - Player-specific variance data

## Step 4: Collect Full Dataset (Optional - 200 players)

```bash
# Collect 200 players (takes ~5-10 minutes)
python collect_full_data.py 200

# Or specify multiple seasons
python collect_full_data.py 200 2022-23,2023-24
```

## Step 5: Use the Raw Data

The raw files are saved in CSV and JSON format:

### Weekly Data CSV Format
```csv
PLAYER_ID,PLAYER_NAME,WEEK_ID,PTS,REB,AST,STL,BLK,TOV,FG3M,FGM,FGA,FTM,FTA,DD,FG_PCT,FT_PCT,FG3_PCT,GAMES_PLAYED
203507,Giannis Antetokounmpo,2023_W1,145,68,34,5,8,25,3,53,98,36,51,5,0.541,0.706,0.231,4
...
```

Columns:
- `PLAYER_ID`: NBA player ID
- `PLAYER_NAME`: Player name
- `WEEK_ID`: Week identifier (YEAR_WWeekNum)
- `PTS`, `REB`, `AST`, `STL`, `BLK`, `TOV`: Counting stats (weekly totals)
- `FG3M`: Three-pointers made
- `FGM`, `FGA`: Field goals made/attempted
- `FTM`, `FTA`: Free throws made/attempted
- `DD`: Double-doubles (weekly count)
- `FG_PCT`, `FT_PCT`, `FG3_PCT`: Shooting percentages
- `GAMES_PLAYED`: Games played that week

### Variance JSON Format
```json
{
  "Giannis Antetokounmpo": {
    "PTS": {
      "mean": 30.2,
      "std": 8.5,
      "var": 72.25,
      "cv": 0.28,
      "weeks": 25
    },
    "REB": {
      "mean": 11.5,
      "std": 3.2,
      "var": 10.24,
      "cv": 0.28,
      "weeks": 25
    },
    ...
  },
  ...
}
```

Fields:
- `mean`: Average weekly value
- `std`: Standard deviation (week-to-week)
- `var`: Variance
- `cv`: Coefficient of variation (std/mean)
- `weeks`: Number of weeks of data

## Troubleshooting

**"conda: command not found"**
- Install Miniconda or Anaconda first

**"ModuleNotFoundError: No module named 'nba_api'"**
- Make sure you activated the environment: `conda activate h_scoring`

**Rate limit errors**
- NBA API has rate limits
- Script includes 0.6s delays
- If it fails, just run again

**No data collected**
- Some players may not have enough weeks (min 20 required for full dataset)
- The script will skip players with insufficient data

## File Locations

All data files are saved to:
```
/Users/rhu/fantasybasketball2/h_scoring/data/
```

Files are timestamped, so you can collect multiple datasets without overwriting.

## Next Steps

After collecting data:

1. **View raw data**: Open CSV in Excel or any spreadsheet software
2. **Use in draft assistant**:
   ```bash
   python draft_assistant.py
   ```
3. **Use programmatically**:
   ```python
   import pandas as pd
   import json

   # Load data
   df = pd.read_csv('data/league_weekly_data_TIMESTAMP.csv')
   with open('data/player_variances_TIMESTAMP.json') as f:
       variances = json.load(f)

   # Analyze
   print(df.groupby('PLAYER_NAME')['PTS'].mean().sort_values(ascending=False).head(10))
   ```