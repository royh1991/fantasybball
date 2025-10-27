# H-Scoring Algorithm - Required Files

## Minimum Required Files to Run

### 1. Core Algorithm Modules (Required)

**Location:** `/Users/rhu/fantasybasketball2/h_scoring/modules/`

```
modules/
├── scoring.py                      # X-score and G-score calculations
├── covariance.py                   # Covariance matrix and baseline weights
└── h_optimizer_final.py            # H-score optimizer with Adam
```

**Purpose:**
- `scoring.py`: Calculates X-scores and G-scores for all players
- `covariance.py`: Builds category correlation matrix, calculates baseline weights
- `h_optimizer_final.py`: Runs gradient descent optimization to find H-scores

---

### 2. Main Execution Scripts (Choose One)

**Location:** `/Users/rhu/fantasybasketball2/h_scoring/`

```
simulate_season.py                  # Draft + Season Simulation (RECOMMENDED)
simulate_draft.py                   # Draft Only
draft_assistant.py                  # Interactive Draft Tool
```

**Usage:**

**Option A - Full Simulation (Draft + 100 Seasons):**
```bash
python simulate_season.py -p 6       # Draft from position 6
python simulate_season.py -p 1 -n 50 # Position 1, 50 seasons
```

**Option B - Draft Only:**
```bash
python simulate_draft.py
```

**Option C - Interactive Draft Assistant:**
```bash
python draft_assistant.py
```

---

### 3. Data Files (Required)

**Location:** `/Users/rhu/fantasybasketball2/h_scoring/data/`

#### A. Historical NBA Data (Required)

```
data/league_weekly_data_20251002_125746.csv    # Weekly aggregated stats (1.2 MB)
data/player_variances_20251002_125746.json     # Per-game variance by player (546 KB)
```

**Contents:**
- `league_weekly_data_*.csv`: 193 players, ~11,000 weekly stat lines (2022-25 seasons)
- `player_variances_*.json`: Per-game variance for each player by category

**Format (league_weekly_data):**
```csv
PLAYER_ID,PLAYER_NAME,NBA_SEASON,SEASON_WEEK_ID,PTS,REB,AST,STL,BLK,TOV,FG3M,FGM,FGA,FTM,FTA,FG_PCT,FT_PCT,FG3_PCT,DD,GAMES_PLAYED
203507,Giannis Antetokounmpo,2023-24,2023-24_W1,85,47,21,3,5,12,2,31,52,21,28,0.596,0.75,0.333,3,3
```

**Format (player_variances):**
```json
{
  "Giannis Antetokounmpo": {
    "PTS": {
      "mean_per_game": 31.1,
      "std_per_game": 8.2,
      "var_per_game": 67.24,
      "cv_per_game": 0.264,
      "games": 214
    },
    ...
  }
}
```

#### B. ADP Rankings (Required for Draft Simulation)

```
/Users/rhu/fantasybasketball2/data/fantasy_basketball_clean2.csv  # ADP rankings (31 KB)
```

**Contents:** ~200 players with projections and ADP ranks

**Format:**
```csv
PLAYER,TEAM,POS,ADP,GP,PTS,TREB,AST,STL,BLK,TO,FG%,FT%,3PTM,3P%,DD
Nikola Jokic,DEN,C,1.1,76,2102.3,837.9,607.9,105.5,53.6,208.1,0.632,0.828,66.2,0.389,57.8
Luka Doncic,DAL,PG,4.7,70,2006.4,628.6,610.4,104.3,33.6,247.8,0.479,0.76,184.8,0.381,39.2
```

#### C. Configuration Files (Optional but Recommended)

```
player_name_mappings.json           # Name mappings for NBA API mismatches
data/do_not_draft.csv               # Players to exclude (injuries, etc.)
```

**player_name_mappings.json:**
```json
{
  "mappings": {
    "jimmy butler": "jimmy butler iii",
    "nicolas claxton": "nic claxton"
  },
  "rookies_2024_check": [
    "alexandre sarr",
    "kel'el ware",
    "jared mccain"
  ],
  "rookies_2025_prospects": [
    "cooper flagg",
    "dylan harper",
    "ace bailey"
  ]
}
```

**do_not_draft.csv:**
```csv
PLAYER
Kawhi Leonard
Joel Embiid
Zion Williamson
```

---

## File Dependencies Diagram

```
simulate_season.py
    ├── simulate_draft.py
    │   ├── draft_assistant.py
    │   │   ├── modules/scoring.py
    │   │   ├── modules/covariance.py
    │   │   └── modules/h_optimizer_final.py
    │   │       └── modules/scoring.py (circular)
    │   ├── data/league_weekly_data_*.csv
    │   ├── data/player_variances_*.json
    │   └── /Users/rhu/fantasybasketball2/data/fantasy_basketball_clean2.csv
    └── (season simulation logic)
```

---

## How to Generate Data Files (If Missing)

### Option 1: Use Existing Data (Fastest - Already Complete!)

Your data files are already up-to-date:
- ✅ `league_weekly_data_20251002_125746.csv` (193 players, 10,968 weeks)
- ✅ `player_variances_20251002_125746.json` (193 players with variance data)
- ✅ ADP file at `/Users/rhu/fantasybasketball2/data/fantasy_basketball_clean2.csv`

**You can run the algorithm immediately!**

### Option 2: Regenerate Data (If Needed)

**Collect fresh NBA data (takes ~20-30 minutes):**

```bash
# Full data collection for all ADP players (2022-25 seasons)
python collect_full_data.py

# Add missing players (if you get warnings)
python collect_missing_data.py
```

**Output:**
- `data/league_weekly_data_YYYYMMDD_HHMMSS.csv`
- `data/player_variances_YYYYMMDD_HHMMSS.json`

---

## Quick Start Guide

### 1. Verify Files Exist

```bash
cd /Users/rhu/fantasybasketball2/h_scoring

# Check core modules
ls modules/scoring.py modules/covariance.py modules/h_optimizer_final.py

# Check data files
ls data/league_weekly_data_*.csv data/player_variances_*.json

# Check ADP file
ls /Users/rhu/fantasybasketball2/data/fantasy_basketball_clean2.csv
```

### 2. Run Draft Simulation

```bash
# Draft from position 6 (default)
python simulate_season.py

# Draft from position 1
python simulate_season.py -p 1

# Draft from position 12 with 50 seasons
python simulate_season.py -p 12 -n 50
```

### 3. View Results

**Console output shows:**
- Draft picks with H-scores and G-scores
- Season simulation results (100 seasons)
- Final team rankings
- Your team's win percentage

**Saved files:**
- `season_results_YYYYMMDD_HHMMSS.csv` - Full season results
- `draft_results_YYYYMMDD_HHMMSS.json` - Draft details

---

## Dependencies (Python Packages)

**Required:**
```
pandas>=1.5.0
numpy>=1.23.0
scipy>=1.9.0
```

**Optional (for data collection):**
```
nba_api>=1.1.0      # Only if regenerating data
requests>=2.28.0    # Only if regenerating data
```

**Install:**
```bash
pip install pandas numpy scipy

# Optional for data collection
pip install nba-api requests
```

---

## File Sizes

```
Total minimum required: ~1.8 MB

Core modules:          ~150 KB
  - scoring.py:         ~25 KB
  - covariance.py:      ~20 KB
  - h_optimizer_final.py: ~40 KB

Data files:           ~1.75 MB
  - league_weekly_data: ~1.2 MB
  - player_variances:   ~546 KB
  - ADP file:           ~31 KB

Scripts:              ~80 KB
  - simulate_season.py:  ~13 KB
  - simulate_draft.py:   ~12 KB
  - draft_assistant.py:  ~20 KB
```

---

## Optional Files

### For Advanced Usage

```
modules/scoring_with_projections.py  # Projection-aware scoring system
collect_missing_data.py              # Add missing players to dataset
test_h_scores_quick.py               # Quick H-score testing
debug_draft_detailed.py              # Detailed draft debugging
```

### For Documentation

```
DRAFT_ANALYSIS.md                    # Draft analysis (strategy-focused)
DRAFT_ANALYSIS_TECHNICAL.md          # Technical deep dive (formulas)
MISSING_PLAYERS_FIX.md               # Guide for fixing missing players
PROJECTION_AWARE_SYSTEM.md           # Projection system documentation
```

---

## Common Issues

### Missing Data Files

**Error:**
```
Error: No data files found!
```

**Solution:**
```bash
# Run data collection
python collect_full_data.py
```

### Name Mismatch Warnings

**Warning:**
```
WARNING: No data for Nicolas Claxton, using zeros
```

**Solution:**
```bash
# Run missing data collection
python collect_missing_data.py
```

### Import Errors

**Error:**
```
ModuleNotFoundError: No module named 'pandas'
```

**Solution:**
```bash
pip install pandas numpy scipy
```

---

## Minimal File Set for Deployment

If you want to package this for deployment, here's the absolute minimum:

```
h_scoring/
├── modules/
│   ├── scoring.py
│   ├── covariance.py
│   └── h_optimizer_final.py
├── simulate_season.py
├── simulate_draft.py
├── draft_assistant.py
├── data/
│   ├── league_weekly_data_20251002_125746.csv
│   ├── player_variances_20251002_125746.json
│   └── do_not_draft.csv
├── player_name_mappings.json
└── /path/to/fantasy_basketball_clean2.csv

Total: ~1.8 MB (7 files + 3 modules)
```

This minimal set allows you to:
- Run draft simulations
- Run season simulations
- Use interactive draft assistant
- Get H-scores for all 193 players

---

## Summary

**Absolutely Required (7 files):**
1. `modules/scoring.py`
2. `modules/covariance.py`
3. `modules/h_optimizer_final.py`
4. `simulate_season.py` OR `draft_assistant.py`
5. `data/league_weekly_data_*.csv`
6. `data/player_variances_*.json`
7. `data/fantasy_basketball_clean2.csv` (ADP file)

**Recommended (2 files):**
8. `player_name_mappings.json`
9. `data/do_not_draft.csv`

**Total:** 9 files, ~1.8 MB

**You already have everything needed to run the algorithm!**

Just run:
```bash
python simulate_season.py -p 6
```
