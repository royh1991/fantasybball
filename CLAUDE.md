# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a fantasy basketball simulation and analysis system that uses Bayesian modeling to predict player performance. The system fetches data from ESPN Fantasy Basketball API and NBA stats API, applies empirical Bayes estimates, and runs Bayesian simulations to forecast player statistics.

## Data Pipeline Architecture

The codebase follows a sequential data pipeline (numbered scripts 1-5):

1. **1_daily_matchups.py** - Fetches daily matchup data from ESPN Fantasy Basketball API
   - Requires ESPN authentication cookies (`swid` and `espn_s2` in lines 23-24)
   - Outputs to `data/daily_matchups/games_played_{TODAY_DT}.csv`
   - Maps ESPN Pro Team IDs to NBA team names
   - Associates games with fantasy matchup weeks

2. **2_collect_historical_gamelogs.py** - Fetches historical game logs from NBA API
   - Uses `nba_api` library to collect data for active players
   - Covers seasons 2019-20 through 2023-24
   - Stores in `data/static/player_stats_historical/`

3. **3_collect_today_gamelogs.py** - Fetches current day's game logs
   - Filters for players who played on current date
   - Uses season '2024-25'
   - Requires files from steps 1 and 2

4. **4_create_combined_stats.py** - Combines player stats from multiple CSV files
   - Aggregates historical data per player
   - Located in `data/static/player_stats_historical/`

5. **5_get_eb_estimates.py** - Calculates Empirical Bayes estimates
   - Fits Beta distributions for shooting percentages (FG%, FT%, 3P%)
   - Applies position-based priors
   - Uses weighted recent games with 0.9 decay factor (10 most recent games)

## Core Simulation Components

### BayesianPlayerSimulator (`bayesian_player_simulator.py`)
The main simulation engine using hierarchical Bayesian models:
- Fits posterior distributions for shooting percentages and counting stats
- Uses PyMC for Bayesian inference
- Stores models in `self.shooting_models` and `self.stat_models`
- Key method: `simulate_game(context, game_type, player_id)` returns simulated box scores

### PureBayesianSimulator (`player_simulation_unified.py`)
Extends BayesianPlayerSimulator with:
- Pure posterior sampling without additional heuristics
- Visualization functions for comparing historical vs simulated distributions
- Box score stats: pts, reb, ast, stl, blk, tov, fgm, fga, fg3m, fg3a, ftm, fta

### BasketballSimulator (`_test_modeling.py`)
Alternative/experimental modeling approach with support for:
- ARIMA time series modeling
- PCA dimensionality reduction
- Team-level and league-level statistics

## Key Data Structures

- **Player Data**: Game-by-game logs with columns: player_id, game_id, date, team_id, opp_team_id, home, minutes, shooting stats (fga/fgm/fg3a/fg3m/fta/ftm), counting stats (pts/reb/ast/stl/blk/tov)
- **Matchup Data**: ESPN fantasy team rosters with injury status, pro team mappings, and game dates
- **EB Estimates**: Position-based Beta distribution parameters (alpha, beta) for shooting percentages

## Data Directories

- `data/daily_matchups/` - Daily fantasy matchup data
- `data/daily_gamelogs/` - Current season game logs
- `data/static/player_stats_historical/` - Historical game logs by player
- `data/static/missing_players.json` - Tracking for players without data
- `data/intermediate/` - Processing artifacts
- `full_simulations/` - Simulation outputs (timestamped by player and run)

## Running Simulations

No explicit test or build commands found. The simulation system is run manually through the numbered Python scripts in sequence.

To run a simulation:
```bash
python bayesian_player_simulator.py  # or
python player_simulation_unified.py
```

Simulation outputs are saved to `full_simulations/{player_name}_{timestamp}/` with visualizations and statistics.

## Dependencies

Key libraries used:
- `pandas`, `numpy` - Data manipulation
- `pymc`, `arviz` - Bayesian modeling
- `scipy`, `statsmodels` - Statistical distributions and time series
- `nba_api` - NBA statistics API client
- `requests` - ESPN Fantasy API calls
- `matplotlib`, `seaborn` - Visualizations
- `tqdm` - Progress bars

## ESPN Fantasy Basketball Integration

League-specific configuration in `1_daily_matchups.py`:
- `league_id = '40204'` (line 14)
- `season_id = 2025` (line 15)
- Update `swid` and `espn_s2` cookies for authentication
- Week mapping spans from 2024-10-22 to 2025-03-30 (lines 29-48)