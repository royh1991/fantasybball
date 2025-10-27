# Data Collection Resume Instructions

## What Changed

The data collector now saves progress after each player, so you can resume if the collection is interrupted by API throttling or timeouts.

## Features Added

1. **Checkpoint System**: After each player is collected, data is saved to temporary files and a checkpoint file tracks progress
2. **Retry Logic**: Failed API calls are retried 3 times with exponential backoff (5s, 10s, 15s)
3. **Increased Timeout**: API timeout increased from default to 60 seconds
4. **Resume Capability**: Can pick up exactly where you left off

## How to Resume Collection

If your data collection gets interrupted (Ctrl+C or API errors), simply run:

```bash
python collect_full_data.py --resume
```

This will:
- Find the most recent checkpoint file
- Load all previously collected data
- Skip players that were already completed
- Continue from the next player

## Fresh Start

To start a new collection from scratch:

```bash
python collect_full_data.py
```

Or with custom parameters:

```bash
python collect_full_data.py 200  # Max 200 players
python collect_full_data.py 200 "2022-23,2023-24,2024-25"  # Custom seasons
```

## What Gets Saved

During collection, for each player:
- `temp_game_{player}_{timestamp}.csv` - Game-level data
- `temp_weekly_{player}_{timestamp}.csv` - Weekly aggregated data
- `temp_variance_{player}_{timestamp}.json` - Per-game variance stats
- `checkpoint_{timestamp}.json` - Progress tracker

When collection completes, all temp files are merged into final files:
- `league_game_data_{timestamp}.csv` - All game data
- `league_weekly_data_{timestamp}.csv` - All weekly data
- `player_variances_{timestamp}.json` - All variance data

## Handling API Throttling

If you see repeated timeout errors like:
```
Error fetching Tyler Herro (2022-23): HTTPSConnectionPool... Read timed out
```

1. Press Ctrl+C to stop the collection
2. Wait a few minutes for the API rate limit to reset
3. Run `python collect_full_data.py --resume`
4. Collection will continue from where it stopped

## Example Workflow

```bash
# Start collection
python collect_full_data.py

# ... gets throttled around player 150 ...
# Press Ctrl+C

# Wait 5 minutes

# Resume collection
python collect_full_data.py --resume

# ... completes successfully ...
```

## Checkpoint File Location

Checkpoint files are saved to:
```
data/checkpoint_YYYYMMDD_HHMMSS.json
```

The `--resume` flag automatically finds the most recent checkpoint.
