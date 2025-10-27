# Fixing Missing Player Warnings

## Problem

When running `simulate_season.py`, you see warnings like:
```
WARNING: No data for Nicolas Claxton, using zeros
WARNING: No data for Jimmy Butler, using zeros
WARNING: No data for Alexandre Sarr, using zeros
WARNING: No data for Cooper Flagg, using zeros
```

## Root Causes

**1. Name Mismatches**
- ADP file uses "Nicolas Claxton" but NBA API has "Nic Claxton"
- ADP file uses "Jimmy Butler" but NBA API has "Jimmy Butler III"

**2. 2024 Rookies**
- Players like Alexandre Sarr, Kel'el Ware, Jared McCain just started playing
- They weren't in the original data collection (ran too early in season)
- They now have NBA data but it wasn't collected

**3. 2025 Prospects**
- Players like Cooper Flagg, Dylan Harper haven't been drafted yet
- They're in ADP projections but have NO NBA games
- These will use projections only (no historical data)

---

## Solution: Run `collect_missing_data.py`

### Step 1: Run the Collection Script

```bash
cd /Users/rhu/fantasybasketball2/h_scoring
python collect_missing_data.py
```

**What it does:**
1. Checks which players are missing from your dataset
2. Attempts to find them in NBA API with name fuzzy matching
3. Collects their game logs (if they exist)
4. Merges the new data with your existing dataset
5. Saves updated files with a new timestamp

**Expected output:**
```
================================================================================
COLLECTING MISSING PLAYER DATA
================================================================================

Loading existing data:
  Weekly data: data/league_weekly_data_20250930_113053.csv
  Variances: data/player_variances_20250930_113053.json

Existing data: 187 players, 10750 weeks

Collecting data for Nicolas Claxton...
  Found match: 'Nicolas Claxton' → 'Nic Claxton'
  Fetching 2022-23...
    Found 65 games
  Fetching 2023-24...
    Found 71 games
  Fetching 2024-25...
    Found 15 games
  ✓ Collected 151 games across 3 seasons
  ✓ Added 47 weeks

Collecting data for Jimmy Butler...
  [Similar output]

Collecting data for Alexandre Sarr...
  Fetching 2024-25...
    Found 25 games
  ✓ Collected 25 games across 1 season
  ✓ Added 8 weeks

Collecting data for Cooper Flagg...
  → Cooper Flagg is a 2025 draft prospect (not in NBA yet) - skipping

================================================================================
SAVING COMBINED DATA
================================================================================

✓ Saved updated data:
  data/league_weekly_data_20251002_150000.csv
  data/player_variances_20251002_150000.json

New totals:
  Players: 192 (+5)
  Weeks: 10863 (+113)

================================================================================
PLAYERS ADDED
================================================================================
  ✓ Nicolas Claxton: 151 games
  ✓ Jimmy Butler: 48 games
  ✓ Alexandre Sarr: 25 games
  ✓ Kel'el Ware: 18 games
  ✓ Jared McCain: 22 games
```

### Step 2: Verify It Worked

Re-run your simulation:
```bash
python simulate_season.py
```

**Before:** 200+ warnings about missing players
**After:** Only warnings for 2025 prospects (Cooper Flagg, Dylan Harper) - these are expected!

---

## What About Future Prospects?

### 2025 Draft Prospects (No NBA Data Yet)

These players will still show warnings:
- Cooper Flagg
- Dylan Harper
- Ace Bailey
- VJ Edgecombe

**This is NORMAL!** They haven't played NBA games yet.

**Two options:**

**Option 1: Ignore them (safest)**
- Just accept the warnings
- They'll contribute 0 stats in simulations
- This is realistic - they're not playing yet!

**Option 2: Use projections (risky)**
- Use the projection-aware system
- Set very high projection weight (0.9-1.0)
- Risk: Projections for rookies are unreliable

**Option 3: Wait until they play**
- After they're drafted and play ~10 games
- Run `collect_missing_data.py` again
- Their names will move from `rookies_2025_prospects` to `rookies_2024_check`

---

## Manual Updates Needed

If a player still isn't found after running the script, add them to `player_name_mappings.json`:

```json
{
  "mappings": {
    "jimmy butler": "jimmy butler iii",
    "nicolas claxton": "nic claxton",
    "YOUR PLAYER NAME": "nba api name"
  }
}
```

**How to find the NBA API name:**

```python
from nba_api.stats.static import players

# Search for player
all_players = players.get_players()
matches = [p for p in all_players if 'LAST_NAME' in p['full_name'].lower()]
for m in matches:
    print(m['full_name'])
```

Then run `collect_missing_data.py` again.

---

## Updating `collect_full_data.py` (Optional)

The main collection script already uses `player_name_mappings.json`, so future runs will automatically pick up the name fixes.

**When to re-run full collection:**
- Start of a new season (to get full 2025-26 data)
- If you want to refresh ALL historical data
- If many new rookies have debuted

**Command:**
```bash
python collect_full_data.py
```

This will take ~15-20 minutes but will get fresh data for all 200+ players.

---

## Summary

**Quick Fix (5 minutes):**
```bash
python collect_missing_data.py
```

This will:
- ✓ Add Nicolas Claxton, Jimmy Butler (name fixes)
- ✓ Add Alexandre Sarr, Kel'el Ware, Jared McCain (2024 rookies)
- ✓ Skip Cooper Flagg, Dylan Harper (2025 prospects - no data yet)
- ✓ Create new updated data files
- ✓ Reduce warnings from 200+ to ~10

**Expected remaining warnings:**
- Cooper Flagg (not in NBA)
- Dylan Harper (not in NBA)
- Ace Bailey (not in NBA)
- Any other 2025 draft prospects

**These are EXPECTED and OK!**
