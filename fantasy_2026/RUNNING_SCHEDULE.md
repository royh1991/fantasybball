# Running Schedule & Data Freshness Guide

## Current Behavior

### What Gets Captured

When you run `python run_all.py`, here's what happens:

1. **Step 1: Roster Extraction**
   - Captures the **current** roster at the moment you run it
   - Takes a snapshot: which players are on which teams **right now**
   - Does NOT capture historical roster changes

2. **Step 2: Historical Game Logs**
   - For **existing players**: Only fetches current season (2025-26) games
   - For **new players** (just added to roster): Fetches all 4 seasons (2025-26, 2024-25, 2023-24, 2022-23)
   - Does NOT track which team owned the player when they played

3. **Step 3: Matchup Data**
   - Captures the **current week's** matchups and box scores
   - Shows current scores/projections as of when you run it

4. **Step 4: Player Mapping**
   - Links current roster names to NBA API names
   - Based on the current roster snapshot

## Scenarios

### Scenario 1: You Run Once Per Week (Recommended)

**Timeline:**
```
Monday Oct 21:  Run pipeline → Captures roster with players A, B, C
                              Fetches all current season games for A, B, C

Tuesday Oct 22: Player A is dropped, Player D is added
                (You don't run the script)

Monday Oct 28:  Run pipeline → Captures NEW roster with players B, C, D
                              Fetches current season for D (new player)
                              Updates games for B, C (existing players)
```

**What You Get:**
- ✅ Latest roster composition
- ✅ All games played by currently rostered players
- ❌ Games played by Player A (dropped before second run)
- ✅ Games played by Player D (even before they were added to your roster)

**Missing Data:**
- Player A's stats from Oct 22-28 (they were dropped)
- Historical record of when players joined/left teams

### Scenario 2: You Miss a Week

**Timeline:**
```
Monday Oct 21:  Run pipeline → Roster: A, B, C

Monday Oct 28:  (You forget to run!)
                Player A dropped, Player D added

Monday Nov 4:   Run pipeline → Roster: B, C, D
                              Fetches all D's 2025-26 games (including games from Oct 22-28)
```

**What You Get:**
- ✅ Current roster (B, C, D)
- ✅ All 2025-26 games for B, C, D
- ❌ Player A's data (they're not on the roster anymore)
- ✅ Player D's games from before they were rostered

**Key Point:** You DON'T lose game log data! The NBA API gives you all games from the season, not just games after they joined your roster.

### Scenario 3: You Run Daily

**Timeline:**
```
Monday:   Run → Roster: A, B, C (3 total games)
Tuesday:  Run → Roster: A, B, C (6 total games - games from Mon + Tue)
Wednesday: Player A dropped, Player D added
Wednesday: Run → Roster: B, C, D (D has all their games, A is now gone)
```

**What You Get:**
- ✅ Maximum freshness - always have latest matchups
- ✅ Daily roster snapshots (can track changes over time)
- ✅ All game logs stay current
- ⚠️ More API calls (but deduplication prevents re-fetching historical data)

## What You're Currently MISSING

### 1. Player Ownership History

Currently, you CANNOT answer:
- "Who owned LeBron James in Week 5?"
- "When was Player X added/dropped?"
- "What was my roster composition on October 15th?"

**Why:** Roster snapshots are point-in-time only. Old snapshots are timestamped but not linked to specific weeks.

### 2. Games Played While Rostered vs Before

Currently, game logs don't show:
- Which team owned the player when they played that game
- Whether a player was rostered or on waivers

**Why:** Historical game logs are player-centric, not ownership-centric.

### 3. Dropped Player Data

If you drop a player and don't run the pipeline before dropping them, their data is lost from your system (but you can always re-add them and re-fetch).

## Recommended Running Schedule

### For Most Users: **Once Per Week**

Run on the **same day each week** (e.g., every Monday morning):

```bash
# Monday morning routine
python run_all.py
```

**Pros:**
- ✅ Captures weekly roster changes
- ✅ Gets all current week matchup data
- ✅ Minimal API calls
- ✅ All game logs stay current

**Cons:**
- ❌ Doesn't track mid-week roster changes
- ❌ Matchup data may be stale by end of week

### For Active Managers: **Twice Per Week**

Run at **start of week** and **mid-week**:

```bash
# Monday: Week starts
python run_all.py

# Thursday: Mid-week check
python run_all.py --step 1  # Just update roster
python run_all.py --step 2  # Update game logs
python run_all.py --step 3  # Get updated matchups
```

### For Maximum Data: **Daily**

If you want complete roster history:

```bash
# Run every day
python run_all.py
```

This creates a daily snapshot history, allowing you to reconstruct roster changes over time.

## What Happens If You Miss Running It

### If you miss 1 week:
- **Lost:** Roster snapshot from that week
- **Lost:** Games played by players who were dropped during that week
- **Kept:** All games for currently rostered players (even games from the missed week)

### If you miss 1 month:
- **Lost:** All roster snapshots from that month
- **Lost:** Games for players who were dropped and not re-added
- **Kept:** All games for currently rostered players (full season history)

### If you miss the entire season start:
- **Kept:** Everything! When you finally run it, it fetches:
  - Current roster
  - All 2025-26 games for current players (from season start to now)
  - All historical seasons (2024-25, 2023-24, 2022-23)

**The key insight:** Game logs are pulled from NBA API based on the current roster, not based on ownership history. So you get all games for a player's season, regardless of when they joined your team.

## Automating the Pipeline

To never forget, set up a cron job:

```bash
# Run every Monday at 8 AM
0 8 * * 1 cd /Users/rhu/fantasybasketball2/fantasy_2026 && python run_all.py >> logs/weekly_$(date +\%Y\%m\%d).log 2>&1
```

Or use a simple scheduler script:

```bash
# Add to your shell's rc file (.bashrc, .zshrc)
alias fantasy-update='cd /Users/rhu/fantasybasketball2/fantasy_2026 && python run_all.py'
```

## Future Enhancement Ideas

If you want to track ownership history, you would need to add:

1. **Ownership Tracking Table**
   - Columns: player_name, team_name, start_date, end_date, is_current
   - Updated with each roster snapshot
   - Tracks adds/drops over time

2. **Game-to-Ownership Linkage**
   - Join game logs with ownership tracking
   - Filter to "games while rostered"

3. **Historical Roster Reconstruction**
   - Query: "Show me my Week 5 roster"
   - Uses timestamped snapshots + ownership history

These features could be added as a new script: `5_track_ownership_history.py`

## Summary

| Frequency | Use Case | Pros | Cons |
|-----------|----------|------|------|
| **Daily** | Active management, full history | Complete snapshots | Many API calls |
| **Weekly** | Normal use (recommended) | Good balance | Some mid-week data loss |
| **Bi-weekly** | Casual use | Minimal effort | Miss roster changes |
| **As-needed** | Data analysis only | No overhead | Historical gaps |

**Bottom Line:** Running once a week is sufficient for most use cases. You won't lose game log data even if you miss weeks, but you'll lose roster composition history for dropped players.
