# Interactive Draft Assistant - Quick Start Guide

## What This Does

Production-ready CLI tool for your actual fantasy basketball draft. Shows H-score recommendations when it's your turn, and tracks all picks from all teams.

## Your League Settings

- **14 teams**, 13 roster spots
- **Your position**: 5 (team name: "my team")
- **Snake draft** (1→14, then 14→1, etc.)

### Team Order:
1. coolest
2. hardowod
3. jay stat
4. team boruica
5. **my team** ← YOU
6. Nadim
7. team perez
8. enter the dragon
9. LF da brocoli
10. team too
11. retutrn burito
12. team menyo
13. mamba
14. bde

---

## How to Use

### 1. Start the draft:
```bash
cd /Users/rhu/fantasybasketball2/h_scoring
python draft_assistant_cli.py
```

### 2. For each pick:

**When it's someone else's turn:**
- Shows their current roster
- Shows top 10 available by ADP
- Prompts: "Who did [team name] draft?"
- Type the player name → Enter
- Confirm → Pick is recorded

**When it's YOUR turn:**
- Shows your current roster
- Calculates H-scores for top 60 available
- Shows TOP 10 RECOMMENDATIONS with H-scores
- Shows category analysis for #1 recommendation
- Prompts: "Enter player you're drafting:"
- Type the player name → Enter
- Confirm → Pick is recorded

### 3. Example interaction:

```
════════════════════════════════════════════════════════════════════════════════
Pick 5 (Round 1)
════════════════════════════════════════════════════════════════════════════════

════════════════════════════════════════════════════════════════════════════════
★ YOUR TURN ★
════════════════════════════════════════════════════════════════════════════════

【 YOUR CURRENT ROSTER (0/13) 】
  (No players yet)

⚙️  Calculating H-scores for top 60 available players...

════════════════════════════════════════════════════════════════════════════════
📊 TOP 10 H-SCORE RECOMMENDATIONS
════════════════════════════════════════════════════════════════════════════════

Rank   Player                         Pos        ADP      H-Score
────────────────────────────────────────────────────────────────────────────────
1      Anthony Davis                  PF,C       9.7      0.6234
2      Domantas Sabonis               C          10.1     0.6189
3      Victor Wembanyama              C          2.8      0.6156
...

────────────────────────────────────────────────────────────────────────────────
📈 TOP RECOMMENDATION ANALYSIS: Anthony Davis
────────────────────────────────────────────────────────────────────────────────

Category contributions (X-Scores):
  BLK       :   2.23 ▲
  REB       :   1.57 ▲
  DD        :   1.48 ▲
  PTS       :   0.91 ▲
  FT_PCT    :   0.86 ▲
  STL       :   0.37 ▲
  FG3M      :  -1.00 ▼

════════════════════════════════════════════════════════════════════════════════
Enter player you're drafting: Anthony Davis

✓ Found: Anthony Davis (PF,C) - ADP 9.7
Confirm this pick? (y/n): y

✓ DRAFTED: my team → Anthony Davis (ADP 9.7)
```

---

## Features

### ✅ H-Score Recommendations
- Top 10 picks ranked by H-score (adaptive to your team)
- Category analysis for top recommendation
- Shows position and ADP for context

### ✅ Auto-Save
- Every pick is auto-saved to `draft_session_YYYYMMDD_HHMMSS.json`
- If interrupted (Ctrl+C), you can resume from saved file
- Never lose your draft progress

### ✅ Smart Name Matching
- Fuzzy player name matching
- Handles partial names ("jokic" → "Nikola Jokic")
- Shows position and ADP for confirmation

### ✅ Team Context
- Always shows current roster before each pick
- Calculates optimal weights based on your team composition
- Updates opponent modeling with all teams' rosters

### ✅ ADP Reference
- Shows top 10 available by ADP for other teams
- Helps you track value and predict picks
- Displays ADP for every pick

---

## Tips for Using During Draft

### Before Draft Starts:
1. Open terminal
2. Navigate to h_scoring folder
3. Run `python draft_assistant_cli.py`
4. Have second screen/window for team rosters

### During Draft:
1. **For speed**: Type just last name (algorithm will find match)
2. **For accuracy**: If multiple players with same last name, use full name
3. **Stay focused**: The algorithm adapts to each pick, so keep it updated
4. **Trust the H-score**: It's optimizing for your specific team needs

### After Each Round:
- Review your roster shown at the top
- Check if you're building a punt strategy
- See how opponents are drafting

### If Something Goes Wrong:
- Press `Ctrl+C` to exit
- Draft is saved automatically
- File location shown on startup

---

## Understanding H-Scores

### What the numbers mean:
- **Higher = Better** for your specific team at this moment
- **Not static**: Changes as you draft
- **Context-aware**: After 2 centers, guards get higher scores

### Example:
```
After drafting AD + Chet (2 centers):

Player              H-Score    Why?
─────────────────────────────────────────────
Jamal Murray        0.5411     ✓ Fills AST/3PM gaps
LaMelo Ball         0.5409     ✓ Elite guard, balances roster
Nikola Vucevic      0.5330     ✗ 3rd center, diminishing returns
```

### H-Score vs ADP:
- **ADP**: Static ranking (average draft position)
- **H-Score**: Dynamic value (adapts to YOUR team)
- **Use both**: H-score for value, ADP to predict opponents

---

## Common Issues

### "Player not found in ADP list"
**Solution**: Type full name or check spelling. Algorithm will still draft them, just won't show ADP.

### "Wrong player selected"
**Solution**: After typing name, you'll see confirmation. Type 'n' to re-enter.

### "Draft interrupted"
**Solution**: Progress is auto-saved. Check the .json file in the folder.

### "Algorithm recommending weird picks"
**Possible reasons**:
1. Punt strategy emerging (check category X-scores)
2. Opponents drafted key players in your target categories
3. Deep in draft (late rounds favor role players)

---

## Files Generated

### During Draft:
- `draft_session_YYYYMMDD_HHMMSS.json` - Auto-save file with all picks

### After Draft:
- Complete roster for all 14 teams
- Your final team with positions and ADPs
- Can be used for season simulation

---

## Quick Reference

### Commands:
```bash
# Start draft
python draft_assistant_cli.py

# Exit draft (saves automatically)
Ctrl+C
```

### When entering player names:
- ✅ "jokic" → Works (fuzzy match)
- ✅ "Nikola Jokic" → Works (exact match)
- ✅ "Luka" → Works (if unambiguous)
- ❌ "luka doncic jones" → Won't work (too many words)

### Confirmation:
- Type `y` to confirm
- Type `n` to re-enter

---

## Good Luck! 🏀

The algorithm has been optimized to:
- ✅ Avoid stacking saturated categories
- ✅ Fill team gaps intelligently
- ✅ Apply diminishing returns
- ✅ Adapt to your team composition
- ✅ Filter players with no data

Trust the H-scores, but use your judgment too. The algorithm is a tool, not a replacement for your basketball knowledge!
