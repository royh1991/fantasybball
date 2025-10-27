# H-Scoring Quick Start Guide

## Installation (2 minutes)

```bash
cd h_scoring

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Test installation
python test_installation.py
```

## Usage (5 minutes to first recommendation)

### Option 1: Interactive Draft Assistant

```bash
python draft_assistant.py
```

This will:
1. Collect NBA data (5-10 min first time, then cached)
2. Show G-score rankings
3. Enter interactive draft mode

Commands:
- `rec` - Get H-score recommendations
- `draft <player_name>` - Draft a player
- `team` - Show your team
- `quit` - Exit

### Option 2: Run Example

```bash
# Full draft simulation
python example_usage.py full

# Single pick analysis
python example_usage.py single

# Strategy comparison
python example_usage.py compare
```

### Option 3: Python Script

```python
from draft_assistant import DraftAssistant

# Initialize
assistant = DraftAssistant()

# Collect data (first time only - takes 5-10 min)
assistant.collect_data(seasons=['2023-24'], max_players=150)

# Get recommendations
recommendations = assistant.recommend_pick(top_n=10)
print(recommendations)

# Draft a player
assistant.draft_player('Nikola Jokic')

# Get next recommendations (adapts to your team!)
recommendations = assistant.recommend_pick(top_n=10)

# Show team
assistant.show_team_summary()

# Export results
assistant.export_results()
```

## What Makes H-Scoring Different?

**Static Rankings (G-score):**
```
Pick 1: Nikola Jokic
Pick 2: Giannis Antetokounmpo
Pick 3: Joel Embiid
Pick 4: Domantas Sabonis
```
Always recommends the same players regardless of your team.

**Dynamic H-Scoring:**
```
Pick 1: Nikola Jokic
Pick 2: Giannis Antetokounmpo
Pick 3: Stephen Curry  (adapts: you need guards now!)
Pick 4: Tyrese Haliburton  (complements your team)
```
Recommends based on your team composition.

## Key Features

1. **Auto-Punt Discovery**: No need to decide "I'm punting FT%" beforehand
2. **Correlation Aware**: Knows REB-BLK correlate, AST-TOV correlate
3. **Values Consistency**: Low variance players valued higher
4. **Real-time Adaptation**: Each pick adapts to your team
5. **Mathematically Optimal**: Maximizes expected win probability

## Example Output

```
Pick #3 RECOMMENDATIONS
============================================================

Top Recommendations:
 RANK  PLAYER_NAME             H_SCORE   G_SCORE
    1  Stephen Curry              8.47      8.23
    2  Tyrese Haliburton         8.31      8.01
    3  Damian Lillard            8.15      7.87

Emerging Strategy:
----------------------------------------
Punting: TOV, FT%
Strong in: REB, BLK, FG%, DD
----------------------------------------
```

## Troubleshooting

**"ModuleNotFoundError: No module named 'nba_api'"**
```bash
pip install -r requirements.txt
```

**"No data found"**
```bash
python draft_assistant.py
# It will collect data automatically on first run
```

**Data collection is slow**
- Normal! Takes 5-10 minutes for 150 players
- Data is cached, only needed once per season
- Reduce max_players if needed: `assistant.collect_data(max_players=100)`

**"RateLimitError"**
- NBA API has rate limits
- Script includes delays (0.6s per player)
- If it fails, just run again - it will resume

## Next Steps

1. Read [README.md](README.md) for full documentation
2. See [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md) for technical details
3. Check [example_usage.py](example_usage.py) for more examples
4. Customize parameters in `h_optimizer.py` (omega, gamma, learning_rate)

## Support

This is an educational implementation of Rosenof (2024).

Paper: https://arxiv.org/abs/2409.09884

For bugs or questions, refer to the implementation notes.