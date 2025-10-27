# Quick Start Guide

Get up and running with the Fantasy Basketball Modeling System in 5 minutes.

## Step 1: Install Dependencies

```bash
cd fantasy_modeling
pip install -r requirements.txt
```

## Step 2: Verify Data Access

The system automatically uses your existing data files. Check that they exist:

```bash
ls -lh ../data/static/active_players_historical_game_logs.csv
ls -lh ../data/fantasy_basketball_clean2.csv
```

If these files exist, you're ready to go!

## Step 3: Run Example Script

```bash
python example_usage.py
```

This will:
- Load your historical data
- Fit a Bayesian model for a player
- Run a game simulation
- Demonstrate fantasy scoring
- Compare two players

## Step 4: Try a Single Player Simulation

Pick a player and simulate their next game:

```bash
python main.py simulate --player "luka_doncic" --n-sims 1000
```

Output will show:
- Projected stats with uncertainty (mean ± std)
- Full distribution percentiles
- Fantasy value across all 11 categories
- Results saved to `simulation_results.json`

## Step 5: Fit Models for Your Entire League

```bash
python main.py fit --min-games 10 --output fitted_models.json
```

This takes ~2 minutes and creates fitted Bayesian models for all players with sufficient data.

## Common Commands

### Simulate a game
```bash
python main.py simulate \
  --player "nikola_jokic" \
  --opponent "LAL" \
  --n-sims 1000 \
  --output jokic_projection.json
```

### Compare two players
```bash
python main.py compare \
  --player1 "stephen_curry" \
  --player2 "damian_lillard"
```

### Batch simulate (e.g., tonight's slate)
```bash
# First create a slate JSON file: tonight_slate.json
# [
#   {"player_id": "lebron_james", "position": "SF", "team": "LAL", "opponent": "GSW", "is_home": true},
#   {"player_id": "stephen_curry", "position": "PG", "team": "GSW", "opponent": "LAL", "is_home": false},
#   ...
# ]

python main.py batch \
  --slate tonight_slate.json \
  --n-sims 1000 \
  --output tonight_projections.csv
```

## Understanding the Output

### Simulation Results

```json
{
  "player_id": "luka_doncic",
  "projections": {
    "pts": {
      "mean": 28.45,
      "median": 28.0,
      "std": 6.23,
      "min": 12,
      "max": 48
    },
    ...
  },
  "percentiles": {
    "pts": {
      "10": 20,
      "25": 24,
      "50": 28,
      "75": 33,
      "90": 37
    }
  },
  "fantasy": {
    "11cat": {
      "pts": 28.45,
      "reb": 8.12,
      ...
    }
  }
}
```

**Key metrics:**
- `mean`: Expected value (use this for projections)
- `std`: Uncertainty (higher = more volatile)
- `percentiles`: Use `10` and `90` for floor/ceiling
- `fantasy.11cat`: Your league's specific stats

### Fantasy Scoring

For head-to-head matchups, the system shows:

```
Matchup Result: Team A
Score: 7-4-0

Category Breakdown:
  FG%     : A=0.467  B=0.479  [B]
  FT%     : A=0.812  B=0.795  [A]
  3P%     : A=0.361  B=0.348  [A]
  3PM     : A=95.0   B=88.0   [A]
  PTS     : A=850.0  B=820.0  [A]
  ...
```

Team A wins 7 categories, Team B wins 4, 0 ties.

## Customization

### Adjust Model Parameters

Edit `config/model_config.yaml`:

```yaml
empirical_bayes:
  recency:
    games_to_consider: 10      # Use last N games
    decay_factor: 0.9          # Weight recent games more

counting_stats:
  negbin_cv_threshold: 1.5     # When to use NegBin vs Poisson
```

### Change League Settings

Edit `config/league_config.yaml` to match your league's categories and roster settings.

### Add ESPN Authentication

```bash
cp config/credentials.env.example config/credentials.env
# Edit credentials.env with your ESPN cookies
```

Get your cookies:
1. Log into ESPN Fantasy Basketball
2. Open browser DevTools (F12)
3. Go to Application > Cookies
4. Find `swid` and `espn_s2`
5. Copy values to `credentials.env`

## Python Integration

Use the system in your own scripts:

```python
from simulation.game_simulator import GameSimulator
from models.bayesian_model import PlayerContext

# Initialize
simulator = GameSimulator(config_path="config")

# Load and fit models (do this once)
from data_pipeline.data_collector import DataCollector
collector = DataCollector()
player_info, game_logs, projections = collector.prepare_modeling_data()
simulator.fit_all_players(player_info, game_logs, projections)

# Simulate a game
context = PlayerContext(
    player_id="luka_doncic",
    position="PG",
    team="DAL",
    opponent="LAL",
    is_home=True
)

results = simulator.simulate_game("luka_doncic", context, n_simulations=1000)
print(f"Expected points: {results['projections']['pts']['mean']:.1f}")
```

## Troubleshooting

### "No data found"
Make sure you're in the `fantasy_modeling/` directory and your data files are in `../data/`.

### "Player not found"
Player IDs use lowercase with underscores: `"lebron_james"` not `"LeBron James"`.

Check available players:
```python
from data_pipeline.data_collector import DataCollector
collector = DataCollector()
mapping = collector.get_player_mapping()
print(list(mapping.keys())[:10])  # Show first 10 players
```

### Import errors
Make sure you've installed requirements:
```bash
pip install -r requirements.txt
```

## Next Steps

1. **Validate on last season**: Run backtesting to check accuracy
2. **Integrate with h_scoring**: Use projections as input to draft optimizer
3. **Automate daily updates**: Set up cron job to fetch and simulate daily
4. **Build visualizations**: Create charts in `notebooks/`

## Support

For issues or questions, check:
- Full documentation: `README.md`
- Example code: `example_usage.py`
- Model details: `models/` directory

---

You're all set! Start simulating games and optimizing your lineup.