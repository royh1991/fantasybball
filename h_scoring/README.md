# H-Scoring Fantasy Basketball Draft Assistant

Implementation of the H-scoring algorithm from Rosenof (2024) for fantasy basketball draft optimization.

## Overview

H-scoring is a dynamic player valuation system that adapts to your team composition during the draft. Unlike static rankings, H-scoring:

- **Dynamically optimizes** category weights based on your team needs
- **Naturally discovers** punt strategies without explicit programming
- **Accounts for correlations** between statistical categories (e.g., AST-TOV, REB-BLK)
- **Models week-to-week variance** to value consistent players appropriately
- **Outperforms static rankings** by 2.6-4.5x in simulations

## Features

- **Data Collection**: Uses `nba_api` to fetch historical NBA game logs
- **G-Score Calculations**: Static player rankings with player-specific variance
- **Covariance Matrix**: Captures correlations between categories
- **Gradient Descent Optimization**: Optimizes category weights using Adam optimizer
- **Interactive Draft Assistant**: Real-time draft recommendations

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Collect Data

```python
from draft_assistant import DraftAssistant

assistant = DraftAssistant()
assistant.collect_data(seasons=['2023-24'], max_players=200)
```

This will:
- Fetch game logs from NBA API
- Aggregate to weekly statistics
- Calculate player-specific variance
- Save data to `data/` directory

### 2. Run Draft Assistant

```bash
python draft_assistant.py
```

Interactive commands:
- `rec` - Get H-score recommendations for current pick
- `draft <player_name>` - Draft a player
- `team` - Show your current team
- `quit` - Exit draft

### 3. Use Programmatically

```python
from draft_assistant import DraftAssistant

# Initialize with existing data
assistant = DraftAssistant(
    data_file='data/league_weekly_data_20240101_120000.csv',
    variance_file='data/player_variances_20240101_120000.json',
    format='each_category'  # or 'most_categories'
)

# Get recommendations
recommendations = assistant.recommend_pick(top_n=15)

# Draft a player
assistant.draft_player('LeBron James')

# Update opponent rosters (after each round)
opponent_rosters = [
    ['Stephen Curry', 'Kevin Durant'],
    ['Giannis Antetokounmpo', 'Luka Doncic'],
    # ... other teams
]
assistant.update_opponent_rosters(opponent_rosters)
```

## How H-Scoring Works

### 1. X-Score Calculation

X-scores normalize player statistics removing between-player variance:

```
X = (μ_player - μ_league) / σ_within
```

Where `σ_within` is the player's week-to-week variance (consistency).

### 2. Future Pick Adjustment (X_delta)

Models expected statistics from future picks based on optimized weights:

```
X_delta = f(weights, covariance_matrix, remaining_picks)
```

This captures correlations - if you weight REB highly, you'll naturally get more BLK.

### 3. Win Probability

For each category, calculate probability of winning:

```
P(win) = Φ((team_total - opponent_total) / σ)
```

Where Φ is the normal CDF.

### 4. Gradient Descent

Optimize weights to maximize total win probability:

```python
while not converged:
    gradient = ∂(sum(win_prob)) / ∂weights
    weights += learning_rate × gradient
```

Uses Adam optimizer for stable convergence.

### 5. Player Selection

Draft the player with highest optimized win probability.

## Module Structure

```
h_scoring/
├── modules/
│   ├── data_collector.py    # NBA API data collection
│   ├── scoring.py           # G-score and X-score calculations
│   ├── covariance.py        # Covariance matrix and baseline weights
│   └── h_optimizer.py       # H-score optimization with gradient descent
├── draft_assistant.py       # Main interactive draft tool
├── data/                    # Collected data (created on first run)
└── README.md
```

## Categories Supported

11-category leagues:
- **Counting Stats**: PTS, REB, AST, STL, BLK, TOV, 3PM, DD (double-doubles)
- **Percentages**: FG%, FT%, 3P%

## Algorithm Parameters

Tunable parameters in `HScoreOptimizer`:

- `omega` (default 0.7): Weight for weighted category strength
- `gamma` (default 0.25): Penalty for generic value sacrifice
- `learning_rate` (default 0.01): Gradient descent step size
- `roster_size` (default 13): League roster size

## Example Output

```
Pick #3 RECOMMENDATIONS
============================================================
Evaluating 100 players...

Top Recommendations:
 RANK  PLAYER_NAME             H_SCORE   G_SCORE
    1  Nikola Jokic              8.47      9.23
    2  Giannis Antetokounmpo     8.31      9.01
    3  Joel Embiid               8.15      8.87
    ...

Emerging Strategy:
----------------------------------------
Punting: TOV, FT%
Strong in: REB, BLK, FG%, DD
----------------------------------------
```

## Data Requirements

Minimum 20 weeks of data per player recommended for reliable variance estimates.

For best results:
- Use 2+ seasons of data
- Collect data for top 200 players
- Update data before each draft season

## Performance Notes

- Data collection: ~5-10 minutes for 200 players
- Covariance calculation: ~10 seconds
- H-score optimization per player: ~0.1-0.5 seconds
- Full pick recommendations (100 players): ~30-60 seconds

## References

Rosenof, Z. (2024). Dynamic quantification of player value for fantasy basketball. arXiv:2409.09884

## License

This implementation is for educational purposes. The H-scoring framework is described in the academic paper cited above.