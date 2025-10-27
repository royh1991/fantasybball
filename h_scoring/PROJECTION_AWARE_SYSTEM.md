# Projection-Aware H-Scoring System

## Overview

The projection-aware system blends **historical performance data** with **expert projections** to create more accurate player valuations that account for:

1. **Player development** (young players improving)
2. **Age regression** (veterans declining)
3. **Injury risk** (availability adjustments)

---

## Key Results

### Impact on Rankings (Top 7 Test Players)

| Rank | Player | Original H-score | Projection-Aware | Change | Why? |
|------|--------|-----------------|------------------|--------|------|
| 1 | **Sabonis** | 0.6408 | **0.6585** | +0.0177 | Durable + strong projections |
| 2 | **LeBron** | 0.6358 | **0.6550** | +0.0192 | Age adjusted but still elite |
| 3 | **KAT** | 0.6258 | **0.6544** | +0.0286 | Better projections than history |
| 4 | **Cade** | 0.6181 | **0.6485** | **+0.0304** | **Young + improving** |
| 5 | Kawhi | 0.6179 | 0.6141 | -0.0037 | Injury risk penalty |
| 6 | **Embiid** | 0.6459 | 0.6023 | **-0.0437** | **Only 37 GP projected!** |
| 7 | **PG** | 0.6043 | 0.5818 | **-0.0226** | **44 GP projected** |

### Cade Cunningham Specifically

**Original System:** Ranked #5
**Projection-Aware:** Ranked **#4** (+1 spot)

**Why Cade Improves:**
- 3 seasons of data → 50% projection weight (balanced)
- Projections show improvement across all categories
- Biggest gains: PTS (+2.4), AST (+1.4), DD (+0.15)
- **H-score boost: +0.0304** (3rd largest improvement)

**Injury Risk:** 70/82 GP projected → No penalty (85% availability)

---

## How It Works

### 1. Projection Weighting by Experience

The system adjusts how much to trust projections vs history based on player experience:

| Experience | Projection Weight | History Weight | Rationale |
|------------|------------------|----------------|-----------|
| 1-2 seasons | **70%** | 30% | Rookies/sophomores - trust projections |
| 3 seasons | **50%** | 50% | Balanced - equal weight |
| 4-5 seasons | **40%** | 60% | Established - trust history more |
| 6+ seasons | **30%** | 70% | Veterans - history matters most |

**Why this makes sense:**
- Young players (Cade, Scottie Barnes) are still improving → projections capture upside
- Veterans (LeBron, CP3) are more predictable → history is reliable
- Prevents over-trusting rookie hype or ignoring vet decline

### 2. Injury Risk Adjustment

Uses projected Games Played (GP) to adjust stats:

| GP Projected | Availability | Risk Factor | Impact |
|--------------|--------------|-------------|---------|
| 70+ games | 85%+ | **1.00x** | No penalty |
| 60-70 games | 73-85% | **0.90-1.00x** | Mild penalty |
| 50-60 games | 61-73% | **0.75-0.90x** | Moderate penalty |
| <50 games | <61% | **0.50-0.75x** | Strong penalty |

**Examples:**
- **Sabonis: 76 GP → 1.00x** (iron man)
- **KAT: 69 GP → 0.99x** (slight concern)
- **Kawhi: 52 GP → 0.79x** (load management)
- **Embiid: 37 GP → 0.56x** (huge injury risk!)
- **PG: 44 GP → 0.67x** (major concern)

**Formula:**
```
adjusted_stats = blended_mean × risk_factor
```

### 3. Blending Formula

For each stat category:

```python
blended_mean = (projection_weight × projected_weekly) + ((1 - projection_weight) × historical_weekly)
```

Then apply injury adjustment:
```python
final_mean = blended_mean × injury_risk_factor
```

**Example (Cade PTS):**
```
historical_mean = 74.47 PTS/week
projected_mean = 79.19 PTS/week (from projections: 2373.2 PTS / (70 GP / 3 games/week))
projection_weight = 0.50 (3 seasons)

blended = 0.50 × 79.19 + 0.50 × 74.47 = 76.83 PTS/week
injury_adjusted = 76.83 × 1.00 = 76.83 PTS/week (no penalty for 70 GP)

Result: +2.37 PTS/week improvement over pure history!
```

---

## Cade Deep Dive

### X-Score Changes

| Category | Historical | Blended | Change | Impact |
|----------|-----------|---------|--------|--------|
| **PTS** | 1.20 | **1.30** | +0.10 | Improved scoring |
| **AST** | 1.83 | **2.00** | +0.17 | Better playmaking |
| **BLK** | -0.12 | **0.13** | +0.25 | Defense uptick |
| **DD** | 0.69 | **0.88** | +0.19 | More double-doubles |
| REB | -0.03 | 0.07 | +0.10 | Slight improvement |

**Total Impact:** +0.0304 H-score boost (moved from #5 → #4)

### Why Projections Help Cade

1. **Improving scorer:** 74.5 → 76.8 PTS/week
2. **Better playmaker:** 25.1 → 26.6 AST/week
3. **More complete:** 1.02 → 1.18 DD/week
4. **Only 3rd year:** Still has room to grow, projections capture this

---

## Impact on Injury-Prone Players

### Joel Embiid (-0.0437 H-score)

**Problem:** Only 37/82 GP projected (45% availability)

**Impact:**
- Risk factor: 0.56x (huge penalty)
- Every stat reduced by 44%
- PTS: 80.7 → 45.2 (historical → adjusted)
- REB: 35.6 → 19.9
- **Falls from #6 → #6 in our test** (but would fall further in full rankings)

**Is this fair?** YES!
- If Embiid only plays 37 games, he's worth much less in fantasy
- You can't win categories with an empty roster spot
- This is the same logic as "games played" in real fantasy scoring

### Kawhi Leonard (-0.0037 H-score)

**Problem:** Only 52/82 GP projected (63% availability)

**Impact:**
- Risk factor: 0.79x (moderate penalty)
- All stats reduced by 21%
- Slight H-score drop (not catastrophic, but noticeable)

---

## Usage

### Initialize the System

```python
from modules.scoring_with_projections import ProjectionAwareScoringSystem
from modules.covariance import CovarianceCalculator
from modules.h_optimizer_final import HScoreOptimizerFinal

# Load data
league_data = pd.read_csv('data/league_weekly_data_*.csv')
player_variances = json.load('data/player_variances_*.json')
projections_file = '../data/fantasy_basketball_clean2.csv'

# Create projection-aware scoring system
scoring = ProjectionAwareScoringSystem(
    league_data=league_data,
    player_variances=player_variances,
    roster_size=13,
    projections_file=projections_file,
    projection_weight=0.5,  # Base weight (adjusted per player)
    injury_penalty_strength=1.0  # 1.0 = moderate injury penalty
)

# Use with H-scoring optimizer as normal
cov_calc = CovarianceCalculator(league_data, scoring)
setup_params = cov_calc.get_setup_params()
optimizer = HScoreOptimizerFinal(setup_params, scoring, omega=0.7, gamma=0.25)

# Evaluate players
h_score, weights = optimizer.evaluate_player(
    "Cade Cunningham", my_team=[], opponent_teams=[[]],
    picks_made=0, total_picks=13
)
```

### Tuning Parameters

**projection_weight** (0-1):
- Higher = trust projections more
- Lower = trust history more
- Default: 0.5 (balanced, but auto-adjusted per player)

**injury_penalty_strength** (0-2):
- 0 = no injury penalty
- 1 = moderate penalty (recommended)
- 2 = aggressive penalty (harsh on injury-prone players)

**Experience-based adjustments:**
- Automatically applied based on seasons in dataset
- Young players (1-2 seasons): 70% projection weight
- Veterans (6+ seasons): 30% projection weight

---

## Advantages

1. **Young Player Upside:** Cade, Scottie Barnes, Chet benefit from projection-based improvements
2. **Injury Risk:** Embiid, Kawhi, PG properly penalized for low GP projections
3. **Balanced:** Veterans still valued based on proven track record
4. **Realistic:** Accounts for the fact that projections capture trends history can't

---

## Caveats

1. **Projection quality matters:** Garbage projections = garbage results
2. **Sample size:** Players with <3 seasons might have unreliable history
3. **Rookies:** Not in historical data - would need pure projection mode (not implemented)
4. **Mid-season trades:** Historical data may not reflect new role/team

---

## Comparison to Pure Historical

### Pure Historical System Issues:

1. **Ignores player development:** Cade's 3rd year breakout not captured
2. **Ignores age curves:** LeBron's decline not anticipated
3. **No injury risk:** Embiid valued as if he plays 75 games
4. **Recency bias:** Weighs 2022-23 equally with 2024-25

### Projection-Aware Advantages:

1. ✅ Captures expected improvement (Cade +0.0304)
2. ✅ Accounts for injury risk (Embiid -0.0437)
3. ✅ Balances history and projections intelligently
4. ✅ Experience-based weighting (rookies vs vets)

---

## Bottom Line

**For Cade:** The projection-aware system correctly values his improving trajectory, boosting him from #5 → #4.

**For injury-prone stars:** Embiid, Kawhi, PG are properly penalized for low GP projections.

**For established stars:** Sabonis, LeBron get slight boosts from combining history + projections.

**Net result:** More accurate draft valuations that reflect both proven performance AND expected changes.
