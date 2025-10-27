# H-Scoring Algorithm Summary (Rosenof 2024)

## Overview

H-scoring is a **dynamic player valuation method** for fantasy basketball drafts that adjusts player values based on your current team composition, remaining picks, and opponent projections.

Unlike static rankings (ADP, projections), H-scores **change throughout the draft** to optimize your team's expected category wins.

---

## Core Concepts

### 1. **G-Scores (Static Player Rankings)**

**Formula:**
```
G_i = (μ_player - μ_league) / sqrt(σ²_between + κ × σ²_within_player)
```

Where:
- `μ_player` = player's season average
- `μ_league` = league average
- `σ²_between` = variance across all players (talent differences)
- `σ²_within` = player's week-to-week variance (consistency)
- `κ = (2N)/(2N-1)` where N = roster size (13)

**Purpose:** Variance-adjusted rankings that penalize inconsistent players

**Example:**
- Sabonis G-score: 7.78 (elite, consistent)
- KD G-score: 4.94 (good, but less value than raw stats suggest)

---

### 2. **X-Scores (Optimization Basis)**

**Formula:**
```
X_i = (μ_player - μ_league) / σ_within_player
```

For percentages (FG%, FT%, 3P%):
```
X_i = (attempts_player / attempts_league) × (pct_player - pct_league) / σ_within
```

**Purpose:** Simplified scores for mathematical optimization
- Removes between-player variance (not needed for optimization)
- Volume-weighted for shooting percentages
- Used in team projections and H-score calculations

**Example:**
- Sabonis DD X-score: 7.50 (4.79 std devs above average)
- KD DD X-score: -0.16 (below average)

---

### 3. **Baseline Category Weights**

**Method:** Coefficient of Variation (CV = σ/μ)

**Rationale:** Categories with high variance relative to their mean are more "scarce" and valuable

**Actual weights:**
```
DD:      22.6%  (highest - most scarce)
BLK:     14.6%
AST:     10.8%
FG3M:    10.4%
REB:      8.4%
TOV:      8.3%
PTS:      7.2%
STL:      6.1%
FG3_PCT:  6.0%
FT_PCT:   3.1%
FG_PCT:   2.5%  (lowest - least scarce)
```

---

### 4. **X_delta (Future Picks Modeling)**

**Purpose:** Estimate expected stats from your remaining draft picks

**Formula:** Complex constraint optimization solving:
```
Minimize: || X_delta ||²
Subject to:
  1. v^T @ X_delta = -γσ  (total value slightly below average)
  2. jC^T @ X_delta = ωσ  (aligned with your category priorities)
```

Where:
- `γ = 0.25` = generic value penalty
- `ω = 0.7` = category alignment strength
- `jC` = current category weights
- `v` = vector converting X-scores to G-scores

**Multiplier:** `0.25 × remaining_picks`

**Example on pick #1:**
- 12 remaining picks
- DD weight = 22.6% (highest)
- **X_delta predicts 5.57 DD** from future picks
- That's 0.46 DD per pick

**Key assumption:** Your future picks will align with your category weight priorities

---

### 5. **H-Score Calculation**

**Full Process:**

1. **Calculate team projection:**
   ```
   Team_X = Current_Team_X + Candidate_X + X_delta
   ```

2. **Calculate opponent projection:**
   - Paper version: `Opponent_X = picks_made × avg_player_profile`
   - Simple assumption: average opponent with same number of picks

3. **Calculate win probabilities for each category:**
   ```
   z_i = (Team_X_i - Opponent_X_i) / sqrt(variance_i)
   P(win category i) = Φ(z_i)  [normal CDF]
   ```

4. **Optimize category weights using gradient descent:**
   - **Objective:** Maximize `sum(P(win category i))`
   - **Method:** Adam optimizer
   - **Iterations:** Up to 100
   - **Starting point:** Baseline weights (or previous optimal weights)

5. **H-score = Maximum achievable objective value**

**Example:**
- KD H-score: 6.29 (expected to win ~6.3 categories)
- Sabonis H-score: 6.20 (expected to win ~6.2 categories)

---

### 6. **Emergent Punting Strategy**

**Key insight:** The optimizer naturally learns to "punt" (give up on) weak categories

**How it works:**
- Gradient descent adjusts weights to maximize total win probability
- Better to win 8 categories at 95% than 11 categories at 60%
- Punting emerges WITHOUT being explicitly programmed

**Example:** After drafting SGA + Harden:
- **Strong:** FT%, PTS, AST, STL → optimizer increases these weights
- **Weak:** REB, TOV, DD → optimizer decreases these weights
- **Result:** Prefer KD (reinforces strengths) over KAT (helps weaknesses)

---

## Algorithm Flow

### During the Draft:

**For each available player:**

1. Calculate candidate's X-scores
2. Calculate current team's total X-scores
3. Estimate opponent X-scores (using simple average model)
4. Calculate X_delta for remaining picks
5. Initialize weights (baseline for pick #1, previous optimal for later picks)
6. **Optimize weights via gradient descent:**
   - Project team stats: `Team = Current + Candidate + X_delta`
   - Calculate win probabilities for each category
   - Adjust weights to maximize total win probability
7. H-score = optimal objective value
8. Rank all candidates by H-score
9. Draft highest H-score player

---

## Key Design Choices

### 1. **Static Baseline Weights**
- No team-aware adjustments to baseline
- Weights derived purely from league-wide scarcity
- Team context only affects the optimization, not the starting point

### 2. **Absolute Win Probability Objective**
- Maximize total expected categories won
- NOT marginal contribution or improvement
- Naturally favors balanced rosters that win many categories

### 3. **Simple Opponent Modeling**
- Paper (lines 83-84): "Fill unknown picks with average G-score"
- `Opponent_X = picks_made × average_player_X_scores`
- Simpler than complex snake draft simulation

### 4. **No Hard-Coded Punting Logic**
- Punting emerges from optimization
- Algorithm discovers optimal strategy through math
- More flexible than pre-defined punt strategies

---

## Theoretical Advantages

1. **Dynamic vs Static:**
   - ADP/projections: Same ranking for everyone
   - H-scoring: Rankings change based on YOUR team

2. **Team Context:**
   - Recognizes when you need rebounds vs assists
   - Adapts to your emerging strengths/weaknesses

3. **Category Synergy:**
   - Accounts for correlations (REB + BLK often come together)
   - Uses covariance matrix in optimization

4. **Future-Aware:**
   - X_delta models what you'll draft later
   - Prevents over-investing in one category early

5. **Mathematically Optimal:**
   - Gradient descent finds best possible weights
   - Not based on heuristics or gut feeling

---

## Implementation Details

### Parameters:
- `γ = 0.25` (generic value penalty)
- `ω = 0.7` (category alignment strength)
- `ridge_frac = 1e-8` (numerical stability for covariance matrix)
- `learning_rate = 0.01` (Adam optimizer)
- `max_iterations = 100` (gradient descent)

### Data Requirements:
- Weekly player statistics (PTS, REB, AST, STL, BLK, TOV, FG3M, DD, FG%, FT%, 3P%)
- Per-game variance for each player
- Covariance matrix across all categories

### Performance:
- Paper claims H-scoring outperforms:
  - ADP-based drafting
  - G-score rankings
  - Traditional "balanced roster" strategies
- Simulations show ~73-88% win rates vs ~50% for ADP teams

---

## Key Formulas Summary

**G-score:**
```
G = (μ - μ_league) / sqrt(σ²_between + κ × σ²_within)
```

**X-score:**
```
X = (μ - μ_league) / σ_within
```

**Team Projection:**
```
Team_X = Current_X + Candidate_X + X_delta
```

**Win Probability:**
```
P(win) = Φ((My_Team - Opponent) / σ_category)
```

**H-score:**
```
H = max_{weights} Σ P(win category_i | weights)
```

---

## Critical Issues We Discovered

### 1. **Scarcity Paradox (Major Flaw)**

**Problem:** X_delta assumes elite players in scarce categories will be available throughout the draft

**Reality:** In snake drafts, if DD is scarce (22.6% weight), OTHER teams also draft DD players early

**Impact:**
- Pick #6: "Pass on Sabonis, I'll get DD later"
- Pick #19: All elite DD players gone (Sabonis, Jokic, AD, Giannis)
- X_delta's promise CANNOT be fulfilled

**Root Cause:** Algorithm doesn't account for:
- Snake draft position (long gaps between picks)
- Scarcity depletion (other teams taking scarce resources)
- Competitive drafting dynamics

### 2. **X_delta Applied Uniformly**

**Problem:** Same X_delta for ALL candidates on the same pick

**Should be:**
- Draft weak DD player → X_delta should predict high DD from future picks
- Draft strong DD player → X_delta should predict low DD from future picks

**Actual:**
- Both get +5.57 DD from X_delta
- This dilutes the advantage of specialists like Sabonis

### 3. **Baseline Weight Optimization on Pick #1**

**Problem:** On pick #1 (no team context), gradient descent still moves away from baseline weights

**Result:**
- DD: 22.6% → 10.9% (dropped 12%)
- FT_PCT: 3.1% → 9.8% (increased 7%)
- This happens BEFORE you draft anyone!

**Should be:** Use pure baseline weights on pick #1, only optimize after you have team context

### 4. **Specialists vs Balanced Players**

**Problem:** Algorithm systematically favors balanced players over category specialists

**Why:**
- Sabonis: Elite in DD (7.50), weak in FG3M (-4.07) → polarized
- KD: Good everywhere, no major weaknesses → balanced
- Win probability function rewards 0.7 across many categories over 0.95 in one

**Impact:**
- KD H-score: 6.29 (recommended)
- Sabonis H-score: 6.20 (despite 2.84 higher G-score!)

### 5. **Category Variance Dilution**

**Problem:** High category variances (like DD = 130.75) dilute specialist advantages

**Example:**
- Sabonis DD advantage: 7.50 vs KD's -0.16 = 7.66 difference
- But z-score = 7.66 / 11.43 = only 0.67 std devs
- Win prob only increases from 68% to 87%

---

## Paper Citation

**Rosenof, A. (2024). "H-scoring: A Dynamic Valuation Method for Fantasy Sports Drafts"**

Key sections:
- Lines 83-84: Opponent modeling
- Lines 229-243: Objective function definition
- Section 4.1: Static weight calculation
- Appendix C: Punting strategies

---

## Bottom Line

**What the paper tries to do:**
- Create a mathematically optimal draft strategy
- Account for team context, future picks, and category correlations
- Automatically discover punting strategies through optimization

**What works:**
- G-scores correctly value consistency
- Gradient descent finds optimal category weights
- Punting emerges naturally from the math
- Better than pure ADP in simulations

**Critical flaw we found:**
- **Scarcity paradox:** X_delta creates false availability assumptions
- Systematically undervalues category specialists
- Doesn't account for snake draft dynamics
- In real drafts, passing on elite specialists (Sabonis) leaves you unable to fulfill X_delta's promises

**For your draft:**
- The algorithm will recommend KD > Sabonis
- But by your 2nd pick, all elite DD players will be gone
- Trust G-scores or ADP over H-scores for early-round specialists
- H-scoring may work better in later rounds or for tie-breaking similar players
