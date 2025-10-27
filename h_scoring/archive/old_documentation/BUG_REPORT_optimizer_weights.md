# BUG REPORT: H-Score Optimizer Not Properly Weighting Team Needs

**Date:** 2025-10-01
**Severity:** HIGH - Core algorithm issue
**Status:** CONFIRMED

---

## Executive Summary

The H-score optimizer is **not properly adjusting category weights based on team composition**. When you desperately need rebounds (0.06 X-score), it assigns only 7.9% weight to REB. Meanwhile, when you're already elite in FT% (6.99 X-score), it assigns 19% weight to FT% - the **highest weight** of all categories.

This causes the algorithm to prefer well-rounded players (like Kevin Durant) over specialists who fill specific needs (like Karl-Anthony Towns), even when the specialist addresses critical weaknesses.

---

## Evidence

### Scenario: After Drafting SGA + Harden

**Team Status:**
- PTS: 9.00 X-score (💪 Strong)
- AST: 9.28 X-score (💪 Strong)
- STL: 3.09 X-score (💪 Strong)
- **FT%: 6.99 X-score (💪 ELITE)**
- **REB: 0.06 X-score (⚠️ CRITICAL WEAKNESS)**
- FG%: -0.02 X-score (⚠️ Weak)
- FG3%: 0.65 X-score (⚠️ Weak)
- TOV: -4.28 X-score (⚠️ Weak)

**Optimizer's Chosen Weights (for KD evaluation):**
```
FT_PCT:  0.1899  (19.0% - HIGHEST WEIGHT!)
BLK:     0.1402  (14.0%)
FG3M:    0.1192  (12.0%)
DD:      0.1180  (11.8%)
STL:     0.1114  (11.1%)
FG_PCT:  0.0859  (8.6%)
REB:     0.0789  (7.9% - VERY LOW for critical need!)
FG3_PCT: 0.0790  (7.9%)
AST:     0.0373  (3.7%)
PTS:     0.0237  (2.4%)
TOV:     0.0166  (1.7%)
```

### The Contradiction

**What the weights SHOULD prioritize:**
1. **REB** (0.06 → desperately need)
2. **FG%** (-0.02 → weak)
3. **FG3%** (0.65 → weak)
4. **TOV** (-4.28 → terrible)

**What the weights ACTUALLY prioritize:**
1. **FT%** (6.99 → already elite, don't need!)
2. **BLK** (1.29 → average, okay to add)
3. **FG3M** (1.93 → average, okay to add)

### The Impact

**KAT vs KD after SGA + Harden:**

| Category | KAT Better? | KD Wins Because |
|----------|-------------|-----------------|
| REB | ✓ +2.10 X-score advantage | Only 7.9% weight (0.116 contribution) |
| DD | ✓ +2.57 X-score advantage | Only 11.8% weight (0.374 contribution) |
| TOV | ✓ +0.69 X-score advantage | Only 1.7% weight (0.006 contribution) |
| **Total KAT advantages** | **+0.496** | |
| FT% | KD +0.60 X-score | **19% weight** (0.143 contribution) |
| BLK | KD +1.83 X-score | 14% weight (0.256 contribution) |
| PTS | KD +2.85 X-score | 2.4% weight (0.089 contribution) |
| + 5 other small edges | | (0.236 contribution) |
| **Total KD advantages** | **+0.724** | |

**KD wins by: +0.228**

Despite KAT filling critical team needs better, KD wins because the weights favor categories you don't need.

---

## Root Cause Analysis

### Issue #1: Baseline Weights Don't Consider Team

**Location:** `modules/covariance.py:calculate_baseline_weights()`

Baseline weights are calculated using "scarcity" (coefficient of variation) across **all players in the league**, not based on your specific team needs:

```python
def calculate_baseline_weights(self, method='scarcity'):
    if method == 'scarcity':
        # Weight by coefficient of variation (higher CV = more scarce = more valuable)
        cv_values = []
        for cat in self.categories:
            cat_values = self.season_averages[cat].values
            mean_val = np.mean(cat_values)
            std_val = np.std(cat_values)
            if mean_val > 0:
                cv = std_val / mean_val
            else:
                cv = 0
            cv_values.append(cv)
```

**Problem:** This is a **static measure** - it's the same for everyone regardless of team composition. If FT% has high scarcity in the league, everyone gets high FT% weight, even if they're already elite at FT%.

### Issue #2: Optimization Objective Doesn't Reward Filling Gaps

**Location:** `modules/h_optimizer.py:calculate_objective()`

The objective function is:
```python
def calculate_objective(self, weights, candidate_x, current_team_x,
                       opponent_x, n_remaining, format='each_category'):
    # Calculate X_delta for future picks
    x_delta = self.calculate_x_delta(weights, n_remaining)

    # Total team projection
    team_projection = current_team_x + candidate_x + x_delta

    # Calculate win probabilities
    win_probs = self.calculate_win_probabilities(
        team_projection, opponent_x, variance
    )

    # Sum of win probabilities
    objective = np.sum(win_probs)
```

**The Problem:**

The objective maximizes **sum of win probabilities for the ENTIRE TEAM**, not **marginal gain from this specific pick**.

Example:
- Your PTS win probability might be 0.95 (very high)
- Your REB win probability might be 0.30 (very low)

When evaluating Kevin Durant:
- KD's PTS (5.52 X-score) pushes PTS win prob from 0.95 → 0.97 (**+0.02 gain**)
- KD's REB (1.49 X-score) pushes REB win prob from 0.30 → 0.35 (**+0.05 gain**)

When evaluating Karl-Anthony Towns:
- KAT's PTS (2.67 X-score) pushes PTS win prob from 0.95 → 0.96 (**+0.01 gain**)
- KAT's REB (3.59 X-score) pushes REB win prob from 0.30 → 0.45 (**+0.15 gain**)

**You'd expect:** KAT should win because he provides +0.16 total vs KD's +0.07 total

**But the optimizer does:** It finds weights that maximize the **final win probability**, not the marginal contribution. Since FT% naturally has high win probability (you're elite), it gets high weight. Since REB naturally has low win probability (you're terrible), it gets low weight.

This is **backwards logic** - the optimizer is essentially giving up on weak categories instead of targeting them.

### Issue #3: Win Probability Saturation

Win probabilities follow a sigmoid curve (normal CDF). Going from:
- **0.30 → 0.50 in REB** (weak → average) = huge value
- **0.95 → 0.97 in FT%** (elite → more elite) = minimal value

But because the optimizer is maximizing the TOTAL, and FT% starts so high, small improvements there still contribute meaningfully to the sum.

The marginal value of adding to an already-strong category should be much lower than adding to a weak category, but the current objective doesn't properly capture this.

---

## Why This Is Wrong

### Conceptual Issue

In fantasy basketball draft strategy, the **marginal value** of a player depends heavily on team context:

**Correct Logic:**
- If you're weak in REB, a rebounder has HIGH marginal value (fills a gap)
- If you're elite in FT%, a FT shooter has LOW marginal value (redundant)

**Current Algorithm Logic:**
- If FT% has high league scarcity, give it high weight
- If REB has low league scarcity, give it low weight
- Team composition barely matters

### Mathematical Issue

The gradient descent optimization is finding a **local optimum** that favors:
1. Categories with naturally high win probabilities (FT%, which you're already good at)
2. Categories that contribute broadly across many players (BLK, FG3M)
3. Well-rounded players who help many categories a little bit

Instead of finding the **global optimum** that:
1. Identifies your weakest categories
2. Maximizes marginal improvement
3. Values specialists who drastically improve specific weaknesses

---

## Proposed Fixes

### Option 1: Team-Aware Baseline Weights (Conservative Fix)

Modify baseline weight calculation to account for team composition:

```python
def calculate_team_aware_weights(self, current_team_x, baseline_weights):
    """
    Adjust baseline weights based on team strengths/weaknesses.

    Categories where you're weak get boosted weight.
    Categories where you're strong get reduced weight.
    """
    adjusted_weights = baseline_weights.copy()

    for i, cat in enumerate(self.categories):
        team_strength = current_team_x[i]

        # Define strength thresholds
        if team_strength < 1.0:  # Weak
            # Boost weight by 2x
            adjusted_weights[i] *= 2.0
        elif team_strength > 3.0:  # Strong
            # Reduce weight by 0.5x
            adjusted_weights[i] *= 0.5

    # Renormalize
    adjusted_weights = adjusted_weights / adjusted_weights.sum()

    return adjusted_weights
```

**Pros:** Simple, interpretable, preserves existing optimizer structure
**Cons:** Heuristic-based, may need tuning

### Option 2: Marginal Contribution Objective (Correct Fix)

Change the objective function to optimize **marginal gain**, not total win probability:

```python
def calculate_objective_marginal(self, weights, candidate_x, current_team_x,
                                opponent_x, n_remaining, format='each_category'):
    """
    Calculate objective as MARGINAL improvement from adding this player.
    """
    # Calculate X_delta for future picks
    x_delta = self.calculate_x_delta(weights, n_remaining)

    # Win probabilities WITHOUT this player
    team_without_player = current_team_x + x_delta
    win_probs_without = self.calculate_win_probabilities(
        team_without_player, opponent_x
    )

    # Win probabilities WITH this player
    team_with_player = current_team_x + candidate_x + x_delta
    win_probs_with = self.calculate_win_probabilities(
        team_with_player, opponent_x
    )

    # Marginal improvement
    marginal_gain = win_probs_with - win_probs_without

    # Objective = sum of marginal gains
    objective = np.sum(marginal_gain)

    return objective
```

**Pros:** Mathematically correct, directly optimizes what we care about
**Cons:** Requires more computation, may need new gradient calculation

### Option 3: Penalize Redundancy (Hybrid Approach)

Add a penalty term for drafting in categories you're already strong in:

```python
def calculate_objective_with_redundancy_penalty(self, weights, candidate_x,
                                                current_team_x, opponent_x,
                                                n_remaining, format='each_category'):
    """
    Objective with penalty for redundant categories.
    """
    # Standard objective
    base_objective = self.calculate_objective(
        weights, candidate_x, current_team_x,
        opponent_x, n_remaining, format
    )

    # Redundancy penalty
    redundancy = 0.0
    for i in range(len(candidate_x)):
        if current_team_x[i] > 3.0:  # Already strong in this category
            # Penalize proportional to how much the candidate adds
            redundancy += candidate_x[i] * 0.2  # 20% penalty

    objective = base_objective - redundancy

    return objective
```

**Pros:** Easy to add, tunable penalty strength
**Cons:** Still somewhat heuristic

---

## Recommended Action

**Immediate:** Implement **Option 1 (Team-Aware Baseline Weights)** as a quick fix.

**Long-term:** Implement **Option 2 (Marginal Contribution Objective)** as the correct solution, as it aligns with the theoretical foundation of H-scoring.

---

## Test Cases to Validate Fix

After implementing a fix, test these scenarios:

### Test 1: Extreme Weakness
```python
# Draft all guards (terrible REB)
my_team = ["Stephen Curry", "Damian Lillard", "Trae Young"]

# KAT should be STRONGLY preferred over KD
assert h_score(KAT) > h_score(KD) + 0.5
```

### Test 2: Redundancy
```python
# Draft all elite FT shooters
my_team = ["Stephen Curry", "Kevin Durant", "Chris Paul"]

# Another elite FT shooter should have REDUCED value
# Ben Simmons (bad FT%) should have HIGHER value
assert h_score("Ben Simmons") > h_score("Damian Lillard")
```

### Test 3: Category Balance
```python
# Balanced team
my_team = ["Nikola Jokić", "Ja Morant", "Mikal Bridges"]

# Specialists in weak categories should be valued higher
# Calculate which categories are weakest
# Then verify specialist for that category has high H-score
```

---

## Files to Modify

1. `modules/covariance.py` - Add team-aware weight calculation
2. `modules/h_optimizer.py` - Modify objective function
3. `modules/h_optimizer_final.py` - Update to use new objective
4. `test_optimizer_team_awareness.py` - NEW: Test suite for fixes

---

## References

- **Debug script:** `debug_optimizer_weights.py`
- **Comparison analysis:** `analyze_player_comparison.py`
- **Diagnostic report:** `DIAGNOSTIC_KD_vs_KAT.md`

---

## Notes

This bug explains many counterintuitive draft decisions:
- Why KD preferred over KAT despite needing rebounds
- Why the algorithm consistently undervalues specialists
- Why it tends to draft well-rounded players regardless of team composition

The algorithm is essentially doing **best player available** based on general value, rather than **best fit** based on team needs.
