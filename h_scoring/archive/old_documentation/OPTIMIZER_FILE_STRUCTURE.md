# H-Score Optimizer: File Structure and Call Chain

## Production Files (Where the Algorithm is Actually Used)

### 1. **Core Optimizer Implementation**

#### `modules/h_optimizer.py`
- **Purpose:** Base H-score optimizer class
- **Key Methods:**
  - `optimize_weights()` - Gradient descent with Adam optimizer
  - `calculate_objective()` - **⚠️ BUG HERE**: Uses total win probability instead of marginal gain
  - `calculate_gradient()` - Numerical gradient calculation
  - `calculate_x_delta()` - Future picks adjustment
  - `calculate_win_probabilities()` - Win prob calculation
- **Status:** Base class, used by h_optimizer_final

#### `modules/h_optimizer_final.py`
- **Purpose:** Production optimizer with bug fixes (extends h_optimizer.py)
- **Key Methods:**
  - `evaluate_player()` - Main entry point for player evaluation
  - `_compute_xdelta_simplified()` - Fixed X_delta calculation
  - `calculate_win_probabilities()` - Overridden with category-specific variances
  - `_calculate_category_variances()` - Variance caps
- **Inherits from:** `HScoreOptimizer` (h_optimizer.py)
- **Status:** **CURRENT PRODUCTION OPTIMIZER** ✅

### 2. **Draft Assistant (Interactive Drafting)**

#### `draft_assistant.py`
- **Purpose:** Interactive draft tool for real drafts
- **Uses:** `HScoreOptimizerFinal` (from modules/h_optimizer_final.py)
- **Key Methods:**
  - `recommend_pick()` - Gets H-score recommendations for available players
  - `draft_player()` - Adds player to your team
  - `update_opponent_rosters()` - Updates opponent info
- **Calls optimizer at:** Line 109-114
```python
self.optimizer = HScoreOptimizerFinal(
    self.setup_params,
    self.scoring,
    omega=0.7,
    gamma=0.25
)
```
- **Status:** PRODUCTION - Used for real drafts

### 3. **Draft Simulator (Testing/Validation)**

#### `simulate_draft.py`
- **Purpose:** Simulate 12-team snake draft (H-scoring vs ADP)
- **Uses:** `DraftAssistant` → `HScoreOptimizerFinal`
- **Key Methods:**
  - `_your_pick_h_score()` - Uses H-scoring for your picks
  - `_opponent_pick_adp()` - Uses ADP for opponent picks
  - `run_draft()` - Executes full 13-round draft
- **Calls optimizer via:** `self.assistant.recommend_pick()` (Line 214)
- **Status:** TESTING - Validates algorithm performance

### 4. **Season Simulator (Full Validation)**

#### `simulate_season.py`
- **Purpose:** Run draft + 100 seasons to validate strategy
- **Uses:** `DraftSimulator` → `DraftAssistant` → `HScoreOptimizerFinal`
- **Flow:**
  1. Runs draft simulation
  2. Simulates 100 seasons with drafted teams
  3. Calculates win rates
- **Status:** TESTING - Full end-to-end validation

---

## Call Chain Diagram

```
simulate_season.py
    │
    └─→ DraftSimulator (simulate_draft.py)
            │
            └─→ DraftAssistant (draft_assistant.py)
                    │
                    └─→ HScoreOptimizerFinal (modules/h_optimizer_final.py)
                            │
                            └─→ HScoreOptimizer (modules/h_optimizer.py)
                                    │
                                    ├─→ optimize_weights() ⚠️ BUG: baseline weights
                                    ├─→ calculate_objective() ⚠️ BUG: total win prob
                                    ├─→ calculate_gradient()
                                    └─→ calculate_win_probabilities()
```

---

## Where the Bug Lives

### Bug Location #1: Baseline Weights
**File:** `modules/covariance.py`
**Method:** `calculate_baseline_weights(method='scarcity')`
**Issue:** Uses league-wide scarcity, ignores team composition

```python
# Line ~120-140
def calculate_baseline_weights(self, method='scarcity'):
    if method == 'scarcity':
        # ⚠️ BUG: This is the same for EVERYONE
        # Doesn't consider your team's specific needs
        cv_values = []
        for cat in self.categories:
            cat_values = self.season_averages[cat].values
            mean_val = np.mean(cat_values)
            std_val = np.std(cat_values)
            if mean_val > 0:
                cv = std_val / mean_val
```

### Bug Location #2: Objective Function
**File:** `modules/h_optimizer.py`
**Method:** `calculate_objective()`
**Lines:** ~228-280

```python
# Line ~228
def calculate_objective(self, weights, candidate_x, current_team_x,
                       opponent_x, n_remaining, format='each_category'):
    # Calculate X_delta for future picks
    x_delta = self.calculate_x_delta(weights, n_remaining)

    # Total team projection
    team_projection = current_team_x + candidate_x + x_delta

    # ⚠️ BUG: Calculate win probabilities for TOTAL TEAM
    # Should calculate MARGINAL gain from adding this player
    win_probs = self.calculate_win_probabilities(
        team_projection, opponent_x, variance
    )

    # ⚠️ BUG: Sum of total win probabilities
    # Should be sum of MARGINAL improvements
    objective = np.sum(win_probs)

    return objective
```

### Bug Location #3: Optimize Weights Initial Weights
**File:** `modules/h_optimizer.py`
**Method:** `optimize_weights()`
**Lines:** ~383-390

```python
# Line ~383
def optimize_weights(self, candidate_x, current_team_x, opponent_x,
                    n_remaining, initial_weights=None, ...):
    # Initialize weights
    if initial_weights is None:
        # ⚠️ BUG: Uses baseline weights that don't consider team
        weights = self.baseline_weights.copy()
    else:
        weights = initial_weights.copy()
```

---

## Supporting/Analysis Files (Not Production)

These files use the optimizer for debugging/analysis but are NOT part of the production flow:

### Analysis Tools
- `analyze_player_comparison.py` - Compares two players (e.g., KD vs KAT)
- `debug_optimizer_weights.py` - Shows exact weights chosen by optimizer
- `debug_wemby.py` - Debug specific player valuations
- `debug_h_score_issue.py` - General debugging
- `debug_top_players.py` - Debug top player rankings

### Test Files
- `test_h_scores.py` - Test H-score calculations
- `test_h_scores_quick.py` - Quick validation test
- `test_draft_variance.py` - Test draft randomness
- `test_do_not_draft.py` - Test exclusion list

### Historical Debug Files
- `diagnose_covariance.py`
- `debug_variances.py`
- `debug_x_delta.py`
- `fix_xdelta_exact.py`
- `modules/h_optimizer_fixed.py` (old version)

---

## Where to Implement Fixes

To fix the bugs, you need to modify:

### Fix #1: Team-Aware Weights (Quick Fix)
**Files to modify:**
1. `modules/covariance.py` - Add new method `calculate_team_aware_weights()`
2. `modules/h_optimizer.py` - Call new method in `optimize_weights()`

### Fix #2: Marginal Objective (Correct Fix)
**Files to modify:**
1. `modules/h_optimizer.py` - Replace `calculate_objective()` method
2. `modules/h_optimizer.py` - Update `calculate_gradient()` to work with new objective
3. `modules/h_optimizer_final.py` - Override if needed

### Fix #3: Create New Optimizer Class (Safest)
**Recommended approach:**
1. Create `modules/h_optimizer_team_aware.py` - New optimizer class
2. Test thoroughly with `test_optimizer_team_awareness.py`
3. Update `draft_assistant.py` to use new optimizer ONLY when ready
4. Keep old optimizer as fallback

---

## Testing Changes

After implementing fixes, test with:

```bash
# Unit tests
python test_optimizer_team_awareness.py

# Comparison analysis
python analyze_player_comparison.py

# Weight debugging
python debug_optimizer_weights.py

# Draft simulation
python simulate_draft.py

# Full validation
python simulate_season.py
```

---

## Summary

**Production entry point:** `draft_assistant.py`
**Production optimizer:** `modules/h_optimizer_final.py` (extends `modules/h_optimizer.py`)
**Bug locations:**
- `modules/covariance.py` - Baseline weights
- `modules/h_optimizer.py` - Objective function

**To fix:** Modify the base classes or create a new optimizer class that properly accounts for team composition.
