# Diagnostic Report: KD vs KAT Preference Analysis

**Date:** 2025-10-01
**Issue:** H-scoring consistently prefers Kevin Durant over Karl-Anthony Towns even in contexts where KAT appears to be a better fit

---

## Summary of Findings

After running `analyze_player_comparison.py`, we found that H-scoring prefers Kevin Durant in **all tested scenarios**, including contexts where team composition clearly needs what KAT provides.

---

## Key Data Points

### Static Rankings (No Team Context)

**Kevin Durant:**
- ADP: 22.7 (later pick)
- Total G-Score: 4.94
- Dominates in: PTS (27.5), AST (4.8), BLK (1.3), FG% (53.3%), FT% (87.1%), 3PM (2.3)

**Karl-Anthony Towns:**
- ADP: 10.5 (much earlier pick)
- Total G-Score: 4.76
- Dominates in: REB (10.3), TOV (2.8 < 3.2), DD (0.55)

**Winner:** KD by 0.18 G-score

---

## The Suspicious Case: After Drafting SGA + Harden

### Team Composition Analysis

After drafting **Shai Gilgeous-Alexander + James Harden**, your team has:

**Strong categories (don't need):**
- PTS: 9.00 X-score 💪
- AST: 9.28 X-score 💪
- STL: 3.09 X-score 💪
- FT%: 6.99 X-score 💪

**Weak categories (desperately need):**
- **REB: 0.06 X-score** ⚠️ CRITICAL NEED
- **FG%: -0.02 X-score** ⚠️
- **FG3%: 0.65 X-score** ⚠️
- **TOV: -4.28 X-score** ⚠️

### Who Fills the Needs Better?

| Category | Team Need | KD Contribution | KAT Contribution | Better Fit |
|----------|-----------|-----------------|------------------|------------|
| REB      | **HIGH** (0.06) | 1.49 | **3.59** | ✓ KAT |
| FG%      | **HIGH** (-0.02) | **0.78** | 0.22 | ✓ KD |
| FG3%     | **HIGH** (0.65) | **0.64** | 0.47 | ✓ KD |
| TOV      | **HIGH** (-4.28) | -2.52 | **-1.83** | ✓ KAT |
| DD       | MED (1.22) | -0.16 | **2.41** | ✓ KAT |
| BLK      | MED (1.29) | **1.92** | 0.09 | ✓ KD |

**KAT fills 3 critical needs:** REB, TOV, DD
**KD fills 3 critical needs:** FG%, FG3%, BLK

### H-Score Result

**Kevin Durant: 7.8145**
**Karl-Anthony Towns: 7.6182**

**Winner:** KD by +0.1963

---

## Why This Is Counterintuitive

1. **Positional Scarcity:** After drafting 2 guards, you'd expect rebounds to be highly valued
   - KAT provides **2.4x more rebounds** (3.59 vs 1.49 X-score)
   - Yet this only closes the gap by 0.2 H-score points

2. **Unique Value:** KAT's DD contribution (2.41) is massive compared to KD (-0.16)
   - Double-doubles are harder to find from non-bigs
   - This advantage doesn't seem to be weighted heavily

3. **ADP Value:** KAT's ADP is 10.5 vs KD's 22.7
   - KD is available **12 picks later**
   - You could potentially get KD in round 3, making KAT at pick 2.6 (round 3, pick 6 in 12-team snake) questionable value

4. **Category Balance:**
   - With SGA + Harden, you have elite PTS/AST/FT% already
   - Adding KD gives you MORE of what you already have (scoring efficiency)
   - Adding KAT fills holes (REB, DD)

---

## Possible Explanations

### Hypothesis 1: Efficiency Categories Overweighted

The algorithm may be overweighting shooting percentages:
- KD's FG% advantage (0.78 vs 0.22 X-score) = **+0.56**
- KD's FG3% advantage (0.64 vs 0.47 X-score) = **+0.17**
- Combined efficiency edge: **+0.73 X-score**

Compare to:
- KAT's REB advantage (3.59 vs 1.49 X-score) = **+2.10**
- KAT's DD advantage (2.41 vs -0.16 X-score) = **+2.57**
- Combined positional edge: **+4.67 X-score**

Despite KAT having a **+3.94 net advantage** in needed categories, KD still wins. This suggests percentage categories may have disproportionate weight in the H-score optimization.

### Hypothesis 2: X_delta Favors KD's Profile

The X_delta term (expected future picks adjustment) may be systematically favoring well-rounded players like KD over specialists like KAT:
- KD wins in 7/11 categories
- KAT wins in 3/11 categories (REB, TOV, DD)

If X_delta assumes future picks will be balanced, it might undervalue drafting a specialist when you need that specialty.

### Hypothesis 3: Opponent Modeling Issue

With `opponent_teams=[]` (empty list), the algorithm may default to assuming average opponents. This could:
- Undervalue scarce stats (REB from non-traditional bigs)
- Overvalue "safe" categories that everyone has (FG%, FT%)

### Hypothesis 4: G-Score Ceiling Effect

KD's higher total G-score (4.94 vs 4.76) might create a "ceiling" that team composition can't overcome:
- 0.18 G-score advantage
- H-score only amplifies this to 0.20 despite clear team needs

This would suggest the "H" in H-scoring isn't as "dynamic" as intended - it's mostly just G-score with minor adjustments.

---

## Recommendations for Investigation

### 1. Test with Extreme Scenarios

Create a team that is **extremely weak** in REB (e.g., all guards):
```python
extreme_guards = ["Damian Lillard", "Trae Young", "Stephen Curry", "Tyrese Maxey"]
```

If KD still wins over KAT, the algorithm isn't properly weighting team needs.

### 2. Inspect Optimal Weights

Log the `optimal_weights` returned by the optimizer:
- Are REB weights actually increasing when you need rebounds?
- Are efficiency weights decreasing when you already have strong FG%/FT%?

### 3. Test Category-Specific Builds

Draft teams that intentionally punt a category and see if H-scoring adapts:
- Punt FT% team (Giannis, Westbrook) - should prefer KAT (good FT%) over Drummond (bad FT%)
- Punt 3PM team - should prefer traditional bigs

### 4. Compare to Actual Draft Results

Look at your actual season simulation results:
- You drafted KD at pick 6 and finished 3rd (73.6% win rate)
- Team 10 (KAT) finished 11th (21.5% win rate)

This validates KD > KAT in actual play, but Team 10 also made other picks. Need to isolate the KD vs KAT contribution.

---

## Conclusion

The H-scoring algorithm consistently prefers Kevin Durant over Karl-Anthony Towns by ~0.2 H-score points across all contexts, even when:
1. Team composition clearly needs rebounds
2. KAT provides 2.4x more value in the needed category
3. KAT fills multiple holes (REB, DD, TOV) while KD stacks existing strengths

**This suggests one of:**
1. The algorithm is working correctly and efficiency > positional need
2. There's a bug in how team composition affects optimization
3. The weighting between category types (efficiency vs counting) needs tuning

**Next Step:** Run the extreme scenario tests to determine if this is intentional design or a bug.

---

## Test Commands

```bash
# Run the comparison
python analyze_player_comparison.py

# Generates:
# - Terminal output with full analysis
# - comparison_Kevin_Durant_vs_Karl-Anthony_Towns.png
```
