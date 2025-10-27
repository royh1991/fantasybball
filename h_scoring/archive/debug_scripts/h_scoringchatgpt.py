#!/usr/bin/env python3
"""
h_scoring_monolith.py

Single-file implementation to compute Z, G, and H scores from raw per-game player data.

Usage:
    python h_scoring_monolith.py --input raw_games.csv --out scores.csv

Output:
    CSV with per-player aggregated stats, Z per category, G-score, H-score and diagnostics.

Notes:
 - This is a simplified, practical implementation of Rosenof's H-scoring ideas.
 - X_delta has two implementations:
     * exact (Sigma-weighted) with regularization and diagnostics
     * linear fallback (a*jC + b*v) solving 2x2 system — more robust, lower variance
 - By default we compute H-scores for "Each Category" objective (sum of category win probabilities).
 - Tweak gamma, omega, multiplier_scale to calibrate magnitudes.
"""

import argparse
import math
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.linalg import solve, pinv

# ---------------------------
# Configuration defaults
# ---------------------------
DEFAULT_CATEGORIES = [
    "PTS", "REB", "AST", "STL", "BLK", "FG3M", "TOV", "FT_PCT", "FG_PCT"
]
# Mapping from CSV column names -> canonical category names (adjust if your CSV differs)
CSV_TO_CAT = {
    "PTS": "PTS",
    "REB": "REB",
    "AST": "AST",
    "STL": "STL",
    "BLK": "BLK",
    "FG3M": "FG3M",
    "TOV": "TOV",
    "FT_PCT": "FT_PCT",
    "FG_PCT": "FG_PCT"
}

# H-scoring hyperparams
GAMMA = 0.25   # paper used 0.25
OMEGA = 0.7    # paper used 0.7
RIDGE_FRAC = 1e-8   # regularize Sigma by ridge_frac * trace(Sigma)
MUL_SCALE = 0.25    # scale on multiplier (N-K-1) to avoid explosive values (tunable)
USE_SIGMA_IN_B = True  # whether to multiply b by sigma (paper uses true sigma)
USE_EXACT_XDELTA = True  # if False uses linear fallback
CLIP_XDELTA_NORM = None  # e.g. 10.0 to clip; None to keep unclipped
DIAG = True  # print diagnostics

# Draft parameters (example 12-team, 13-player rosters similar to paper)
N_TEAMS = 12
ROSTER_SIZE = 13
# For a given pick (0-indexed seat), remaining picks (first pick): N-K-1  => N = roster slots total? in paper they used N as number of slots per team?
# We'll treat multiplier = (ROSTER_SIZE - picks_already_made - 1). For first pick, picks_already_made=0 => multiplier = 12.
# In practice you may want to use (N_TEAMS * ROSTER_SIZE?) but we'll use roster-centric multiplier as in the paper.
# We'll parameterize multiplier calculation in compute_x_delta()

# ---------------------------
# Utilities
# ---------------------------
def safe_normalize_vec(x, eps=1e-12):
    x = np.asarray(x, dtype=float).reshape(-1)
    s = x.sum()
    if abs(s) < eps:
        # fallback: uniform
        return np.ones_like(x) / float(x.size)
    return x / float(s)

# ---------------------------
# Data ingestion & aggregation
# ---------------------------
def load_and_aggregate(csv_path, categories=DEFAULT_CATEGORIES, player_id_col="PLAYER_ID"):
    """
    Read raw per-game CSV and aggregate to season/player-level average per category.
    Returns a DataFrame `players_df` with one row per player and the aggregated category means.
    """
    df = pd.read_csv(csv_path)
    # Ensure category columns exist: if percents are in decimals (0-1) keep as is; if in percent (0-100), convert externally
    # Select only rows with meaningful minutes (filter out DNPs if any)
    df = df[df["MIN"].notnull()]

    # Simple aggregation: mean of each stat per player (across games in the provided seasons)
    agg_funcs = {c: "mean" for c in categories if c in df.columns}
    agg_funcs.update({"MIN": "mean", "GAME_DATE": "count"})  # GAME_DATE count = games played
    by = df.groupby(player_id_col)

    agg = by.agg(agg_funcs).rename(columns={"GAME_DATE": "GAMES_PLAYED"})
    # For a player, defensive counting stats may be sparse. Fill NA with 0
    for c in categories:
        if c in agg.columns:
            agg[c] = agg[c].fillna(0)
        else:
            # If any category missing, add zero column
            agg[c] = 0.0
    # Add name if present
    if "PLAYER_NAME" in df.columns:
        names = df.groupby(player_id_col)["PLAYER_NAME"].first()
        agg = agg.join(names)
    agg = agg.reset_index().rename(columns={player_id_col: "PLAYER_ID"})
    return agg

# ---------------------------
# Z & G computation
# ---------------------------
def compute_z_scores(players_df, categories=DEFAULT_CATEGORIES):
    """
    Compute Z-scores for each player and each category.
    Return players_df with Z_{<cat>} columns plus G-score (sum of Zs).
    """
    df = players_df.copy()
    # Compute league mean and std for each category
    mu = {}
    sigma = {}
    for c in categories:
        vals = df[c].values.astype(float)
        mu[c] = float(np.nanmean(vals))
        sigma[c] = float(np.nanstd(vals, ddof=0))
        # Avoid zero-std
        if sigma[c] <= 1e-8:
            sigma[c] = 1.0
    # compute z
    for c in categories:
        df[f"Z_{c}"] = (df[c] - mu[c]) / sigma[c]
    # G-score is sum of Zs
    z_cols = [f"Z_{c}" for c in categories]
    df["G_SCORE"] = df[z_cols].sum(axis=1)
    return df, mu, sigma

# ---------------------------
# Build X-scores basis (paper's X-score)
# ---------------------------
def compute_x_scores_from_z(players_df, sigma_counts_map=None, sigma_pct_map=None, categories=DEFAULT_CATEGORIES):
    """
    X-score basis: paper uses X-score (G-score variant with some variance terms removed).
    For simplicity here we will treat X-score = Z-score but with careful handling of percentage categories (FT_PCT, FG_PCT).
    The paper defines X-score differently; this is a practical mapping.
    """
    df = players_df.copy()
    # For counting stats (non-percentage), use Zs directly.
    # For percentage stats, treat them as Z but we might multiply by expected volume factor (games/minutes) optionally.
    x_cols = []
    for c in categories:
        if c.endswith("_PCT"):
            # Use the Z but scale by sqrt(avg minutes played / max) to reflect volume (simple heuristic)
            # If MIN exists:
            if "MIN" in df.columns:
                vol = np.sqrt(df["MIN"].clip(lower=1) / (df["MIN"].max() + 1e-6))
            else:
                vol = 1.0
            df[f"X_{c}"] = df[f"Z_{c}"] * vol
        else:
            df[f"X_{c}"] = df[f"Z_{c}"]
        x_cols.append(f"X_{c}")
    return df, x_cols

# ---------------------------
# Sigma (covariance) computation in X-basis
# ---------------------------
def compute_sigma_matrix(df_x, x_cols):
    """
    Compute covariance matrix Sigma across players for X-scores basis.
    Returns Sigma (m x m) and mean vector (m).
    """
    X = df_x[x_cols].values.astype(float)
    # compute covariance of the per-player means (not game-sample cov)
    # Using population covariance (ddof=0)
    Sigma = np.cov(X, rowvar=False, ddof=0)
    # If covariance returns scalar because only one category, wrap
    if Sigma.ndim == 0:
        Sigma = np.atleast_2d(Sigma)
    means = np.nanmean(X, axis=0)
    return Sigma, means

# ---------------------------
# X_delta implementations
# ---------------------------
def compute_xdelta_exact(jC, v, Sigma, gamma=GAMMA, omega=OMEGA,
                         multiplier=1.0, ridge_frac=RIDGE_FRAC,
                         use_sigma_in_b=USE_SIGMA_IN_B, clip_norm=CLIP_XDELTA_NORM):
    """
    Exact Sigma-weighted solution per the paper with numerical regularization.

    Returns vector x_delta (length m) and diagnostics dict.
    Formula implemented: xδ = multiplier * (Sigma_reg @ U.T @ z)
      where U = [v; jC] (shape 2 x m), M = U @ Sigma_reg @ U.T, z = solve(M, b)
      and b = [-gamma * sigma, omega * sigma] (optionally without sigma).
    """
    jC = np.asarray(jC, dtype=float).reshape(-1)
    v = np.asarray(v, dtype=float).reshape(-1)
    m = jC.size
    Sigma = np.asarray(Sigma, dtype=float)
    # normalize jC and v if not already
    jC = safe_normalize_vec(jC)
    v  = safe_normalize_vec(v)

    trace = np.trace(Sigma)
    ridge = max(ridge_frac * (trace + 1e-12), 1e-12)
    Sigma_reg = Sigma + ridge * np.eye(m)

    # projection of jC onto v in Sigma-metric: coeff = (v^T Sigma jC) / (v^T Sigma v)
    denom_vSv = float(v.T @ Sigma_reg @ v)
    if denom_vSv <= 0 or not np.isfinite(denom_vSv):
        denom_vSv = 1e-8
    proj_coeff = float((v.T @ Sigma_reg @ jC) / denom_vSv)
    jC_perp = jC - v * proj_coeff

    sigma2 = float(max(jC_perp.T @ Sigma_reg @ jC_perp, 0.0))
    sigma = math.sqrt(sigma2) if sigma2 > 0 else 0.0

    # build U and b
    U = np.vstack([v, jC])   # shape (2, m)
    if use_sigma_in_b:
        b = np.array([-gamma * sigma, omega * sigma], dtype=float)
    else:
        b = np.array([-gamma, omega], dtype=float)

    M = U @ Sigma_reg @ U.T  # 2x2
    cond_M = np.linalg.cond(M) if np.all(np.isfinite(M)) else 1e16
    if cond_M > 1e12 or not np.isfinite(cond_M):
        # regularize M
        jitter = 1e-8 * (np.trace(M) + 1e-12)
        M = M + jitter * np.eye(2)
        cond_M = np.linalg.cond(M)

    # solve
    try:
        z = solve(M, b)
    except Exception:
        z = pinv(M) @ b

    xdelta_unit = Sigma_reg @ U.T @ z  # length m
    xdelta = multiplier * xdelta_unit

    # optional clipping
    if clip_norm is not None:
        normx = np.linalg.norm(xdelta)
        if normx > clip_norm and normx > 0:
            xdelta = xdelta * (clip_norm / normx)

    diag = {
        "proj_coeff": proj_coeff,
        "sigma": sigma,
        "sigma2": sigma2,
        "cond_M": cond_M,
        "ridge_added": ridge,
        "norm_xdelta_unit": float(np.linalg.norm(xdelta_unit)),
        "norm_xdelta": float(np.linalg.norm(xdelta)),
        "multiplier": float(multiplier),
    }
    return xdelta, diag

def compute_xdelta_linear_fallback(jC, v, Sigma, gamma=GAMMA, omega=OMEGA,
                                   multiplier=1.0, clip_norm=CLIP_XDELTA_NORM):
    """
    Simpler fallback: express x_delta as linear combo of jC and v:
       x_delta = multiplier * (a * jC + b * v)
    Solve for a,b to meet dot constraints:
       jC^T x_delta = omega_sigma'   and  v^T x_delta = -gamma_sigma'
    We choose sigma' = 1 (or optionally the computed sigma), but the fallback
    focuses on direction so we set sigma' = 1 to avoid magnitude explosion.
    """
    jC = safe_normalize_vec(jC)
    v = safe_normalize_vec(v)
    # set sigma' = 1 to avoid scaling explosion
    rhs = np.array([omega * 1.0, -gamma * 1.0])  # [jC^T x, v^T x] targets
    A = np.array([[jC @ jC, jC @ v], [v @ jC, v @ v]], dtype=float)
    # protect A invertibility
    cond = np.linalg.cond(A)
    if cond > 1e12:
        A = A + 1e-8 * np.eye(2)
    try:
        a_b = solve(A, rhs)
    except Exception:
        a_b = pinv(A) @ rhs
    a, b = a_b
    xdelta = multiplier * (a * jC + b * v)
    if clip_norm is not None:
        normx = np.linalg.norm(xdelta)
        if normx > clip_norm and normx > 0:
            xdelta = xdelta * (clip_norm / normx)
    diag = {"cond_A": cond, "a": float(a), "b": float(b), "norm_xdelta": float(np.linalg.norm(xdelta))}
    return xdelta, diag

def compute_xdelta_wrapper(jC, v, Sigma, picks_remaining, gamma=GAMMA, omega=OMEGA,
                           use_exact=USE_EXACT_XDELTA, multiplier_scale=MUL_SCALE,
                           ridge_frac=RIDGE_FRAC, use_sigma_in_b=USE_SIGMA_IN_B,
                           clip_norm=CLIP_XDELTA_NORM):
    """
    Top-level wrapper to compute xdelta. picks_remaining = N - K - 1 (paper) or computed by caller.
    This wrapper applies a multiplier scaling to avoid runaway magnitudes and chooses exact vs fallback.
    """
    # effective multiplier
    multiplier = max(0.0, picks_remaining) * float(multiplier_scale)
    if use_exact:
        xdelta, diag = compute_xdelta_exact(jC, v, Sigma, gamma=gamma, omega=omega,
                                            multiplier=multiplier, ridge_frac=ridge_frac,
                                            use_sigma_in_b=use_sigma_in_b, clip_norm=clip_norm)
        diag["impl"] = "exact"
    else:
        xdelta, diag = compute_xdelta_linear_fallback(jC, v, Sigma, gamma=gamma, omega=omega,
                                                      multiplier=multiplier, clip_norm=clip_norm)
        diag["impl"] = "linear_fallback"
    diag["multiplier_used"] = multiplier
    return xdelta, diag

# ---------------------------
# Win probability and H-score computation
# ---------------------------
def compute_win_probs(team_proj, opp_proj, variance):
    """
    team_proj, opp_proj are vectors over categories (same length).
    variance is scalar or vector (per-category variance). We'll assume scalar per paper.
    Returns per-category win probability (cdf of normal at zero for differential).
    """
    diff = team_proj - opp_proj
    # If variance is scalar:
    if np.isscalar(variance):
        sigma = math.sqrt(max(variance, 1e-12))
        z = diff / sigma
    else:
        sigma = np.sqrt(np.maximum(variance, 1e-12))
        z = diff / sigma
    return norm.cdf(z), z

def compute_h_score_for_candidate(candidate_x, current_team_x, x_delta, opponent_x, variance, objective="each"):
    """
    candidate_x, current_team_x, x_delta, opponent_x are 1D arrays (category X-basis).
    variance: scalar or vector (per-category)
    objective: "each" (sum of per-category win probs) or "most" (probability of winning majority) [most not implemented; use each].
    Returns objective value (H-score) and win_probs (per-category).
    """
    team_proj = current_team_x + candidate_x + x_delta
    win_probs, z = compute_win_probs(team_proj, opponent_x, variance)
    if objective == "each":
        V = float(win_probs.sum())
    else:
        # Placeholder: Most-categories requires enumerating scenarios; omitted for brevity but can be added.
        V = float(win_probs.sum())
    return V, win_probs, z

# ---------------------------
# Opponent modeling
# ---------------------------
def compute_default_opponent_projection(players_df, x_cols, n_opponents_known=None, picks_made=0):
    """
    If opposing rosters are unknown, approximate opponent projection using average top G-score players.
    Simpler heuristic: compute average X-vector of top-K candidates by G-score (or top 6 per position).
    Returns a vector (length m) of opponent expected per-category X.
    """
    # Use top players by G_SCORE to approximate the typical player a competitor might have next
    df = players_df.copy()
    if "G_SCORE" not in df.columns:
        raise ValueError("G_SCORE required. Run compute_z_scores first.")
    # Take the top 'sample_size' players to average - tune as you see fit
    sample_size = max(20, int(len(df) * 0.1))
    top = df.nlargest(sample_size, "G_SCORE")
    avg_vec = top[x_cols].mean(axis=0).values.astype(float)
    # Opponent projection per slot: for a single team we need projection of K+1 known players; for simplicity return avg_vec (per slot)
    # To get opponent team vector (sum across their K+1 known players), multiply avg_vec by an estimate of number of players considered:
    approx_players_known = 1  # per paper they may use K+1; here we will model matchup against a single average player for differential
    return avg_vec

# ---------------------------
# Orchestration: compute all scores
# ---------------------------
def compute_all_scores(players_agg_df, categories=DEFAULT_CATEGORIES,
                       gamma=GAMMA, omega=OMEGA, picks_made=0,
                       multiplier_scale=MUL_SCALE, use_exact=USE_EXACT_XDELTA,
                       clip_xdelta_norm=CLIP_XDELTA_NORM, diag=DIAG):
    """
    Master function to compute Z, G, X, Sigma, X_delta, and H-score for each candidate.
    Returns a DataFrame with scores and optional diagnostics.
    """
    # 1) compute Z & G
    df_z, mu_map, sigma_map = compute_z_scores(players_agg_df, categories=categories)
    # 2) compute X columns
    df_x, x_cols = compute_x_scores_from_z(df_z, categories=categories)
    # 3) compute Sigma across players in X-basis
    Sigma, means = compute_sigma_matrix(df_x, x_cols)
    m = len(x_cols)
    # 4) compute v: convert from X basis to G basis vector - simple choice: v = normalized vector of per-category variances inverse (or all ones)
    # in the paper v is conversion vector; here we pick v as normalized per-category importance equal to 1/m uniform
    v = np.ones(m) / float(m)
    # 5) define jC: starting category weights; default = v (but must perturb slightly because gradient undefined at jC==v)
    jC = v.copy() + 1e-6 * (np.random.rand(m) - 0.5)
    jC = safe_normalize_vec(jC)

    # 6) compute picks_remaining: the paper uses (N - K - 1); we use (ROSTER_SIZE - picks_made - 1)
    picks_remaining = max(0, ROSTER_SIZE - picks_made - 1)

    # 7) compute X_delta (same for all candidates at this draft step)
    x_delta, xdelta_diag = compute_xdelta_wrapper(
        jC, v, Sigma, picks_remaining,
        gamma=gamma, omega=omega,
        use_exact=use_exact, multiplier_scale=multiplier_scale,
        ridge_frac=RIDGE_FRAC, use_sigma_in_b=USE_SIGMA_IN_B,
        clip_norm=clip_xdelta_norm
    )

    if diag:
        print("X_delta diagnostics:", xdelta_diag)

    # 8) opponent projection: use average G-score players in X-basis as default opponent slot vector
    opp_vec_single = compute_default_opponent_projection(df_x, x_cols, picks_made=picks_made)

    # 9) variance: approximate as "2*N + (N-K-1) * X_sigma^2" per paper simplified; we'll just use scalar variance = 2*ROSTER_SIZE + picks_remaining
    variance = float(2 * ROSTER_SIZE + picks_remaining * 1.0)

    # 10) compute H score for each candidate (one-row per player)
    results = []
    for idx, row in df_x.iterrows():
        candidate_x = row[x_cols].values.astype(float)
        current_team_x = np.zeros_like(candidate_x)  # this function assumes measuring H-score at draft start; for mid-draft pass current team state here
        # Note: in a draft simulation you'd fill current_team_x with sum of already-drafted players Xs
        V, win_probs, z = compute_h_score_for_candidate(candidate_x, current_team_x, x_delta, opp_vec_single, variance)
        rec = {
            "PLAYER_ID": row["PLAYER_ID"],
            "PLAYER_NAME": row.get("PLAYER_NAME", ""),
            "G_SCORE": row.get("G_SCORE", 0.0),
            "H_SCORE": V,
            "H_ZNORM": float(np.linalg.norm(z)),
            "XDELTA_NORM": float(np.linalg.norm(x_delta))
        }
        # attach per-category z/win prob optionally
        for i, c in enumerate(x_cols):
            rec[f"X_{c}"] = float(candidate_x[i])
            rec[f"WINP_{c}"] = float(win_probs[i])
        results.append(rec)
    results_df = pd.DataFrame(results)
    # merge on names, G
    final = results_df.merge(df_z[["PLAYER_ID", "G_SCORE"]], on="PLAYER_ID", how="left")
    return final, {"Sigma": Sigma, "x_cols": x_cols, "x_delta_diag": xdelta_diag}

# ---------------------------
# Entry point & CLI
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Compute Z, G, H scores from raw per-game CSV")
    parser.add_argument("--input", "-i", required=True, help="path to raw games CSV")
    parser.add_argument("--out", "-o", default="scores_out.csv", help="output CSV for scores")
    parser.add_argument("--diag", action="store_true", help="print diagnostics")
    parser.add_argument("--use_exact_xdelta", action="store_true", help="use exact X_delta (default small fallback may be used)")
    parser.add_argument("--mult_scale", type=float, default=MUL_SCALE, help="multiplier scale for picks_remaining")
    args = parser.parse_args()

    # Use local variables instead of modifying globals
    diag_mode = args.diag
    use_exact_mode = args.use_exact_xdelta
    mul_scale_value = args.mult_scale

    print("Loading and aggregating data...")
    players_df = load_and_aggregate(args.input, categories=DEFAULT_CATEGORIES)
    print(f"Players aggregated: {len(players_df)}")

    print("Computing all scores...")
    final_df, meta = compute_all_scores(players_df, categories=DEFAULT_CATEGORIES,
                                        gamma=GAMMA, omega=OMEGA,
                                        picks_made=0,
                                        multiplier_scale=mul_scale_value,
                                        use_exact=use_exact_mode,
                                        clip_xdelta_norm=CLIP_XDELTA_NORM,
                                        diag=diag_mode)
    print("Saving results to", args.out)
    final_df.to_csv(args.out, index=False)

    if diag_mode:
        print("--- Diagnostics ---")
        print("x_cols:", meta["x_cols"])
        Sigma = meta["Sigma"]
        print("Sigma diag (per-category variances):", np.diag(Sigma))
        print("X_delta diag:", meta["x_delta_diag"])
    print("Done.")

if __name__ == "__main__":
    main()
