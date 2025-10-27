# Archive Folder

This folder contains old debug scripts, test results, and outdated documentation that were cleaned up on October 16, 2024.

## Contents

### debug_scripts/ (42 files)
Debugging and testing scripts used during development:
- `debug_*.py` - Various debugging scripts for H-score calculations
- `test_*.py` - Testing scripts for different components
- `analyze_*.py`, `check_*.py`, `compare_*.py` - Analysis utilities
- `explain_*.py`, `investigate_*.py`, `trace_*.py` - Diagnostic tools
- `fix_*.py` - Old fix attempts for various issues
- `generate_report.py` - HTML report generator
- `h_scoringchatgpt.py` - Reference implementation
- `example_usage.py` - Example script

### results/ (73 files total)

#### draft_results/ (36 files)
Draft simulation results from various test runs:
- `draft_results_YYYYMMDD_HHMMSS.json` - Draft outcomes with team rosters
- Dates range: Sep 30 - Oct 16, 2024

#### season_results/ (28 files)
Season simulation results:
- `season_results_YYYYMMDD_HHMMSS.csv` - 100-season simulation outcomes
- Dates range: Sep 30 - Oct 16, 2024

#### debug_outputs/ (3 files)
Detailed debugging outputs:
- `draft_debug_raw_*.txt` - Raw debug logs
- `draft_debug_detailed_*.json` - Structured debug data
- `draft_debug_run.txt` - Debug run output

#### other_results/ (6 files)
Miscellaneous result files:
- `draft_report_*.html` - HTML report from draft analysis
- `comparison_*.png` - Player comparison charts
- `scores_output*.csv` - H-score calculation outputs
- `all_players_h_scores.csv` - Full player rankings
- `final_h_scores_all_players.csv` - Final rankings

### old_documentation/ (6 files)
Outdated or superseded documentation:
- `ALGORITHM_STATUS.md` - Old algorithm status report
- `BUG_REPORT_optimizer_weights.md` - Old bug report
- `CHANGES_SUMMARY.md` - Old changes summary
- `DIAGNOSTIC_KD_vs_KAT.md` - Specific player diagnostic
- `OPTIMIZER_FILE_STRUCTURE.md` - Old file structure doc
- `RESUME_INSTRUCTIONS.md` - Outdated resume instructions

## Why These Were Archived

These files were part of the iterative development process but are no longer needed for:
1. Running the H-scoring algorithm
2. Understanding the current implementation
3. Daily usage of the system

The active documentation (CLAUDE.md, DRAFT_ANALYSIS.md, etc.) now contains all the relevant information from these archived files.

## Total Archived: 121 files

You can safely delete this entire archive folder if you're confident in the current implementation, or keep it for historical reference.
