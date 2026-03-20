# Tier-1 Upgrade Notes

Implemented improvements:

1. **Model validation + diagnostics**
   - Added Monte Carlo validation report with:
     - Wilson confidence intervals for failure / ruin / shortfall rates
     - bootstrap confidence intervals for real median and real P10 ending balances
     - convergence status, final stderr, quantile drift, simulation utilization
     - regime-start concentration diagnostics (effective distinct start months, entropy)
   - Integrated into cached Monte Carlo artifacts, UI table, and workbook export.

2. **Testing + reproducibility**
   - Expanded test suite from 20 to 25 tests.
   - Added tests for validation report contents, cached MC validation, explicit objective metadata, export provenance/version, and packaging version consistency.
   - Added a Streamlit compatibility shim so non-UI tests run in environments without Streamlit installed.

3. **Explicit objective framework**
   - Replaced heuristic policy ranking with an explicit Pareto + lexicographic objective framework.
   - Added objective descriptions, recommendation basis, Pareto-efficient flag, dominated-count metadata, and objective rank.
   - Kept backward-compatible `Rank` column.

4. **Packaging + CI + versioning**
   - Added `pyproject.toml`, `requirements.txt`, `README.md`, GitHub Actions CI workflow, package version module, and provenance version/build metadata.

Additional improvement:
- Replaced fragility heuristic ranking with explicit stress-priority ordering and retained backward-compatible `Fragility Rank`.

Verification performed:
- `pytest -q` -> 25 passed
- module compilation checks passed
