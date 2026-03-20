# Retirement Planning Lab

Retirement planning application with historical simulation, Monte Carlo analysis, explicit decision ranking, and workbook export.

## Local setup

```bash
pip install -e .[dev]
pytest
streamlit run streamlit_app_v241_refactored.py
```

## Reproducibility controls

- Package version is tracked in `v241_refactor_app.version`
- Monte Carlo seed is captured in workbook provenance
- CI runs the test suite on every push / PR
