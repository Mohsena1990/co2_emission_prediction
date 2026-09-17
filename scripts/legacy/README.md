# Legacy single-configuration pipeline

Scripts `01_run_fs.py` through `06_interpret_champion.py` are the
project's original single-flat-matrix pipeline (`X_full.parquet`, no
A1-A4 configuration split, no Stream A/B distinction). They predate the
current Q-DECEM design (`scripts/00_make_dataset.py` + `10`-`27`) and are
kept here for historical reference only.

**Do not use these for new work and do not cite results from them.**
They are:

- not exercised by the test suite (`tests/` imports the current `src/`
  modules these scripts also happen to call, but no test runs these
  scripts end-to-end),
- not referenced by the manuscript, `RESEARCH_OVERVIEW.md`, or
  `LIMITATIONS.md`,
- superseded in every respect (predictor governance, nested validation,
  Pareto/VIKOR decision support, leakage checks) by the current pipeline.

If you need the current pipeline, start from the repository root
`README.md`'s "One-command run" section instead.
