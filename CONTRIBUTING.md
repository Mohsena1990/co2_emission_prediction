# Contributing

This is a research-reproducibility codebase attached to a specific
manuscript (see `CITATION.cff`). Contributions are welcome, particularly:
bug reports against a specific reported number, replication attempts
(including on other countries' grid data - see `RESEARCH_OVERVIEW.md`'s
generalisation note), and test coverage improvements.

## Before you start

1. Read `RESEARCH_OVERVIEW.md` (core idea, algorithm design) and
   `LIMITATIONS.md` (what the current results do and do not support) -
   most "is this a bug?" questions are answered by one of these two.
2. Check `PAPER_MAPPING.md` if your question is "which script produced
   Table/Figure N in the paper?"

## Making a change

1. `pip install -r requirements.txt` (exact pinned versions - the pipeline's
   reproducibility depends on them; see the comment at the top of that
   file before bumping anything by hand - Dependabot PRs are checked by
   CI automatically).
2. `make test` (or `pytest tests/ -v`) before and after your change.
   `make test-leakage` specifically runs the leakage/nested-CV isolation
   suite - anything touching `src/splits/`, `src/fs/`, or
   `src/optimization/` should pass this explicitly.
3. If you change a number that ends up in the manuscript (an entry in
   `outputs/tables/` or `outputs/robustness/`), update `CHANGELOG.md` and,
   if it affects an existing claim, `LIMITATIONS.md`.
4. Keep the leakage-governance discipline: any new predictor must be
   registered in `config/feature_registry.yaml` (source, lag, release
   delay, target dependence) before it can enter a feature matrix - this
   is enforced, not just documented, by `src/features/registry.py` and
   `tests/test_feature_registry.py`.

## Pull requests

CI (`.github/workflows/tests.yml`) runs the full suite plus a dedicated
leakage/nested-CV job on every PR - it must pass before merge. Describe
*why* the change is needed, not just what it does; if it changes a
reported number, say which one and by how much.
