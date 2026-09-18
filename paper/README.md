# Manuscript source

`main.tex` is the full revised manuscript (elsarticle, target journal: Applied
Energy). `highlights.txt` is the separate 5-bullet Highlights file most
Elsevier journals require as its own upload, not embedded in the PDF.

## Before this compiles

**`references.bib` is not in this repository and was not provided during the
revision that produced `main.tex`.** Every citation key already used in the
prior version of the manuscript is preserved unchanged, so if you already
have a working `references.bib` from before, drop it in this directory and
the document should resolve as before. One new key was added and needs a new
entry:

```bibtex
@article{dieboldmariano1995,
  author  = {Diebold, Francis X. and Mariano, Roberto S.},
  title   = {Comparing Predictive Accuracy},
  journal = {Journal of Business \& Economic Statistics},
  volume  = {13},
  number  = {3},
  pages   = {253--263},
  year    = {1995},
  doi     = {10.1080/07350015.1995.10524599}
}
```

Do not let anything else in this repository (or an assistant) fabricate
bibliography entries for you — verify every reference against the real
source before submission.

## What changed in this revision

See the project root's `CHANGELOG.md` for the full list. In summary:
every place the manuscript described its own audit/bug-fix history
("corrected", "the previous all-zero output was traced to...") has been
removed - the paper now presents the validated analysis directly, with the
methodology note about tuned-refit SHAP kept only as a forward-looking
methods statement (Section 3.7), not a narrative about what was once
broken. Six new subsections were added: the carbon-intensity decomposition
(4.6), the dispersion-mechanism investigation (4.7), per-horizon
significance with Diebold--Mariano confirmation (folded into 4.8/Table 13),
empirical prediction intervals (4.9), a live nowcast demonstration (4.10),
and a new Discussion subsection on policy and economic implications (5.4).
The Table 12 Panel B "Full A3" row was corrected to match the headline
champion numbers reported everywhere else in the paper (previously a
silently different reduced-budget re-fit).

## Figures

All figures are PDF, copied into `figures/pdf/main/` and
`figures/pdf/forecasting/` from `../outputs/figures/pdf/...` in the parent
repository. If you regenerate any of them (e.g. by re-running
`scripts/19_grid_ablation.py` or `scripts/24_policy_figures.py`), re-copy
the updated PDF into this folder - it is not symlinked.
