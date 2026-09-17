## What changed and why

<!-- Not just "what" - why was this needed? Link an issue if there is one. -->

## Does this change any reported number?

<!-- If this touches outputs/tables/, outputs/robustness/, or outputs/figures/,
     say which manuscript table/figure (see PAPER_MAPPING.md) and by how much. -->

- [ ] No reported number changes
- [ ] A reported number changes - updated `CHANGELOG.md` and, if relevant, `LIMITATIONS.md`

## Checklist

- [ ] `make test` passes locally
- [ ] `make test-leakage` passes locally (required if touching `src/splits/`, `src/fs/`, or `src/optimization/`)
- [ ] Any new predictor is registered in `config/feature_registry.yaml`
