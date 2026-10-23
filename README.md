# Q-LACE

Code repository: [github.com/jink-ucla/Q-LACE](https://github.com/jink-ucla/Q-LACE).

**Separating Benchmark Performance from Conditional Reliability in
Longitudinal Chest Radiograph Assessment**

Q-LACE is an experimental framework for asking whether benchmark improvement
is accompanied by reliable finding-conditioned longitudinal behavior. It
separates three questions: what adaptation changes, whether predictions follow
the requested finding, and whether they respond consistently to temporal
interventions. Absolute accuracy or macro-F1 alone does not answer all three.

The associated study uses MedGemma-1.5 4B and report-derived weak supervision.
Its aggregate adaptation gains do not establish comparable gains across
equally weighted finding–class strata or reliable paired-query behavior.
The historical finding-blind comparison is retrospective, not a fully
controlled causal attribution.

## Start here

Use Python 3.10 or later. CPU verification needs no patient data, model weights,
GPU, authentication, or network access after installing the dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.txt
python -m pytest -q tests
python tools/verify_manifest.py
python tools/paper_run_scope.py
python tools/verify_aggregate_tables.py
python tools/verify_manuscript_results.py
python tools/check_public_release.py --history
```

The tests use synthetic examples. They do not train or download a model.
GPU dependencies in `requirements-gpu.txt` are an inventory, not a validated
historical environment lockfile; see [ENVIRONMENT.md](ENVIRONMENT.md).

## Code and workflows

| Question | Entry point or guide |
|---|---|
| Obtain licensed inputs and reconstruct local datasets | [Data access](docs/DATA_ACCESS.md), [reconstruction recipe](RECONSTRUCTION.md), `dataset/` |
| Train an adapter or prepare a finding-blind control | `tools/run_paper_training.py`, `tools/make_finding_blind.py` |
| Run the fixed three-option interface | `scripts/cmig_medgemma_baseline.py`, `protocols/` |
| Compare pooled and finding-balanced metrics | `tools/compare_local_predictions.py` |
| Construct and analyze finding/temporal interventions | [Conditional reliability workflow](docs/CONDITIONAL_RELIABILITY.md) |
| Check the manuscript's reported results | [RESULTS_INDEX.md](RESULTS_INDEX.md), `aggregates/` |

The training wrapper prints the proposed command by default; starting training
requires an explicit `--execute`. Adaptation includes language/vision LoRA
**and trainable input embeddings and output head**; it is not LoRA-only.

Use the explicit-input tools for new local analyses. The `cmig_*` historical
analysis scripts preserve reusable scoring functions and study mappings, but
their archival `main()` entry points require separately held study artifacts.
The older `eval_ms_cxr_t_classification.py` source-ID scorer is not the
recommended evaluation workflow. Occurrence-preserving identities must include
the actual prior image, current image, finding, and repeated occurrence.

## Reported results

[RESULTS_INDEX.md](RESULTS_INDEX.md) maps all **24 manuscript tables**—four
main-text tables and A1–A20—and Appendix Figure A1's numerical panels to CSV/JSON.
The repository includes permitted aggregate comparisons, query interventions,
temporal controls, scale/budget analyses, transfer diagnostics, and descriptive
50-case review summaries. Table snapshots retain the manuscript's rounding;
aggregate confusion matrices support separate arithmetic checks.

The task means use **three random seeds**, labeled Run 1–3. Native-interface
evaluations use those same adapters. Primary-run behavioral diagnostics are
identified separately; `PAPER_RUN_SCOPE.json` records the aggregate membership.

The verification tools recompute available point metrics, means, sample SDs,
and contrasts. Reported patient-bootstrap confidence intervals are included,
but reproducing the resampling requires restricted row-level predictions and
patient mappings; aggregate counts alone are insufficient.

## Data and model access

This is a **code and aggregate-results repository**, not a release of
patient-level research data. It includes no radiographs, source radiology
reports, patient-linked records, raw predictions, or model/adapter tensors.
Obtain source datasets and models independently under their respective access
agreements. Keep generated manifests, predictions, checkpoints, and credentials
outside Git. See [docs/DATA_ACCESS.md](docs/DATA_ACCESS.md).

Original per-run sampled-row lists and temporary base-model copies were not
retained; validation-monitoring schedules also differed. The documented recipe
supports new runs, not byte-identical recovery of historical training. The
published MedGemma evaluation score was not recovered locally. Direct weak-label
precision and report-to-supplied-prior compatibility remain unmeasured; the
selected single-reader image review is not a validation of those properties or
clinical deployment.

## Manuscript and attribution

Associated manuscript: *Q-LACE: Separating Benchmark Performance from
Conditional Reliability in Longitudinal Chest Radiograph Assessment*.

Authors: Jin Kim, Fereidoun Abtin, Matthew S. Brown, and Dan Ruan.
Software citation metadata are in [CITATION.cff](CITATION.cff).

The source snapshot and aggregates derive from Supplementary Software S1;
[PORTABILITY.md](PORTABILITY.md) and `provenance/` record the relationship.
Code is distributed under the existing [Apache License 2.0](LICENSE), with
[Visual-RFT](https://github.com/Liuziyu77/Visual-RFT) attribution in
[NOTICE](NOTICE). Dataset and model terms are separate; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
