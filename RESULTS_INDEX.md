# Published manuscript results index

Exact active published-table snapshot, not independent recomputation. LaTeX cells are authoritative; text is a convenience projection. Stored intervals are not regenerated.

Every active rendered table, including specifications and context counts, is included. CSVs use generic tabular/row/cell coordinates and preserve LaTeX plus readable text; no metric is guessed. Label aliases share one table export.

| Published table | Primary label | CSV | Canonical aggregate families / role |
|---|---|---|---|
| Table 1 (p. 3) | `tab:label-semantics` | [CSV](aggregates/manuscript_tables/table_1.csv) | specification |
| Table 2 (p. 7) | `tab:aggregation_effects` | [CSV](aggregates/manuscript_tables/table_2.csv) | [baseline_comparison.json](aggregates/baseline_comparison.json) |
| Table 3 (p. 7) | `tab:main_benchmark` | [CSV](aggregates/manuscript_tables/table_3.csv) | [baseline_comparison.json](aggregates/baseline_comparison.json), [catalogue_and_frozen_heads.json](aggregates/catalogue_and_frozen_heads.json) |
| Table 4 (p. 7) | `tab:query_intervention_controls` | [CSV](aggregates/manuscript_tables/table_4.csv) | [query_interventions_and_scale.json](aggregates/query_interventions_and_scale.json) |
| Table A1 (p. 13) | `tab:protocol-counts` | [CSV](aggregates/manuscript_tables/table_A1.csv) | [data_training_runtime.json](aggregates/data_training_runtime.json) |
| Table A2 (p. 13) | `tab:label-distributions` | [CSV](aggregates/manuscript_tables/table_A2.csv) | [data_training_runtime.json](aggregates/data_training_runtime.json) |
| Table A3 (p. 14) | `tab:weak_label_rules` | [CSV](aggregates/manuscript_tables/table_A3.csv) | specification |
| Table A4 (p. 14) | `tab:data-overlap` | [CSV](aggregates/manuscript_tables/table_A4.csv) | [data_training_runtime.json](aggregates/data_training_runtime.json) |
| Table A5 (p. 14) | `tab:stage1-stage2-bridge` | [CSV](aggregates/manuscript_tables/table_A5.csv) | specification |
| Table A6 (p. 14) | `tab:query_intervention_agreement` | [CSV](aggregates/manuscript_tables/table_A6.csv) | [query_interventions_and_scale.json](aggregates/query_interventions_and_scale.json) |
| Table A7 (p. 15) | `tab:appendix_temporal_controls_full` | [CSV](aggregates/manuscript_tables/table_A7.csv) | [temporal_details.json](aggregates/temporal_details.json), [temporal_metrics.tsv](aggregates/temporal_metrics.tsv) |
| Table A8 (p. 15) | `tab:appendix_per_finding_controls` | [CSV](aggregates/manuscript_tables/table_A8.csv) | [canonical_diagnostics.json](aggregates/canonical_diagnostics.json) |
| Table A9 (p. 15) | `tab:appendix_transfer` | [CSV](aggregates/manuscript_tables/table_A9.csv) | [secondary_diagnostics.json](aggregates/secondary_diagnostics.json) |
| Table A10 (p. 16) | `tab:appendix_secondary_diagnostics` | [CSV](aggregates/manuscript_tables/table_A10.csv) | [canonical_diagnostics.json](aggregates/canonical_diagnostics.json), [baseline_comparison.json](aggregates/baseline_comparison.json), [query_interventions_and_scale.json](aggregates/query_interventions_and_scale.json), [catalogue_and_frozen_heads.json](aggregates/catalogue_and_frozen_heads.json) |
| Table A11 (p. 16) | `tab:fixed_prompt_catalogue` | [CSV](aggregates/manuscript_tables/table_A11.csv) | [catalogue_and_frozen_heads.json](aggregates/catalogue_and_frozen_heads.json) |
| Table A12 (p. 17) | `tab:existing_fifty_review` | [CSV](aggregates/manuscript_tables/table_A12.csv) | [existing_50_image_review.json](aggregates/existing_50_image_review.json) |
| Table A13 (p. 17) | `tab:matched_finding_blind` | [CSV](aggregates/manuscript_tables/table_A13.csv) | [baseline_comparison.json](aggregates/baseline_comparison.json) |
| Table A14 (p. 17) | `tab:detailed_query_swap` | [CSV](aggregates/manuscript_tables/table_A14.csv) | [canonical_diagnostics.json](aggregates/canonical_diagnostics.json) |
| Table A15 (p. 18) | `tab:query_swap_finding_pairs` | [CSV](aggregates/manuscript_tables/table_A15.csv) | [canonical_diagnostics.json](aggregates/canonical_diagnostics.json) |
| Table A16 (p. 18) | `tab:finding_blind_oracle` | [CSV](aggregates/manuscript_tables/table_A16.csv) | [canonical_diagnostics.json](aggregates/canonical_diagnostics.json) |
| Table A17 (p. 19) | `tab:baseline_metric_bridge` | [CSV](aggregates/manuscript_tables/table_A17.csv) | [baseline_comparison.json](aggregates/baseline_comparison.json) |
| Table A18 (p. 19) | `tab:appendix_nse_stress` | [CSV](aggregates/manuscript_tables/table_A18.csv) | [temporal_details.json](aggregates/temporal_details.json) |
| Table A19 (p. 19) | `tab:appendix_nse_hard_negative` | [CSV](aggregates/manuscript_tables/table_A19.csv) | [secondary_diagnostics.json](aggregates/secondary_diagnostics.json) |
| Table A20 (p. 19) | `tab:appendix_seed_stability` | [CSV](aggregates/manuscript_tables/table_A20.csv) | [baseline_comparison.json](aggregates/baseline_comparison.json) |

## Figure and protocol context

Q-LACE pipeline. Report-derived weak supervision adapts a medical VLM for held-out finding-conditioned interval-change classification using 2,048 weak-training examples per run. Adaptation includes language/vision LoRA and trainable input embeddings and output head; other base weights remain frozen. Historical validation-monitoring schedules are specified in Appendix A.10. "No evidence" is an appendix diagnostic, while the three change labels define the benchmark task. The illustrative radiographs are separate NIH Clinical Center images from different patients, arranged to depict input roles; they are not a real longitudinal pair or study evaluation case. NIH credit and download: https://nihcc.app.box.com/v/ChestXray-NIHCC; [wang2017chestxray]. Author-created schematic, exported with Inkscape 1.4.4/cairo 1.18.4.

Protocol counts are in Table A1 and its table export above.

Figure A1 graph data: [figure_A1_data.json](aggregates/figure_A1_data.json), 13 panel-B rows and 12 panel-C comparisons, excluding the uncertified comparison. Connectors are metric differences, not confidence limits. The primary run and three-random-seed mean are distinct.

## Results reported in prose

[data_training_runtime.json](aggregates/data_training_runtime.json) also covers configured trainability, reconstructed training distributions, descriptive PTX errors, dataset/overlap and physical-pair counts, runtime input/parsing checks, and heterogeneous compute accounting. These are indexed context/results, not newly inferred metrics.

Run `python tools/verify_manuscript_results.py` offline to check coverage and JSON/CSV/figure consistency. This verifies the exported snapshot, not the original experiments, patient-bootstrap intervals, restricted case-level inputs or manuscript typesetting.
