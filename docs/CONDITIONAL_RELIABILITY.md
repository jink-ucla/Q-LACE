# Rerunning conditional-reliability diagnostics

These CPU tools separate benchmark classification from query-change compliance,
paired correctness, and temporal consistency. They operate on newly generated
local predictions; they do not recover missing historical training snapshots
or claim byte-identical reproduction of the manuscript's archived runs.

Obtain restricted datasets and MedGemma independently as described in
[RECONSTRUCTION.md](../RECONSTRUCTION.md). All manifests, images and predictions
below remain on authorized local storage. Only the analysis JSON outputs contain
aggregate results. A repository code license does not authorize redistribution
of generated patient-level manifests or predictions.

## Native input contract

Prepare an occurrence-preserving evaluation manifest with the included harness:

```bash
QL_WORK=/authorized/qlace-work
QL_MODEL=/authorized/models/medgemma
python scripts/cmig_medgemma_baseline.py prepare --input-jsonl "$QL_WORK/benchmark/test.jsonl" --output-dir "$QL_WORK/native"
python scripts/cmig_medgemma_baseline.py generate --manifest "$QL_WORK/native/evaluation_manifest.jsonl" --protocol protocols/protocol_native_mcq.json --model-path "$QL_MODEL" --output-dir "$QL_WORK/native_predictions"
```

`generate` requires a compatible GPU runtime and, for adapted-model evaluation,
an additional `--adapter-path` pointing to an independently available adapter.
Preparation and analysis are CPU-only and never start inference. The harness
expects 1,326 rows by default; for another population explicitly pass its actual
`--expected-rows` to both harness commands. Preserve the same frozen protocol,
checkpoint and runtime when comparing conditions.

The native manifest identifies each occurrence by its row index and a hash of
actual prior-image path, current-image path and finding. Predictions must contain
the same `row_id`, `row_index`, `physical_key`, `query_finding`, `target_label`,
and `patient_id`, plus `pred_label`. Duplicate legacy `source_id` or `pair_id`
values are never used to merge records. Every prediction must be present exactly
once; additional, missing, duplicate, or physically mismatched records fail.

## Query conditioning

For each exact patient/prior/current image group, the analysis enumerates all
directed source-to-target occurrences with different finding identities. Every
original row remains distinct, including repeated physical finding occurrences.
Pairs with different actual priors are never pooled merely because their old
`pair_id` matches.

```bash
python tools/analyze_conditional_reliability.py query --manifest "$QL_WORK/native/evaluation_manifest.jsonl" --predictions "$QL_WORK/native_predictions/predictions.jsonl" --output "$QL_WORK/query_aggregate.json"
```

This reuses the target finding's existing prediction because its images and
finding are exactly the swapped input. The output names this reuse explicitly;
it does not claim new model generations were performed. To generate each
directed control separately instead:

```bash
python tools/build_query_swap_controls.py --manifest "$QL_WORK/native/evaluation_manifest.jsonl" --output "$QL_WORK/query_swaps.jsonl"
python scripts/cmig_medgemma_baseline.py generate --manifest "$QL_WORK/query_swaps.jsonl" --expected-rows "$QL_SWAP_ROWS" --protocol protocols/protocol_native_mcq.json --model-path "$QL_MODEL" --output-dir "$QL_WORK/swap_predictions"
python tools/analyze_conditional_reliability.py query --manifest "$QL_WORK/native/evaluation_manifest.jsonl" --predictions "$QL_WORK/native_predictions/predictions.jsonl" --swap-manifest "$QL_WORK/query_swaps.jsonl" --swap-predictions "$QL_WORK/swap_predictions/predictions.jsonl" --output "$QL_WORK/generated_swap_aggregate.json"
```

Set `QL_SWAP_ROWS` to the number of directed rows printed by the builder. The
builder writes a restricted local manifest, not a public research-data archive.

The principal outputs have different denominators and meanings:

- Required-change compliance: predictions differ on different-gold-label swaps.
- Paired target success: both original and target predictions are correct on
  those different-label swaps.
- Same-label invariance: predictions remain unchanged on same-gold-label swaps.
- Same-label paired success and target accuracy: correctness measures reported
  separately from label-changing behavior.

Compliance and invariance are descriptive output comparisons, not correctness.
For example, changing from an invalid answer to a label can count as a change
without yielding paired success. All invalid answers are normalized to one
invalid bucket; NSE (`no_supported_evidence`) remains a separate output bucket.
Both are errors on supported three-class gold labels. Invalid/NSE pair counts
are included so those behaviors are visible, and no such rows are dropped.

## Temporal consistency

Supply a local uniformly black image for prior-drop. The preparer verifies its
pixels are black; it does not create or download a radiograph. Set `QL_CONTROL_SEED`
to a locally chosen integer for deterministic random-prior donor selection.

```bash
python tools/analyze_conditional_reliability.py prepare-temporal --manifest "$QL_WORK/native/evaluation_manifest.jsonl" --blank-prior-image "$QL_WORK/black.png" --seed "$QL_CONTROL_SEED" --output-dir "$QL_WORK/temporal"
for condition in baseline prior_drop duplicate_pair time_reversal random_prior; do
  python scripts/cmig_medgemma_baseline.py generate --manifest "$QL_WORK/temporal/$condition.jsonl" --protocol protocols/protocol_native_mcq.json --model-path "$QL_MODEL" --output-dir "$QL_WORK/temporal_predictions/$condition"
done
python tools/analyze_conditional_reliability.py temporal --suite-dir "$QL_WORK/temporal" --predictions "$QL_WORK/temporal_predictions" --output "$QL_WORK/temporal_aggregate.json"
```

Use the same optional adapter and nondefault population count in every
generation command where applicable. Each condition retains all baseline
occurrences and an explicit anchor to its original row:

- Baseline: original inputs and targets, generated within this temporal suite.
- Prior-drop: shared supplied black image replaces the prior.
- Duplicate pair: current image is supplied twice.
- Time reversal: actual prior/current paths are exchanged and improving/worsening
  gold labels are inverted; stable is preserved.
- Random prior: a deterministic donor prior comes from another patient in the
  supplied manifest. A dataset with no other patient is rejected, not assigned
  a same-patient fallback. Donors are not label-rematched.

Prior-drop and random-prior F1 differences use only the supplied suite baseline,
never a separately generated standalone baseline. The scorer validates physical
transformations and lineage; a condition cannot silently omit occurrences.
These three-class scores retain original task targets for corrupted-input
diagnostics; they are not newly adjudicated clinical ground truth.

Temporal reversal consistency inverts improving/worsening predictions and keeps
stable/NSE unchanged. Invalid predictions never pass. A second supported-only
consistency rate excludes NSE successes from the numerator while retaining all
rows in the denominator. Baseline-and-reversal paired correctness is reported
separately, since consistency alone does not imply correctness.

Duplicate stable-output rate uses **all** duplicate-image occurrences. Its
complement includes changed labels, NSE and invalid outputs; a separate
change-label rate counts only improving/worsening. The tool does not report
duplicate-pair classification F1 against unchanged original targets.

## Output and uncertainty

Analysis files contain counts, rates, supported macro-F1, confusion matrices and
condition deltas, not patient identifiers, image paths or row records. The
portable tools report descriptive point estimates only; they do not attach
uncomputed confidence intervals. Manuscript intervals remain stored reported
results in `aggregates/`. New patient-bootstrap analyses require the appropriate
locally retained patient-level predictions and a declared resampling procedure.

Run synthetic tests without a GPU, network, model or restricted dataset:

```bash
python -m pytest -q tests/test_public_conditional_reliability.py
```
