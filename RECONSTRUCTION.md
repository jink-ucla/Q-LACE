# Credentialed local reconstruction

All commands run from the extracted archive root. The recipe documents the
three random-seed runs included in the manuscript and their reported controls;
it does not certify historical row identity.
Paths below are placeholders
chosen by the user; keep generated manifests, reports, images, annotations, and
predictions on authorized local storage. Do not upload them with this software.

## 1. Obtain inputs independently

Follow the respective credentialing and data-use processes for:

- MIMIC-CXR-JPG 2.0.0: <https://physionet.org/content/mimic-cxr-jpg/2.0.0/>
- MIMIC-CXR reports: <https://physionet.org/content/mimic-cxr/2.0.0/>
- MS-CXR-T 1.0.0: <https://physionet.org/content/ms-cxr-t/1.0.0/>
- Chest ImaGenome 1.0.0: <https://physionet.org/content/chest-imagenome/1.0.0/>

Obtain MedGemma-1.5-4B-IT under its own model terms and approval. One acquisition
method, **only after access is granted**, is:

```bash
hf download google/medgemma-1.5-4b-it --revision e9792da5fb8ee651083d345ec4bce07c3c9f1641 --local-dir models/medgemma
```

This revision identifies the current local evaluation download, not Google's
unavailable evaluation revision or the missing historical temporary base copies.
No model files are in this archive. Check availability and terms at the source;
the archive does not grant access or model/data redistribution rights.

Set task-specific path variables (adapt these local examples):

```bash
QL_DATA=/authorized/data
QL_WORK=/authorized/qlace-reconstruction
QL_MODEL=models/medgemma
```

`QL_DATA/mimic` below denotes the MIMIC-CXR-JPG root, whose `files/` directory
contains the `pXX/pID/sSTUDY/DICOM.jpg` hierarchy. The report ZIP must contain
`files/pXX/pID/sSTUDY.txt`, as expected by the original builder; reconstructing
that ZIP locally from licensed reports does not authorize its redistribution.

## 2. Reconstruct benchmark and projected secondary manifests

```bash
python dataset/build_ms_cxr_t_longitudinal_stage1.py --ms-cxr-t-csv "$QL_DATA/ms-cxr-t/MS_CXR_T_temporal_image_classification_v1.0.0.csv" --mimic-metadata-csv "$QL_DATA/mimic/mimic-cxr-2.0.0-metadata.csv" --mimic-jpg-root "$QL_DATA/mimic/files" --output-dir "$QL_WORK/benchmark" --split-policy all_test
python dataset/build_chest_imagenome_gold_stage1.py --gold-tsv "$QL_DATA/chest-imagenome/gold_dataset/gold_object_comparison_with_coordinates.txt" --mimic-metadata-csv "$QL_DATA/mimic/mimic-cxr-2.0.0-metadata.csv" --mimic-jpg-root "$QL_DATA/mimic/files" --output-dir "$QL_WORK/transfer" --normalize-to-ms-cxr-t-heads
```

The benchmark's expected evaluation population is 1,326 query occurrences,
800 patients, and 1,045 physical image pairs. Repeated source IDs are not a reason
to discard rows. The projected secondary export is not external institutional
validation. Inspect each builder's summary and keep the original source versions.

## 3. Replace the missing custom master CSV

```bash
python tools/build_master_csv.py --metadata-csv "$QL_DATA/mimic/mimic-cxr-2.0.0-metadata.csv" --split-csv "$QL_DATA/mimic/mimic-cxr-2.0.0-split.csv" --jpg-root "$QL_DATA/mimic/files" --output "$QL_WORK/master.csv"
```

The tool joins the official files on subject, study, and image identifiers;
preserves metadata order; derives image paths; and rejects duplicate/missing join
keys, unsafe identifiers, or missing image files. Unmatched records cause validation to fail. This reconstructs the columns needed by the chronology builder
but does not assert byte identity to the historical custom `master_modified.csv`.

## 4. Build overlap-excluded weak supervision

```bash
python dataset/build_mimic_cxr_longitudinal_manifest.py --mimic-root "$QL_DATA/mimic" --master-csv "$QL_WORK/master.csv" --metadata-csv "$QL_DATA/mimic/mimic-cxr-2.0.0-metadata.csv" --reports-zip "$QL_DATA/mimic/mimic-cxr-reports.zip" --output-dir "$QL_WORK/chronology" --max-pairs 10000000 --include-report-sections --require-reports --exclude-records "$QL_WORK/benchmark/test.jsonl" --exclude-records "$QL_WORK/transfer/test.jsonl"
python dataset/build_mimic_cxr_longitudinal_weak_tuples.py --input-manifest "$QL_WORK/chronology/manifest.jsonl" --output-dir "$QL_WORK/weak"
python dataset/build_mimic_cxr_longitudinal_stage1_benchmark.py --input-jsonl "$QL_WORK/weak/weak_tuples.jsonl" --output-dir "$QL_WORK/train" --same-view-only --split-policy patient_hash --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15
python scripts/audit_longitudinal_benchmark_overlap.py --local "$QL_WORK/train" --benchmark "$QL_WORK/benchmark/test.jsonl" --benchmark "$QL_WORK/transfer/test.jsonl" --sample-limit 0 --json-output "$QL_WORK/overlap.json"
```

The chronology builder's default cap is a 128-pair preview. A large explicit cap
is used above; zero does **not** disable the cap in the preserved implementation.
The strict historical source has 25,261 training and 5,459 weak-validation rows.
New reconstructions can differ because original intermediate/source snapshots
were not completely retained. Investigate and report differences; never delete
or add rows merely to match these totals. Overlap checking does not validate
report extraction or which prior examination a report references.

For the full-pool frozen-head protocol, omit only the Chest ImaGenome exclusion
and construct a separate source tree; do not substitute that source for the
stricter Q-LACE training pool. Additional encoder software is not vendored here.

## 5. Explicit training recipe and a new finding-blind comparison

Set `QL_RUN_SEED` to a locally chosen integer for the run.

```bash
python tools/run_paper_training.py --model "$QL_MODEL" --data "$QL_WORK/train" --output "$QL_WORK/run1/adapter" --seed "$QL_RUN_SEED"
```

By default this **prints** the full command without training. Add `--execute`
only when deliberately allocating a GPU. The command explicitly sets 2,048
training/512 validation caps, one epoch, batch size/accumulation 1, bf16, eager
attention, no 4-bit quantization, length 1,024, learning rate 2e-4, LoRA r/alpha 16,
dropout 0.05, all linear modules, and fully trainable saved input embeddings and
output head. Validation is not used for checkpoint selection; the final
checkpoint is saved. The first two archived full-query runs have 512 validation
rows and stepwise evaluation; the third archived fast run has validation
disabled (use `--no-eval`), despite retaining the configured 512-row cap.
The archived finding-blind controls also have evaluation disabled. Match and
disclose these run-level differences rather than claiming identical historical
evaluation schedules. No `/tmp` model override or finding-first
intervention is used. This is **not** LoRA-only adaptation.

Use three random seeds and label the results Run 1--3.
The training sampler shuffles before taking the cap, with validation using the
training seed plus one. Preserve and hash all generated row lists in a new run.

To create matched finding-blind sources, run the following for each split:

```bash
python tools/make_finding_blind.py --input "$QL_WORK/train/train.jsonl" --output "$QL_WORK/blind/train.jsonl"
python tools/make_finding_blind.py --input "$QL_WORK/train/val.jsonl" --output "$QL_WORK/blind/val.jsonl"
```

For a new controlled comparison, train with the identical seed and complete
settings, including evaluation schedule, on `blind/`. Before starting either arm,
save immutable sampled-row manifests and hashes of the exact base weights,
processor files, training code, and complete arguments. The recipe alone cannot
recover the missing historical sampled rows or temporary base copies. The transform preserves
order, images and targets and replaces only query identity; label-mode training
rebuilds the actual prompt. No claim of natural-language superiority over a
finding ID or a finding-specific head follows from this control.

## 6. New three-option inference and scoring

The adapter path below refers to a checkpoint generated by the new training in
Section 5. It is not a historical adapter download: historical trained adapters
are neither bundled nor publicly distributed through this package. Evaluating
an exact archived checkpoint requires the authors to supply that adapter and
compatible base/processor assets separately, if available and permitted.
Downloading the base model alone does not reproduce the adapted-model rows.

```bash
python scripts/cmig_medgemma_baseline.py prepare --input-jsonl "$QL_WORK/benchmark/test.jsonl" --output-dir "$QL_WORK/eval_manifest"
python scripts/cmig_medgemma_baseline.py generate --manifest "$QL_WORK/eval_manifest/evaluation_manifest.jsonl" --protocol protocols/protocol_native_mcq.json --model-path "$QL_MODEL" --output-dir "$QL_WORK/base_native"
python scripts/cmig_medgemma_baseline.py generate --manifest "$QL_WORK/eval_manifest/evaluation_manifest.jsonl" --protocol protocols/protocol_native_mcq.json --model-path "$QL_MODEL" --adapter-path "$QL_WORK/run1/adapter" --output-dir "$QL_WORK/adapter_native"
python scripts/cmig_medgemma_baseline.py score --manifest "$QL_WORK/eval_manifest/evaluation_manifest.jsonl" --predictions "$QL_WORK/base_native/predictions.jsonl" --output "$QL_WORK/base_native/rescored.json"
python tools/compare_local_predictions.py --manifest "$QL_WORK/eval_manifest/evaluation_manifest.jsonl" --arm base="$QL_WORK/base_native/predictions.jsonl" --arm adapted="$QL_WORK/adapter_native/predictions.jsonl" --output "$QL_WORK/paired_aggregate.json"
```

The padding sensitivity changes only the protocol argument to
`protocols/protocol_square_pad_mcq.json` and uses a separate output directory.
Keep the same model, processor, hardware and runtime for a clean new contrast.
For a three-run mean, compute the mean of three run-level metrics, not a voting
ensemble; the bundled historical aggregate JSON documents the study's averaging.
The generic explicit comparator reports individual supplied-arm contrasts.
Patient bootstrap uncertainty conditions on supplied runs and does not estimate
variation over newly trained seeds. Invalid/NSE outputs remain errors.

The scripts write restricted local manifests/predictions; do not redistribute
those outputs. Package source hashes intentionally differ after portability
edits. New runs record their own content hashes, not the original frozen hashes.
