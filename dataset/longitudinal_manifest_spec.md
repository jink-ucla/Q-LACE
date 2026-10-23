# Longitudinal MIMIC-CXR Manifest Spec

This spec covers the unlabeled chronology manifest emitted by
`dataset/build_mimic_cxr_longitudinal_manifest.py`.

It is not a supervised training schema. It is a pairing/provenance contract
for prior/current chest X-ray studies reconstructed from the local
`mimic-cxr-jpg` mirror.

## Required Fields

- `pair_id`: `{patient_id}:{prior_study_id}:{current_study_id}`
- `patient_id`
- `source_id`: must equal `pair_id`
- `split`: one of `train`, `validate`, `test`
- `prior_study_id`, `current_study_id`
- `prior_dicom_id`, `current_dicom_id`
- `prior_image_path`, `current_image_path`
- `prior_view_position`, `current_view_position`: currently restricted to `PA` or `AP`
- `problem`: a baseline prompt string for later longitudinal experiments
- `metadata`

## Optional Fields

- `delta_days`: non-negative integer day gap between prior/current studies
- `prior_report_path`, `current_report_path`
- `prior_report_sections`, `current_report_sections`
- `prior_study_date`, `current_study_date`
- `prior_study_time`, `current_study_time`

## Metadata Contract

The current builder emits:

```json
{
  "builder": "build_mimic_cxr_longitudinal_manifest.py",
  "task_family": "longitudinal_cxr_change_rft",
  "label_status": "unlabeled_manifest"
}
```

Treat `label_status = unlabeled_manifest` literally. These rows are chronology
artifacts, not interval-change labels.

## Report Sections

`prior_report_sections` and `current_report_sections` are best-effort heading
extracts from `mimic-cxr-reports.zip`.

Canonical downstream keys to rely on:

- `comparison`
- `findings`
- `impression`

Other extracted headings may appear and should be treated as optional extras.

## Validation

Use `scripts/validate_mimic_cxr_longitudinal_manifest.py` or
`medgemma_visual_rft.longitudinal_manifest.validate_manifest_record(...)`
before consuming the manifest downstream.
