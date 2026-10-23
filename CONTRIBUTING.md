# Contributing

Run the CPU tests, aggregate checks, manifest verification, and public-release
check described in README.md before proposing a change.

Use synthetic examples in tests and bug reports. Do not attach source medical
images, reports, patient-linked identifiers, manifests, predictions, credentials,
or model tensors. Keep data-dependent runs on authorized local storage.

When changing a derived source, preserve existing attribution and record the
change in PORTABILITY.md. Regenerate FILE_MANIFEST.json with
`python tools/update_file_manifest.py` after intentional payload changes.
Aggregate files represent the manuscript's reported study; changes to a
published estimate require a corresponding documented correction, not an
automatic overwrite from a new experiment.
