# Source data and access

The repository distributes software, synthetic tests, protocols, and aggregate
statistics. It does not distribute patient-level data or grant dataset access.

## Required source resources

Obtain access and the specified source versions directly from the providers:

| Resource | Role |
|---|---|
| [MIMIC-CXR 2.0.0](https://physionet.org/content/mimic-cxr/2.0.0/) | Radiology reports and chronology |
| [MIMIC-CXR-JPG 2.0.0](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) | Images, metadata, and source splits |
| [MS-CXR-T 1.0.0](https://physionet.org/content/ms-cxr-t/1.0.0/) | Finding-conditioned temporal evaluation |
| [Chest ImaGenome 1.0.0](https://physionet.org/content/chest-imagenome/1.0.0/) | Projected secondary transfer evaluation |

The relevant PhysioNet pages specify credentialing, required training, and a
signed data-use agreement. Follow the terms applicable to each resource. The
[PhysioNet Credentialed Health Data License 1.5.0](https://physionet.org/about/licenses/physionet-credentialed-health-data-license-150/)
restricts sharing access to its data and requires associated publication code
to be contributed to a repository open to the research community. A code
release does not authorize redistribution of restricted source data.

Keep radiographs, reports, identifiers, patient mappings, case annotations,
generated manifests, and row-level predictions on authorized local storage.
Do not paste such content into public issues, pull requests, or example files.
Only the aggregate summaries supplied in `aggregates/` are release payloads.

## Models

Obtain [MedGemma-1.5 4B](https://huggingface.co/google/medgemma-1.5-4b-it)
under its separate model terms. The reconstruction recipe records the revision
used for the local fixed-interface evaluation; it does not identify the missing
historical temporary base copies. No weights or trained adapter are bundled.

[RECONSTRUCTION.md](../RECONSTRUCTION.md) documents an `hf download` command
to use after access is granted. Authentication is performed by the researcher;
never commit access tokens or put them into a command intended for sharing.

## What the included results support

The aggregate files permit checks of reported counts, point estimates, means,
sample SDs, table snapshots, and figure arithmetic. They do not reconstruct
patient-level examples or permit a fresh patient-bootstrap analysis. Historical
checkpoint evaluation requires the compatible adapters and base/processor
assets separately, where available and permitted. A new training run is not
an exact historical-run reconstruction.

Use [RESULTS_INDEX.md](../RESULTS_INDEX.md) to locate the manuscript results and
[RECONSTRUCTION.md](../RECONSTRUCTION.md) for the authorized local workflow.
