# Portable snapshot versus executing research sources

FILE_MANIFEST.json records SHA-256 hashes for the research sources and their
portable copies. Portable copies replace owner-specific default paths and mount
aliases; supplied model and data arguments select the user's local assets.

New runs record their current source, model, image and protocol hashes.
Verification of archived executions uses their recorded source hashes and run
files. The legacy analysis entrypoints refer to the study's archived files;
the explicit-argument tools support analysis of locally available predictions.

RECONSTRUCTION.md documents the chronology cap and training settings.
Newly authored reconstruction utilities accompany the source snapshot.
Tests use synthetic identifiers and paths. The archive contains code,
protocols and aggregate results.

## Public repository

The public repository starts from the verified Supplementary Software S1 text
payload, without the development repository's Git history. The source S1
manifest is retained in `provenance/s1_file_manifest.json`; the current
`FILE_MANIFEST.json` covers the public repository's payload.

The public README, data-access guide, citation metadata, upstream attribution,
Git ignore rules, and CPU CI workflow describe repository use. The third-party
notice now describes only the code-only release. Explicit-input conditional
reliability tools and synthetic tests support new local analyses without the
historical scripts' archival path assumptions. Public-release checks inspect
payload integrity, aggregate boundaries, and Git objects. These are software
portability additions, not new study results.

`tools/build_query_swap_controls.py` adapts the exact-image construction logic
from `scripts/build_query_swap_controls.py` with occurrence-preserving native
row identities. `tools/analyze_conditional_reliability.py` implements the
canonical query/temporal metric definitions with explicit supplied inputs and
uses the included native inference harness. Their synthetic regression tests
and workflow guide are new. Documentation-command tests additionally parse
the public conditional-reliability examples.

All aggregate-result files are preserved byte-for-byte from the source S1.
