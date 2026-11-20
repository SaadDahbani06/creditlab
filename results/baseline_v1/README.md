# Frozen baseline results (CreditLab v1)

This folder is a read-only copy of the original research outputs, taken on 2026-11-19 **before**
the PD-estimation-uncertainty extension was started. Nothing in this folder may be edited.
If a later run of `creditlab all` produces different numbers, this folder is the reference.

| Contents | Source |
|---|---|
| `metrics/` | `artifacts/metrics/` (selection, validation and test metrics, risk, policy) |
| `predictions/` | `artifacts/predictions/` (validation and test PDs for all 7 candidates) |
| `models/` | training metadata and CV tables (the model binaries are not copied; see checksums) |
| `SHA256SUMS_original_artifacts.txt` | SHA-256 of every original artifact, the model binaries and the raw data file |

## Status of the original evaluation

* All model, calibration and threshold choices were made on the validation split, and the test
  split was then scored (`selection.json` was written before the test split was loaded).
* **The test split is not "untouched".** Test metrics were printed during development, *before*
  the calibration-selection rule was tightened (decision D9 in `docs/DECISIONS.md`). The
  independent review (`docs/INDEPENDENT_REVIEW.md` §2.1) shows that the change did not alter the
  primary model or any reported test number. It remains a procedural deviation and is disclosed.
* A full re-run on the original machine reproduced every file here byte for byte
  (`docs/INDEPENDENT_REVIEW.md` §4.1).
* Everything produced after this snapshot, including the PD-uncertainty extension, is
  **post-hoc exploratory analysis** on the same data. No genuinely independent (new, untouched)
  validation data exists for this project.

Verify the integrity of the live artifacts against this snapshot with:

```bash
cd artifacts && shasum -a 256 -c ../results/baseline_v1/SHA256SUMS_original_artifacts.txt --ignore-missing
```
