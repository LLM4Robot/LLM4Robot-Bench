# Release-preparation validation · 2026-10-21

The table below records the original release-preparation checks at commit
`6adaa49b8b7a2986b469df451716350f34de0aa8`, before sequential renumbering.
Task labels in this document use the current [numbering map](TASK_NUMBERING.md).

The reference task state was captured before release edits. The public naming
change was validated against this baseline rather than regenerating the baseline
after changing tasks. Derived-float fingerprints use 12 significant digits (magnitudes below 1e-12 treated as zero);
all 247 source/configuration file hashes remain exact. The portability migration
was derived from 256 verified pre-change component fingerprints, not a new
post-change task baseline.

| Check | Result |
|---|---|
| Effective task contract | 32/32 unchanged; 247 protected files unchanged; passes local Linux, Windows and cluster Python |
| Runtime assets | 325 files, 1,504,358,748 bytes; original SHA-256 inventory unchanged |
| USD dependency closure in a clean checkout | 22 delivered/shared scene entries resolved; only Isaac built-in MDL is external |
| Actual Isaac startup on the maintained cluster | 32/32 passed; normal initialization and three image views captured |
| Scene and visual setup hashes | 32/32 identical to pre-release baseline |
| Robot reset readings and camera calibration | 32/32 identical to pre-release baseline |
| Rigid initial state and initial evaluation | 28/28 identical; remaining four configurations use GPU FEM |
| Offline software suite in working checkout | 773 tests, 5 conditional skips; passed |
| Additional asset-installer / budget tests | 2 tests; passed |
| Fresh source + restored asset bundle, new environment | 775 tests, 7 conditional skips; passed |
| All-task dry-run from clean checkout | 32/32 exited successfully; zero model calls |
| Browser interaction checks | 16 task selections, 32 settings, 96 image loads; passed |
| Responsive layouts | 390, 768, 1024, 1440 px; no horizontal overflow |
| Browser errors / failed page requests | 0 / 0 |
| Slider and clipboard interactions | passed |
| Portable deployment wrappers | Generated and invoked with paths containing spaces; shell syntax passed |

The clean test environment lacks optional OpenCV, so its two video/image fixture
tests skip in addition to the five conditional skips in the working environment.
Those two fixtures were exercised by the working-checkout suite. Environment
versions are recorded in `cpu_test_environment.json` and `reference_runtimes.json`.

GPU FEM states in L2-04 and L2-06 (original and alternative) and rendered pixels
can vary numerically across initializations. Their configured initial placement,
asset geometry, materials, physics parameters and scoring code are unchanged.
The four FEM startup checks passed. Bit-identical rendered pixels or GPU FEM
trajectories are not claimed.

These checks establish package completeness, unchanged task definitions, usable
initial scenes and execution plumbing. They do not establish model success on
all tasks, repeat paper results or replace real-robot evaluation. No live model
was called during release validation.

Detailed local before/after images and machine-generated evidence are kept in
`exports/release_validation_20261021/`, outside the published source package.

## Sequential numbering validation

The renumbering was checked against all 256 effective component values captured
from the preceding commit (32 configurations × 8 components). After translating
identity labels and the independently verified placement-profile hash, no
behavioral differences remain. All 320 non-metadata asset files retain identical
SHA-256 values; five asset manifests/documentation files update identities only.
The 22 USD scene entry points still resolve. Historical visual-setup records and
raw experiment directories retain their original bytes and source identities.

The migration does not repeat live model experiments or claim new task successes.
