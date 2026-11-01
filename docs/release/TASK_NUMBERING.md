# Task numbering

The public catalogue uses L1-01–L1-05, L2-01–L2-07, and
L3-01–L3-04. Each has an independent `_variant01` setting; `_v3` remains a
suffix alias with the same **current** task number.

On 2026-10-24, whiteboard wiping moved from L3-03 to L2-07, including its
alternative setting. On 2026-10-26, stacked-book pushing moved from L3-04 to
L3-03 and block sweeping from L3-05 to L3-04 to match the paper. This numbering
change preserves each task's existing execution, budget and scoring profile.
The table below maps the original, pre-2026-10-21 delivery IDs.

| Original delivery ID | Current ID | Task |
| --- | --- | --- |
| L1-01 | L1-01 | Cylinder insertion |
| L1-03 | L1-02 | Plug alignment |
| L1-07 | L1-03 | Hat on a rack |
| L1-08 | L1-04 | Block stacking |
| L1-09 | L1-05 | Flower arrangement |
| L2-01 | L2-01 | Uncap a marker |
| L2-03 | L2-02 | Extract the middle book |
| L2-04 | L2-03 | Pour glass beads |
| L2-06 | L2-04 | Lift an orange segment |
| L2-07 | L2-05 | Open and close a drawer |
| L2-08 | L2-06 | Lift a jelly |
| L3-03 | L3-01 | Unscrew a bottle cap |
| L3-05 | L3-02 | Thread a nut |
| L3-11 | L2-07 | Wipe a whiteboard |
| L3-14 | L3-03 | Push stacked books |
| L3-15 | L3-04 | Bimanual sweeping |

Only task identities, configuration/media filenames, and references changed.
Instructions, scoring, action limits, robot poses, cameras, lighting, materials,
colliders, and initial placements are preserved. The website cover remains the
sweeping task, now L3-04.

Historical runs retain their recorded IDs and source revision. Some old numbers
now mean different tasks, so old numbers are **not** implicit CLI aliases. Use
`configs/benchmark/task_numbering.json` to interpret an old record, and use its
recorded source revision to reproduce or resume a pre-renumbering experiment.
An archived asset directory such as `assets/tuojing/source/L3-15/` retains its
original delivery name and bytes; the current manifest maps L3-04 to that scene.
The content-addressed visual-setup history is also immutable.

For records from the `level_reclassification_20261024` scheme, translate L3-04
to L3-03 and L3-05 to L3-04. For `sequential_20261021`, also translate L3-03 to
L2-07. In that scheme L3-03 means wiping; in the original delivery scheme it
means bottle-cap removal; in the current `paper_alignment_20261026` scheme it
means stacked-book pushing. Always select the mapping using the source revision.
