# L3-03 initial-state translations

The original task retains its delivered initial state. The default
`L3-03_variant01` (`L3-03_v3`) restores the original aligned two-book stack and
translates both books, the starting rectangle, and the target rectangle by
`-0.10 m` along world X, placing the stack in the middle of the fixed overview.

The optional task setting `stacked_book_offset_x_m` accepts a finite offset up
to 30 cm in either direction; any newly chosen layout needs its own tabletop and
reachability checks. It is applied once, before physics initialization,
on the USD session layer. Source USD files and historical runs remain intact.
The rendered rectangle transforms, reset poses, and scoring polygons receive
the same translation. Heights, dimensions, materials, cameras, robot initial
states, and relative book/rectangle orientations are preserved from the original.

The translation changes placement only. As of 2026-10-23, the original and
variant use the same prompt and require both books completely inside the
starting and target rectangles. Both require a full 360-degree stack rotation
during tabletop transfer, with the existing support and no-grasp/no-lift
constraints. An upper book overhanging the target cannot complete the task.

Earlier variant runs used a lower-book-only boundary rule. Their saved
configurations and outcomes retain that rule; this alignment applies to new
runs. The existing per-configuration call and time budgets are unchanged.

In an installed Isaac environment, render the centered variant without a
model request:

```bash
python tools/release/render_initial.py L3-03_variant01 --gpu 0 --reset-only \
  --output build/stacked_book_center_preview
```

Startup, table-clearance, and joint-limited reach checks establish initial
placement feasibility, not successful execution of the complete manipulation.

Additional placements can be reviewed with `--config` and an ignored
`*.local.toml` override; they do not create or change a registered task.
