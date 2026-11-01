# Measured L2 asset calibration

The 2026-10-23 calibration uses the confirmed real-prop targets below. It applies
to new original and `variant01` episodes; historical runs retain their recorded
asset settings and outcomes.

| Asset | Previous simulation value | Confirmed target |
| --- | ---: | ---: |
| L2-01 capped marker | 146.50 mm visible axial length | 150 mm |
| L2-06 jelly | 37.27 g from mesh volume and density | 26.4 g net mass |

The length target is a ruler-based nominal measurement, not a claim of sub-mm
metrology. The jelly mass was confirmed as the net target after checking the
weighing photograph.

`tasks/bench/measured_assets.py` authors the calibration on the USD session layer.
For the marker, all axial render and collision geometry, including the hollow
cap, share the same affine stretch about the original midpoint. The passive
shoe positions, spring anchors, centres of mass, inertia and evaluator geometry
are transformed consistently. The internal spherical pads keep their original
0.5 mm radius while moving with the calibrated cap. Diameters, masses, radial stiffness, friction,
task prompts, action space and success tolerances are preserved. The L1 placement
task's distinct marker assembly is outside this L2 calibration.

The nib capsule keeps spherical ends and the original radius; its cylinder
height carries the calibrated axial extent. This avoids PhysX's unsupported
nonuniform capsule scaling.

The jelly receives an explicit `UsdPhysics.MassAPI` total mass on its deformable
mesh, overriding the source material's density-derived total. Geometry,
elasticity, friction and optical material remain unchanged. The task metadata
uses the same 0.0264 kg value.

## Verification

```bash
python -m unittest discover -s tests -p test_measured_assets.py
python tools/release/render_initial.py L2-01 --gpu 0 --reset-only --output build/check-marker
python tools/release/render_initial.py L2-01_variant01 --gpu 0 --reset-only --output build/check-marker-variant
python tools/release/render_initial.py L2-06 --gpu 0 --reset-only --output build/check-jelly
python tools/release/render_initial.py L2-06_variant01 --gpu 0 --reset-only --output build/check-jelly-variant
```

The render commands require the Isaac environment. They perform physical reset,
camera capture and initial-state checks without a model request; they do not
establish task-solving success.

## Documentation views

```bash
python tools/diagnostics/render_measured_assets.py drawer --output build/assets/drawer
python tools/diagnostics/render_measured_assets.py marker --output build/assets/marker
python tools/diagnostics/render_measured_assets.py jelly --output build/assets/jelly
```

These commands render the actual meshes and materials using separate studio
lights and a documentation camera. The drawer views show its original partly
open upper drawer and an additional 50 mm opening along the authored local rail.
No task camera, initial layout, source USD or experiment record is saved over.
