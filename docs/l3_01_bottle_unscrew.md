# L3-01 bottle closure contact revision

`l301-passive-closure-v1` applies to both `L3-01` and `L3-01_variant01`
(including the compatibility alias `L3-01_v3`). It changes the closure physics
and initial thread engagement. Earlier recordings retain their original setup.

The supplied scene starts with the cap already lifted 7.5 mm and rotated 300°
along a three-start thread with a 9 mm lead. The remaining engagement is small.
An independent force test of that setup reproduced release under a direct
0.15 N pull, followed by deep penetration into the neck. Raising the physics
rate alone did not resolve it. GPU dynamics also disables the original swept
CCD setting in the tested Isaac Sim 5.1 runtime.

The runtime now makes the following session-layer changes:

- Move the cap 3 mm back along the screw path and rotate it back 120°. Its
  initial lift is 4.5 mm and rotation is 180°. The bottle and assembly placement
  retain the original or variant positions.
- Replace the large neck/cap SDF colliders with hollow convex shell sectors and
  short convex segments derived from the supplied helical meshes. Split the
  neck profile at its elbow so convexification does not fill the cap clearance.
- Use 0.15 mm contact offsets, speculative CCD and 64/8 solver iterations for
  the closure. The physics step is 1/1920 s; control and camera cadence remain
  12 Hz. The source bulk collider and seal contact are retained.

The visible meshes, masses, friction and fixed bottle remain as supplied.
There is no attachment joint, task-specific force, prescribed screw motion or
runtime pose correction during manipulation. Reset, scene reporting and
geometric scoring use the same adjusted initial cap pose. The English prompt,
arm roles, success thresholds, camera calibration and image resolutions are
unchanged. `scene.json` records the collision revision and initial adjustment.

The regression commands are documented in the
[engineering diagnostics](../src/robot_tasks/integrationtest/diagnostics/README.md).
They check retained contact under specified loads, opening under torque and a
drop onto the neck, together with scene/camera startup. Opening torque has not
been calibrated against the real bottle; these checks are not evidence of GPT
task success or a guarantee against penetration under arbitrary loads.
