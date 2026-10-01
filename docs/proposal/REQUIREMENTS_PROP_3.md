# Requirements

Project restart, 2026-10-01. Built from the plan emailed to the professor on 2026-09-30 and
his four replies that day and the next. Sensor and build-order decisions are argued out in
`SENSOR_ROUTE_DECISION.md`. His code and emails are in `vertegaal_demo/` and stay confidential.

Each requirement has an id. R is a hard requirement, S is a stretch goal, T is a test that
gates a later requirement. "The planner" means the Hamiltonian planner from his
`hamiltonian_planner.py`, reimplemented in our own code and never pasted from his.

## Start from scratch

- R0. This is a clean start. Nothing from `NeuromorphicPaths/` or `NHCI_Project_Restart/` is
  read, referenced or ported until milestone M1 below is met. Not the Kotlin app, not the
  replay tooling, not the surprise math, not the display. The new code is written against
  this file and his planner only.
- M1. The Pixel pipeline runs live end to end: ARCore depth and pose into the laptop, points,
  surprise, the planner, an arrow on the Pixel, timed per stage on real data (T4). A person
  has walked a room with it.
- After M1, the previous projects are opened and judged feature by feature for porting. A
  feature is ported only if it serves a requirement in this file. Candidates are the ones the
  old work got right that this file also needs, such as replay of recordings, the display
  tunables and the step-cadence speed estimate. Nothing is ported because it exists.

## Goal

- R1. Propose a walking path to a user from a live camera, shown as an arrow, followed blind.
- R2. No classification anywhere. No object detector, no labels. Every point in the cloud
  speaks for itself.
- R3. The same planner code runs on every sensor. The sensor only changes where the depth map
  comes from.

## Hardware

- R4. Pixel 8 running ARCore is the first sensor. It streams a depth image and pose per frame to
  the laptop over TCP on a shared hotspot.
- R5. Pupil Labs Neon glasses are the second sensor. Scene video and IMU come through the
  Pupil Labs real-time API live, or from a recording offline.
- R6. The laptop runs everything after the sensor. Python, numpy, Open3D.
- R7. The arrow is drawn by the Pixel app when the Pixel is the sensor, and by a web page on
  any phone's browser when the glasses are the sensor.
- R8. No iPhone and no Mac are assumed. If a LiDAR iPhone turns up, it becomes a third depth
  source behind the same interface, via Record3D or an ARKit app.

## Input

- R9. Per frame the pipeline takes a depth image, camera intrinsics, a pose and a ground plane.
  Nothing else. This is the depth frame source interface.
- R10. On the Pixel, ARCore supplies all four.
- R11. On the glasses, a pretrained monocular depth estimator supplies the depth image. Offline
  that is Pupil Labs' Neon Player depth plugin, which runs Depth Anything 3. Live it is the same
  model called directly. No training by us.
- R12. On the glasses, pose does not come from the depth estimator. Orientation comes from the
  Neon IMU quaternion. Position comes from a monocular-inertial SLAM or is absent.
- R13. The pose record carries a `has_position` flag. When false, the cloud lives in the body
  frame and is rebuilt each frame. When true, points are placed in a world frame.
- R14. Depth pixels are unprojected into 3D points. Using the ground plane, points below ankle
  height are dropped as floor and points above head height are dropped as ceiling or overhang.
  Everything in the band between is an obstacle point. The two heights are parameters.
- R14a. Surviving points are flattened to the planner's 2D ground plane by dropping height.
  The planner never sees a height. Stairs, curbs and holes are therefore invisible to it,
  which is the flat-ground limit below.
- R15. Points are grouped into clouds by a spatial rule, a voxel or ground grid cell, never by
  what they are. The grouping rule is a parameter.
- R16. Each point keeps a distance history over the noise window, either through Open3D's
  fused volume for static points or through cluster tracking for moving ones.
- R17. Live and recorded input go through the same depth frame source interface. Recorded
  replay is the primary development and tuning mode.

## Surprise

- R18. Each point gets a collision surprise, half of (N over S) squared, with S the clearance
  from the walker's footprint to the point and N the standard deviation of that point's
  distance over the noise window.
- R19. The noise window is a parameter. It starts at his half second and is tuned on
  recordings. There is no fixed correct value.
- R20. Each cloud is represented by its most surprising point. Cloud surprises are summed.
- R21. Wall and edge points get a stiffer N through a multiplier, as the road edges do in his
  code. The multiplier is a parameter.
- R22. The clearance has a floor and the surprise has a cap, as in his code, so the planner
  never divides by zero.

## Planning

- R23. The planner plans lateral offset against time down the walking corridor, on a lateral
  grid over a horizon, with forward progress at the walker's speed. The grid, horizon and
  speed are parameters set for walking, not his 5 m/s.
- R24. The cost is H equals T plus U. U is the summed collision surprise at that time and
  lateral state. T is the lateral kinetic cost of moving between lateral states, weighted by a
  parameter.
- R25. Dynamic programming over (time, lateral state) with a reachability limit per step
  returns the minimum accumulated cost path s*(t).
- R26. Static mode first, with obstacles held where they are. Motion prediction, sliding each
  cloud ahead at its tracked velocity, comes after cluster tracking is shown to hold.
- R27. The planner runs in the walker's body frame and re-plans every frame, so the corridor
  follows the walker around corners.
- R28. Only the first step of s* is shown. The rest exists to make the first step right.

## Turning

- R29. The Lagrangian is a user model, not a controller. The walker's heading error is modeled
  as relaxing with the critically damped spring using their own b.
- R30. b is measured once per walker with a quick pointing test, in seconds per bit.
- R31. The model is used to predict turn time and to measure work per avoidance. It never
  drives the user. His vehicle execution code is not used.

## Warning

- R32. Capacity ends at signal-to-noise one to one, from the car-following study. That is one
  second to contact.
- R33. Below that, the display goes red and stops guiding.

## Display

- R34. The display shows the arrow for the first step of s*, a red state for the warning, and
  nothing else. No video.
- R35. The path reaches the display over the network as a few numbers per frame, never as
  video.
- R36. Timestamps are logged at every hop so glasses-to-arrow latency is a measured number.

## Evaluation

- R37. One metric, work in bits per avoidance, plus leftover surprise.
- R38. Same route, same obstacles, with and without the arrow.

## Tests that gate the glasses

- T1. Run the Pupil Labs depth plugin on one hallway recording and inspect the depth maps.
  Gates R11.
- T2. Feed those depth maps to Open3D dense SLAM and record whether its pose holds or wanders.
  Decides whether Open3D may supply position or only the point cloud. Gates R12.
- T3. Run a monocular-inertial SLAM on the same recording as the fallback pose source. Gates
  R12.
- T4. Time the Pixel pipeline per stage on real data before quoting any frame rate.

## Build order

1. Pixel, live. R4, R10, R7 Pixel side, planner, arrow. Ends at M1.
2. Porting review of the previous projects, per R0. Only after M1.
3. Glasses, offline. T1 through T3, then R11 and R12 on recordings. Tune R19 here.
4. Glasses, live. S1 below.

## Stretch

- S1. Glasses live, calling Depth Anything 3 on the stream. Needs an NVIDIA GPU on the laptop.
- S2. Motion prediction in the planner, R26 second half.
- S3. Open3D fused volume as memory of obstacles out of view, if T2 shows its tracking holds.

## Known limits, stated up front

- ARCore depth is computed from motion. It stops updating when the walker stands still, drops
  out on blank walls, and is unreliable on moving objects. Static obstacles first.
- Estimator depth flickers between frames and its scale can drift. That is why R12 keeps it
  away from pose.
- Without position there is no memory of obstacles out of view.
- Open3D takes a depth image per frame as input and has no monocular mode. It cannot work from
  the glasses' 2D video alone. Documented in its reconstruction system docs.

## Removed from the plan

- Spiking neurons, anywhere. He called them unnecessary.
- Optical flow, looming and feet-on-the-ground ranging. The planner computes surprise from
  point distances and their history, so the flow stage has nothing left to do.
- The fan of candidate headings. His lateral grid replaces it.
- A one-second noise window as a fixed value. It is a parameter.
