"""
Scores the planner's arrow against what a walker actually did on a recorded walk.

The walker's real path comes from the pose every recorded frame carries. The arrow is the planner's
own `lookahead_heading_radians`, read exactly as the server returned it. Nothing in this package builds,
gates or corrects an arrow, because the planner's math is what is being measured.

Every definition the evaluation classifies by, what a turn is, what counts as straight walking, what
counts as an obstacle ahead, is a constant in `EvaluationConfig`. None is read from the planner or the
scene, so tuning the planner moves the arrow's scores and nothing else.
"""
