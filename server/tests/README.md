# Tests

How the laptop pipeline's suite is run, where a new test goes, what kinds of run exist, and which
of them is switched on. The last question is the one this file exists for: nothing runs these
tests but a person at a keyboard.

## Running tests locally

From `server/`, with the venv `../server/README.md` describes:

```
.venv/Scripts/python -m pytest
```

Dependencies come from `pyproject.toml`'s `dev` extra. The suite needs no GPU, no torch, no
glasses, no phone, and no display. It takes about two and a half minutes on this laptop. About a
minute of that is `test_evaluation_replay.py`, which writes synthetic recordings through the real
recording tap and replays them through the real scene, and about 30 seconds is the golden test
below. `pytest -rs` is already in
`pyproject.toml`, so a skipped test prints its reason in the summary. With only the `dev` extra,
the Neon tests that need PyAV or the Pupil Labs client skip, because those come with the
`glasses` extra. With both extras installed, nothing skips: 1326 passed on 2026-10-06. With only
the `dev` extra, 1311 passed and 15 skipped the same day.

## Adding a test

One file per module under test, `tests/test_<module>.py`, importing from `nav` the way anything
else does. There is no `sys.path` trick and no package marker in `tests/`, so a test file is a
script that `pytest` collects, and the helpers beside them (`stubs.py`, `synthetic_depth.py`,
`fake_arcore_sender.py` for standing in for the phone on the depth port, synthetic frames or a
recorded walk, `fake_path_reader.py` for standing in for it on the path port, `neon_captures.py`
for writing Neon captures with real H.264, and `fake_neon_client.py` for standing in for the
Pupil Labs client) are imported by bare name. The two phone stand-ins also run as scripts, which
is how a recorded walk is measured without the phone.

A new test is obliged to assert:

- **The refusal and what it says.** A negative test names the exception type and matches a piece
  of the message, usually the field or flag the message names. `pytest.raises(ValueError)` alone
  lets a rename turn a precise error into a vague one with the suite still green.
- **The fact, not the implementation.** Expected values come from the requirement or from
  arithmetic done in the test, never pasted from what the code printed.
- **Against real neighbours.** No mocks. A fake is a small class the test writes, with real
  behavior (`stubs.StubDepthEstimator` computes a floor from the pinhole model; `test_arcore_tcp.py`
  sends over a real socket; `test_web_sink.py` connects a real websocket client).
- **Nothing from the device packages.** torch, Depth Anything 3, the Pupil Labs client and
  aiohttp are kept out of the test environment or confined to one file each, and
  `test_layering.py` and `test_import_boundaries.py` fail when that changes.

Every test here was made to fail at least once before it was accepted, by breaking the thing it
guards and watching it go red. Do the same for a new one.

## Test run types

| Run type | What belongs in it | Trigger | Budget | Status |
|---|---|---|---|---|
| Gate | line endings, the import boundaries, the kind enums staying in `config.py`, the no-environment rule, the wire format document matching the encoder | every pull request | seconds | ❌ not in place. These tests exist and run locally; no pipeline runs them |
| Verify | the rest of the suite: codec, scene, planner, user model, sinks, runtime, end to end on a synthetic video, and the planner's golden numbers on two committed slices of recorded walks | on request, or when a pull request leaves draft | about two and a half minutes | ❌ not in place. Run by hand before every commit |
| Perf | `test_the_default_grid_plans_in_under_ten_milliseconds` and `test_the_default_grid_plans_and_measures_information_in_under_ten_milliseconds` are the two timing assertions, and they live in Verify because they take milliseconds. The depth view's drawing time is measured by hand, not asserted | | | ❌ not in place, and not expected: the pipeline's budget is a frame rate on one laptop, measured by `--verbose` on a real recording |
| Stress | | | | ❌ not in place, and not expected: one sender, one browser, one phone |
| Nightly | a run of the real estimator on the committed outdoor recording, checking the floor height it reports against the measured 1.84 m | | | ❌ not in place. This is the gap a pull-request check cannot see: the model, the weights and the recording are all outside the repository |
| Weekly | | | | ❌ not in place, and not expected |

## Deciding which run type a new test belongs in

Can it answer from this repository alone, with no model weights, no device and no display? Then
it is Verify, and Gate if it also needs no OpenCV or Open3D call.

The web page's drawing is the one thing here no test reaches, because the repository has no
JavaScript runner. `test_web_sink.py` checks that the served page reads every key the laptop sends.
Whether it draws them correctly is checked by eye, on a replay served with `--sink web`. Everything else is Nightly, and
until Nightly exists it is a command in `server/README.md` run by hand with the result written
down.

## Updating the golden values

`test_planner_golden.py` plans two committed slices of recorded walks and checks the planner's
whole-walk numbers on each: how often the arrow sits at its sidestep limit, overall, with something
3 to 5.32 m ahead (as plainly counted and as restated) and with a clear corridor, how often it
swings from one limit to the other, how many headings it takes, how often the alarm is on and how
often it switches, how many raise decisions came with nothing within 1 m, how long the alarm stayed
up after its last raise, the noise of nearby points, and how far consecutive plans disagree. The
slices are under
`fixtures/`, one from `pixel_walk_3` and one from the classroom walk `pixel_display_run`, about
1 MB together. It takes about 30 seconds.

These are the one place in the suite where the expected values were captured from the code rather
than worked out in the test. That is what a golden test is: a record of what the planner does,
so a change to it can't go unnoticed. The arithmetic check beside it is
`test_a_hand_built_slice_gives_the_numbers_worked_out_by_hand` in `test_evaluation_fixture.py`.

**When the values move.** A planner change moves them, and that is the point. Update the values
in `EXPECTED` in the same commit as the change, and say in the commit message which figures moved,
by how much, and why that is the intended effect. A commit that moves them with no planner change
in it has found a defect. Tolerances are 0.5 percentage points on a share, 1 on a count,
0.005 m on a disagreement, 0.003 s on the alarm's hold and 0.0005 m on the noise. Below 200 frames one frame is worth more than 0.5 points, so a share
over fewer frames has to match to the frame. Another machine or numpy install can flip a plan that
is a near tie, and a share moving by one frame there is real information, not noise to widen the
tolerance for.

**When a slice has to be re-cut.** When the scene changes what it hands the planner, or the walker
changes size, the slices no longer show what today's scene would see. The test refuses a slice
cut with another walker by name. Re-cut from the recordings with `python -m nav.evaluation.fixture`.
The exact commands are in `test_planner_golden.py`'s docstring, and `find` proposes a window when
the old one no longer qualifies. Rerun the test, update `EXPECTED`, and commit the slices and the
values together.

The cutter refuses a slice whose 0.1 mm rounding would change any plan, an `--out` not named
`golden_*.json.gz`, and a set of slices over 1 MB. A refused re-cut leaves the slice it would have
replaced where it was.

## What CI actually runs

Nothing. There is no workflow file in this repository. No check is required for merge, so a pull
request with a red suite can be merged by anyone who did not run it. Whether to add a pipeline is
the team's decision, and until it is made the only gate is the person committing.
