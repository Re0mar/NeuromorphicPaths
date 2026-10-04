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
glasses, no phone, and no display. It takes about a minute on this laptop, about half of it in
`test_evaluation_replay.py`, which writes synthetic recordings through the real recording tap
and replays them through the real scene. `pytest -rs` is already in
`pyproject.toml`, so a skipped test prints its reason in the summary. Today nothing skips.

## Adding a test

One file per module under test, `tests/test_<module>.py`, importing from `nav` the way anything
else does. There is no `sys.path` trick and no package marker in `tests/`, so a test file is a
script that `pytest` collects, and the helpers beside them (`stubs.py`, `synthetic_depth.py`,
`fake_arcore_sender.py`) are imported by bare name.

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
| Verify | the rest of the suite: codec, scene, planner, user model, sinks, runtime, end to end on a synthetic video | on request, or when a pull request leaves draft | under a minute | ❌ not in place. Run by hand before every commit |
| Perf | `test_the_default_grid_plans_in_under_ten_milliseconds` is the one timing assertion, and it lives in Verify because it takes milliseconds | | | ❌ not in place, and not expected: the pipeline's budget is a frame rate on one laptop, measured by `--verbose` on a real recording |
| Stress | | | | ❌ not in place, and not expected: one sender, one browser, one phone |
| Nightly | a run of the real estimator on the committed outdoor recording, checking the floor height it reports against the measured 1.84 m | | | ❌ not in place. This is the gap a pull-request check cannot see: the model, the weights and the recording are all outside the repository |
| Weekly | | | | ❌ not in place, and not expected |

## Deciding which run type a new test belongs in

Can it answer from this repository alone, with no model weights, no device and no display? Then
it is Verify, and Gate if it also needs no OpenCV or Open3D call. Everything else is Nightly, and
until Nightly exists it is a command in `server/README.md` run by hand with the result written
down.

## What CI actually runs

Nothing. There is no workflow file in this repository. No check is required for merge, so a pull
request with a red suite can be merged by anyone who did not run it. Whether to add a pipeline is
the team's decision, and until it is made the only gate is the person committing.
