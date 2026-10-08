# Scoring the arrow against real turns

quick how-to. results + what they mean: [../evaluation/arrow_against_turns.md](../evaluation/arrow_against_turns.md)

## what it answers

did the arrow point where the walker actually went? takes a recorded walk, replays every frame through
the real scene + planner, finds the walker's real turns from the phone's tracked position, and checks
the arrow before each one. also counts sidesteps while walking straight, and a lagged correlation.

## setup

- the `server/` venv, made the way `server/README.md` says. nothing extra to install
- recordings live in `server/frame_logs/`, **not in git** (gitignored, ~60 KB a frame). get them off
  whoever recorded, or make your own:
  - live: run the laptop with `--record-to frame_logs/<name>` (server/README, "Recording and replaying")
  - no wifi where you walk: record on the phone, replay to the laptop later (pixel_app/README,
    "Recording a walk without the laptop")

## run it

from the repo root. Git Bash:

```bash
cd server
source .venv/Scripts/activate
python -m nav.evaluation frame_logs/wifi_run_2 --scene-set floor_ransac_seed=0 --scene-set floor_ransac_success_probability=0.99999999 --out results_wifi_run_2.md
```

PowerShell, same thing:

```powershell
cd server
.venv\Scripts\Activate.ps1
python -m nav.evaluation frame_logs/wifi_run_2 --scene-set floor_ransac_seed=0 --scene-set floor_ransac_success_probability=0.99999999 --out results_wifi_run_2.md
```

one walk per command, because each needs its own scene flags:

| walk | flags | why |
|---|---|---|
| `wifi_run_2` | `--scene-set floor_ransac_seed=0 --scene-set floor_ransac_success_probability=0.99999999` | recorded before those two settings existed. these are the defaults |
| `pixel_walk_3` | the two above + `--scene-set floor_max_offset_meters=inf` | also predates the camera-height check, its live run had none |
| `pixel_display_run` | `--scene-defaults --scene-set floor_max_tilt_degrees=50` | no `run_config.json` at all. 50 is the documented live tilt |
| anything recorded from now on | none | its `run_config.json` has every setting |

the command reads the scene settings from the recording's own `run_config.json`, so it sees the floor
the live run saw. if a setting is missing it refuses rather than quietly using today's default, which
is why the old walks need flags.

another old recording refused for a setting not in the table? look the setting up in
`server/nav/scene/config.py`. if it only makes replays repeatable (the seed) or is a library default,
pass its default. if it's a check the live run didn't have, pass the value that turns the check off,
like `inf` for a maximum. if you can't tell, ask before quoting numbers from it.

## what you get

takes 3 to 7 min a walk (three replays of the scene, the slow part). ends with a table like:

```
- Turns: 6 at a 11 deg threshold (left -72, right +15, right +46, left -31, right +37, left -98)

| Tag | Turns | Arrow read | Agreed | Wrong side | Carry on | Not read | Median lead s (known, at limit) |
|---|---|---|---|---|---|---|---|
| obstacle ahead | 4 | 4 | 4 | 0 | 0 | 0 | 0.40 (3, 1) |
| open ahead | 0 | 0 | 0 | 0 | 0 | 0 | unknown (0, 0) |
| unknown | 2 | 2 | 1 | 1 | 0 | 0 | 0.06 (1, 1) |
```

plus sidesteps, the correlation per segment, how far the phone pointed off the walking direction,
and why the scene refused any frames it couldn't plan. "at limit" means the lead is at least that
long: it hit the 5 s cap, the previous turn, or a stretch where the walker's heading isn't known.
top of the report has the commit + a hash of the code that made it. commit before a run you'll quote.

`--out file.md` writes the same report to a file. otherwise it's only on screen.

## exit codes

- `0` scored
- `1` refused: missing `run_config.json` (add `--scene-defaults`), a setting missing from it
  (add the `--scene-set` it names), a bad flag, a corrupt recording. the message says which
- `2` usage error from the flags themselves
- `3` nothing to score: no segment of 20 s or more, no turns found, or no turn had an arrow before it

## after changing the planner

just rerun. turns + tags don't depend on the planner, only the arrow numbers move. try a planner
setting without editing code: `--set lateral_kinetic_weight=0.2`. iterating? `--cached` does one replay
through a cache in `server/.replay_cache/` (same place whatever folder you run from), much faster.
not for numbers you'll quote, use the default 3. to compare against the planner before the contact
term: `--set lateral_kinetic_weight=0.055 --set contact_term_enabled=false`.

## the turn threshold

11 deg, measured from a straight walk. rerun the measurement:

```bash
python -m nav.evaluation --spread-only frame_logs/straight_walk_4
```

reads only the poses, takes seconds. exits `3` with the reason if the walk can't set a threshold
(too little straight walking, or the result set by the straightness cut-off). to measure from a new
walk: one long straight line, phone held like on the real walks, 60 s+ of it.
