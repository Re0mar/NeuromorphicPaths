# nav, the laptop pipeline

Written for teammates running or extending the pipeline. It takes a depth stream from any of
our sensors, turns it into a planned walking path, and hands that path to whatever display is
listening. The design, and why it is layered the way it is, is in `../../PIPELINE_DESIGN.md` at
the project root. This file is how to run it.

## Install

Python 3.12. Everything below was run on Windows from Git Bash, so paths use forward slashes.

```
cd server
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python -m pytest
```

That installs numpy, OpenCV, Open3D, aiohttp and pytest, and runs the whole test suite on CPU.
It deliberately does not install torch. The suite never needs it, and keeping it out is what
proves the scene and planner layers import without it.

To run the real depth estimator, which the glasses and any plain camera need:

```
.venv/Scripts/python -m pip install -e ".[dev,glasses]"
```

That adds torch, Depth Anything 3 and the Pupil Labs client. The torch that pip picks is the CPU
build, which is fine for a recording and too slow for a live walk. For live use install the CUDA
build that matches your driver from pytorch.org first, then run the line above. The first run
downloads the metric depth checkpoint, 1.3 GB.

If pip stalls on a 401 from a private package index, your machine has a user-level `pip.ini`
pointing somewhere that needs a token. Put this in `.venv/pip.ini` and it talks to PyPI only:

```
[global]
index-url = https://pypi.org/simple
```

## Running it

Every run names a source and a sink. Nothing is read from the environment, so the command line
is the whole configuration, and `--verbose` prints per-stage timings.

```
.venv/Scripts/python -m nav --source <source> --sink <sink> [flags]
```

### Sources

| Source | What it reads | Flags |
|---|---|---|
| `video_file` | a recording, or an IP camera app's stream URL, through the depth estimator | `--path`, checked before the model loads |
| `neon_live` | the Pupil Labs Neon over the network, through the depth estimator | `--neon-address` only if discovery is blocked |
| `arcore_tcp` | the Pixel app's depth frames over TCP | `--arcore-port` (9000), `--arcore-accept-timeout` (30), `--reconnect` |
| `neon_plugin` | a Neon recording the Neon Player depth plugin has run over | `--recording-dir`, `--plugin-model` |
| `logged` | a frame log this pipeline recorded earlier | `--log-dir`, `--realtime` |

`--reconnect` keeps the same listener open after the phone disconnects, so a walk recorded with
`--record-to` continues in the same log with the frame numbers running on. It waits up to the
accept timeout for the phone to come back, and then ends the run normally. Without the flag the
run ends when the phone disconnects. Launching the app by hand takes longer than the default
30 seconds, so a live run usually sets `--arcore-accept-timeout 600`.

### Sinks

| Sink | Where the path goes | Flags |
|---|---|---|
| `debug_window` | an OpenCV window with the arrow, the alarm, the surprise field and the depth view | |
| `web` | a page in any browser on the network: the arrow, the alarm, and the depth view under them | `--web-port` (8765) |
| `phone_app` | the Pixel app over TCP. The phone connects to the laptop, on this port | `--phone-port` (9100) |
| `none` | nowhere. For recording and for tests | |

The depth view is the depth image the planner saw, colored by distance, with each obstacle
group's nearest point as a ring sized by its clearance, magenta for a wall, the chosen path laid
on the floor as a white line, and one line of text: groups in view, the nearest clearance, where
the floor came from (`supplied` by the source, `fitted` from the cloud, or the `previous`
frame's), and `ALARM` when set. It is what a person tuning the planner looks at, and the window
and the browser draw it from the same code. The phone never gets it.

### The flags a phone needs

The floor gate refuses a plane, fitted or supplied by the source, that leans more than
`--floor-max-tilt` degrees from up (35 by default), that puts the camera under 0.3 m above it, or
that puts the camera more than `--floor-max-height` meters above it (2.2 by default). Up is
gravity, read from the pose, whenever the source places the camera in a world, which the Pixel
does. It has to be: the Pixel's depth image arrives in the sensor's landscape orientation
however the phone is held, so with the phone in portrait the image's own up points sideways, and
measured against it every floor leans 90 degrees. Sources with no position, a plain video, get the
image's up. The ceiling came from the first Pixel walk, where ARCore handed over a plane 2.3 m
down, a meter below the real floor, and nothing refused it.

A phone held in the hand and pointed at the pavement still leans about 40 degrees from gravity,
so the live runs set `--floor-max-tilt 50`. The metric model returns no camera intrinsics for a
plain video, so the pipeline assumes `--fallback-fov` degrees of horizontal field of view, 100
by default for the Neon. A phone is nearer 75, and the wrong value stretches the cloud sideways
and overstates the camera's height. That one was found on the first outdoor recording, where the
estimator read the camera 2.27 m above the floor at 100 degrees and 1.84 m at 75.

Phone videos also carry a rotation tag. The video source applies it, so a portrait recording
comes through upright.

### Commands that were run while building this

A phone recording through the estimator, writing every depth frame to a log:

```
.venv/Scripts/python -m nav --source video_file --path walk.mp4 --sink none --record-to frame_logs/walk --floor-max-tilt 50 --fallback-fov 75 --verbose
```

Replaying that log into the debug window, at the recorded frame rate:

```
.venv/Scripts/python -m nav --source logged --log-dir frame_logs/walk --sink debug_window --realtime
```

The TCP source, with the test sender standing in for the Pixel app from a second terminal:

```
.venv/Scripts/python -m nav --source arcore_tcp --arcore-port 9000 --sink none
.venv/Scripts/python tests/fake_arcore_sender.py --port 9000 --count 100
```

The Pixel over wifi, the phone as the display, recording the walk:

```
.venv/Scripts/python -m nav --source arcore_tcp --arcore-accept-timeout 600 --reconnect --sink phone_app --floor-max-tilt 50 --record-to frame_logs/walk --verbose
```

The same with the browser as the display, `--sink web`, and the page at `http://<laptop>:8765`.
Both ports the laptop listens on, 9000 for depth and 9100 for paths (8765 for the page), need an
inbound rule in the Windows firewall, and the rule has to name the Python that owns the socket.
With a venv that is the base interpreter the venv was made from, not the venv's `python.exe`,
which is a launcher. In an elevated PowerShell, once per port:

```
New-NetFirewallRule -DisplayName "nav pipeline, depth port" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 9000 -Program "C:\Users\<you>\AppData\Local\Programs\Python\Python312\python.exe" -Profile Any
New-NetFirewallRule -DisplayName "nav pipeline, path port" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 9100 -Program "C:\Users\<you>\AppData\Local\Programs\Python\Python312\python.exe" -Profile Any
```

Over USB instead, `adb reverse tcp:9000 tcp:9000` and `adb reverse tcp:9100 tcp:9100` on the
laptop, and the app connects to `127.0.0.1`.

The live glasses. Discovery finds the Neon on the local network. University wifi usually blocks
that between subnets, in which case read the address off the Companion app's streaming screen:

```
.venv/Scripts/python examples/check_neon.py
.venv/Scripts/python -m nav --source neon_live --sink web
.venv/Scripts/python -m nav --source neon_live --neon-address 10.0.0.5 --sink web
```

`check_neon.py` connects, receives one frame and exits 0, or says which path it tried and exits
1. The `neon_live` and `neon_plugin` commands above were not run while building this, because
there were no glasses and no plugin recording on hand. The Pixel runs on 2026-10-02 used the
`web` sink over wifi and USB. The `phone_app` sink has been run against the suite's fake phone,
through the same entry point, and not yet with the Pixel. Everything else was run as shown.

## Recording and replaying

`--record-to <dir>` wraps any source and writes every depth frame to that directory as it
passes, one file per frame plus an `index.jsonl`. The directory must be empty. Replay it with
`--source logged --log-dir <dir>`. The replay goes through the same scene and planner code the
live run did, so a walk recorded once can be tuned against as many times as needed.

A frame log holds depth frames, not the scene settings that were used on them. Replay with the
same `--floor-max-tilt` the recording run had, or the floor fit can refuse every frame the
recording accepted. The recording run writes its whole configuration to `run_config.json` in the
log directory, so the flags are there to read back.

Completed avoidances are written to `episodes.jsonl` in the same directory, one JSON line each.

## The layers

**Sources** are device specific and yield `DepthFrame` objects. The two RGB sources share one
depth estimator. **Scene** turns a depth image into obstacles on the ground: unproject, find the
floor, keep what is between ankle and head height, group into cells, and measure how much each
group's distance has been wobbling. That wobble is N and the distance is S. **Planner** builds a
surprise field over future time and lateral position from S and N, and runs dynamic programming
through it. **User model** watches the planner's output and the walker's heading and measures
what each avoidance cost in bits. It never steers. **Sinks** are device specific and take a
`PlannedPath`. The runtime runs scene and planner on the newest frame in a thread so the source
is never blocked.

Each layer owns its configuration in its own `config.py`. The footprint radius lives in
`nav/walker.py` and nowhere else.

## The ARCore contract

`docs/arcore_wire_format.md` is the contract the Pixel app implements: framing, every header
field, the depth encoding, which way is up in the world the pose describes, the path message
that comes back with its own per-field table, a worked example, and what each side does on each
kind of malformed message. The test sender in `tests/fake_arcore_sender.py` sends frames in that
format with the same encoder the laptop decodes with. Two committed fixtures check the contract
across the language boundary: `tests/fixtures/pixel_app_frame.bin`, written by the app's encoder
and decoded here, and `tests/fixtures/laptop_path.bin`, written by `encode_path` from stated
values in `tests/test_laptop_path_fixture.py` and decoded by the app's test.

## Known limits

- ARCore depth is computed from motion. It stops updating when the walker stands still, drops
  out on blank walls, and is unreliable on moving objects. Static obstacles first.
- Estimator depth flickers between frames and its scale can drift. The pose never comes from it.
- Without a position there is no memory of obstacles out of view, and N is approximate because
  the ground grid moves with the walker.
- Motion prediction in the planner is off by default. The scene reports a group's velocity from
  its centroid in the world frame, which needs a source with a position.
- The user model's time constant, b, is a placeholder until a walker is measured.
