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

That installs numpy, SciPy, OpenCV, Open3D, aiohttp and pytest, and runs the whole test suite on CPU.
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

### The CUDA build, as installed on the Quadro T2000 laptop

On 2026-10-03, with NVIDIA driver 581.95 (which reports CUDA 13.0), the line that worked was:

```
.venv/Scripts/python -m pip install torch==2.14.1+cu126 --index-url https://download.pytorch.org/whl/cu126
.venv/Scripts/python -m pip install -e ".[dev,glasses]"
```

The cu126 build was picked over cu130 because its kernels certainly cover the T2000's Turing GPU
(sm_75). Installing the glasses extra afterwards kept the CUDA build in place. Check it did, every
time, because a package that depends on torch can pull the CPU build back in without saying so:

```
.venv/Scripts/python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The version must end in `+cu126` and the third value must be `True`. A `neon_live` run on a CPU
estimator also says so in its log, and still runs.

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
| `web` | a page in any browser on the network: the arrow, the alarm, the planner's view from above, and the depth view | `--web-port` (8765) |
| `phone_app` | the Pixel app over TCP. The phone connects to the laptop, on this port | `--phone-port` (9100) |
| `none` | nowhere. For recording and for tests | |

**`--sink` can be repeated, and a walk usually repeats it.** The arrow belongs on the phone, where
the walker is looking, and the depth view belongs in a browser, where whoever is watching the
laptop is looking. Name both and both are served from the one run:

```
--sink phone_app --sink web
```

Any combination works, all three included. Each display is named at most once, since two of the
same means two servers on one port. One display named is exactly what it always was.

The depth view is the depth image the planner saw, in gray with near bright and far dark, and a dim
brown where there is no reading. It is turned a quarter turn at a time so the floor is at the bottom,
because the Pixel sends its depth image sideways to how the phone is held. On it are each obstacle group's nearest point as a ring sized by
its clearance, amber for a group and magenta for a wall, the chosen path laid on the floor as a
ribbon the body's width, and one line of text. The ribbon is colored and filled the way the band
in the view from above is. Its fill also fades to nothing toward the end of the plan, and its
borders do not, so its direction stays visible. The line of text gives the groups in view, the
nearest clearance, where the floor came from (`supplied` by the source, `fitted` from the cloud,
or the `previous` frame's), and `ALARM` when set. It is what a person tuning the planner looks at, and the window
and the browser draw it from the same code. The phone never gets it.

The browser also draws the planner's view from above, walking up the screen, about 5.3 m ahead and
3 m to each side. Three layers, each with a checkbox that hides it in the browser alone:

- **Field.** What every spot ahead costs to walk through, at the moment the walker would reach it.
  Brighter costs more. It is clipped at the frame's 98th percentile, so one costly point does not
  leave the rest dark.
- **Path.** The planned path as a band the body's width, from a dot at the walker to an arrowhead
  where the plan ends. Its color runs blue to red as something in the way gets closer, red from
  one second to contact. Its fill is more solid the more the scene shaped the plan, and its borders
  stay at one opacity so its direction always shows.
- **Obstacles.** Each group's nearest point, amber dots for groups and magenta squares for walls,
  the same colors the depth view uses.

Under it are two numbers. *How much the scene shaped the plan* is how far what the camera saw moved
the plan from what the planner would do with nothing in view, in bits, measured where the arrow
reads the path. It is not a confidence: an empty corridor gives a plan the planner is sure of and
0 bits. *How soon something is in the way* is the course's avoidance surprise for the nearest group
in the walker's path, in bits, 0.72 at one second to contact. The laptop computes every color,
opacity and number on the page. The page only draws them. To see it on a recorded walk:

```
.venv/Scripts/python -m nav --source logged --log-dir frame_logs/walk --sink web --realtime
```

A display that cannot start ends the run with its own message, because a display asked for and
silently missing is worse than a run that says why it stopped. A display that fails once it is
running is dropped with its traceback and the rest keep going, because by then a walker is
looking at one of them.

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
plain video, so `video_file` assumes `--fallback-fov` degrees of horizontal field of view, 100 by
default. A phone is nearer 75. The metric model answers as if every camera had a 300 pixel focal
length, and the depth source converts its output to meters with the focal it was given. So the
assumed field of view sets how far away things straight ahead come out: too wide, and they read
short. The camera's height above the floor does not depend on it, because the focal cancels out of
up and sideways. Before that conversion was added, the first outdoor recording read the camera
2.27 m above the floor at 100 degrees and 1.84 m at 75, and those two figures no longer apply. `neon_live` ignores the flag. It
reads the glasses' own calibration when it connects, straightens every frame and the gaze point
with it, and hands the straightened camera matrix on, so there is nothing to guess.

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

The Pixel over wifi, the arrow on the phone and the depth view in a browser, recording the walk:

```
.venv/Scripts/python -m nav --source arcore_tcp --arcore-accept-timeout 600 --reconnect --sink phone_app --sink web --floor-max-tilt 50 --record-to frame_logs/walk --verbose
```

The page is at `http://<laptop>:8765` and serves from the moment the run starts, before the phone
has connected. All three ports the laptop listens on, 9000 for depth, 9100 for paths and 8765 for
the page, need an
inbound rule in the Windows firewall, and the rule has to name the Python that owns the socket.
With a venv that is the base interpreter the venv was made from, not the venv's `python.exe`,
which is a launcher. In an elevated PowerShell, once per port:

```
New-NetFirewallRule -DisplayName "nav pipeline, depth port" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 9000 -Program "C:\Users\<you>\AppData\Local\Programs\Python\Python312\python.exe" -Profile Any
New-NetFirewallRule -DisplayName "nav pipeline, path port" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 9100 -Program "C:\Users\<you>\AppData\Local\Programs\Python\Python312\python.exe" -Profile Any
```

Over USB instead, `adb reverse tcp:9000 tcp:9000` and `adb reverse tcp:9100 tcp:9100` on the
laptop, and the app connects to `127.0.0.1`.

### Reserve the three ports on Windows

A run that ends with `could not listen on 0.0.0.0:9100` is a port another process holds, and the
other process is usually not a server. Windows hands out a local port for every outbound
connection from its dynamic range, and this laptop's range is the whole of 1024 to 65534, so a
browser, a Flutter tool or any other program can be given 9000, 9100 or 8765 and keep it for as
long as its connection lasts. Check with `netsh int ipv4 show dynamicport tcp`.

Reserve the three so Windows stops handing them out. Once, in an elevated PowerShell:

```powershell
netsh int ipv4 add excludedportrange protocol=tcp startport=9000 numberofports=1 store=persistent
netsh int ipv4 add excludedportrange protocol=tcp startport=9100 numberofports=1 store=persistent
netsh int ipv4 add excludedportrange protocol=tcp startport=8765 numberofports=1 store=persistent
```

`netsh int ipv4 show excludedportrange protocol=tcp` lists them afterwards. A port already held
by a running process has to be given up first, and `Get-NetTCPConnection -LocalPort 9100` names
the process holding it.

The live glasses. Discovery finds the Neon on the local network. University wifi usually blocks
that between subnets, in which case read the address off the Companion app's streaming screen:

```
.venv/Scripts/python examples/check_neon.py
.venv/Scripts/python -m nav --source neon_live --sink web
.venv/Scripts/python -m nav --source neon_live --neon-address 10.0.0.5 --sink web
```

`check_neon.py` connects, receives one frame and exits 0, or says which path it tried and exits
1. On success it also prints the camera matrix after straightening, how many degrees of field of
view the straightening crops off, and the measured offset between the laptop's clock and the
glasses'. It exits 1 if the glasses do not hand over their calibration. `check_neon.py` and the
`neon_live` command with `--neon-address` were run on 2026-10-05 with the glasses worn, on a school
network where discovery was not tried. The figures from that day are under *Measured on the
glasses* below. The `neon_plugin` command has not been run, because there was no plugin recording
on hand. The Pixel runs on 2026-10-02 used the
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

### Recording what the glasses send

A frame log starts after the depth model, so it cannot test anything upstream of it. To replay the
glasses from the network up, record their raw stream while they are worn:

```
.venv/Scripts/python examples/capture_neon_stream.py --neon-address 10.0.0.5 --seconds 240 frame_logs/captures/walk
.venv/Scripts/python -m nav --source neon_live --neon-replay frame_logs/captures/walk --sink web
```

The capture keeps the scene video as the compressed packets the glasses sent, with the gaze, the
IMU, the calibration and the clock offset beside it. Nothing is decoded while recording, so the
capture never falls behind. `--neon-replay` feeds it through the same decoder, depth model, scene
and planner as a live run, at the pace it was recorded, with the timestamps moved to now. The run
ends when the capture does. A 240 s capture is about 200 MB.

The glasses' client runs in a process of its own, at above-normal priority, and hands frames over
through shared memory. In the same process, the depth model and the planner held Python's global
lock long enough to starve the video decoder, and frames reached the planner 4 to 12 s late.

Every frame the planner took, or tried to, gets a line in `timing.jsonl` in the same directory:
when the frame was captured, when it reached the laptop, when its depth was ready, when its plan
was done, and where its floor came from. A frame skipped for having no usable floor still gets a
line, with no floor and no plan time, so the floor acceptance rate reads back honestly. The times
are on the laptop's clock. Capture is only there for a source that measured the offset between
its clock and the laptop's, which today is the Neon. Summarize a run with:

```
.venv/Scripts/python examples/timing_report.py frame_logs/walk
```

It leaves out the first 10 seconds by default, while the network and the GPU settle, and prints
each share's median, 95th percentile and worst case, the planned frame rate, the longest gap
between plans, and the floor sources. Run it on the live recording, not on a replay. A replay
carries the walk's capture, arrival and depth times beside its own plan times, so its shares mix
two runs. `--verbose` prints the same shares per frame while a run is going, along with the
observed heading.

## Checking the planner on a recording

`nav.evaluation.check_planner` replays a recording through the scene and planner and prints the
figures the planner is judged by. Pass the same scene flags the recording needs, as for any replay.

```
.venv/Scripts/python -m nav.evaluation.check_planner numbers frame_logs/walk
.venv/Scripts/python -m nav.evaluation.check_planner flips frame_logs/walk
.venv/Scripts/python -m nav.evaluation.check_planner band frame_logs/walk
```

- `numbers`: how often the arrow sits at its sidestep limit, split by how far away the nearest thing
  ahead is, how often it swings from one limit to the other, the alarm figures, and how much
  consecutive plans disagree. Each target gets PASS or FAIL, or NOT JUDGED under 100 frames.
- `flips`: the frame pairs whose plans disagree the most, sorted into side flips, new obstacles,
  phone turns, shifts on the same side and tracker jumps, with the largest few to open in the recording.
- `band`: with something 3 to 5.32 m ahead and the arrow at its limit, which cause would release it,
  found by re-planning each frame without one suspected cause at a time.

`--set <field>=<value>` changes a planner setting for the run, and the output marks it.
`--cached` keeps the scene pass between runs. The measured results and the exact commands behind
them are in `../docs/evaluation/arrow_flips_and_band.md`.

`tests/test_planner_golden.py` pins the same figures on two committed slices of recorded walks. A
planner change moves them on purpose: update the expected values in the same commit and say why.
`tests/README.md` says how to cut a new slice.

## The layers

**Sources** are device specific and yield `DepthFrame` objects. The two RGB sources share one
depth estimator. **Scene** turns a depth image into obstacles on the ground: unproject, find the
floor, keep what is between ankle and head height, group into cells, and measure how much each
group's distance has been wobbling. That wobble is N and the distance is S. **Planner** builds a
field over future time and lateral position from two surprises, the professor's, which grows
with N over S, and the surprise of the body touching something, and runs dynamic programming
through it. It also remembers the plan it made one frame earlier, relative to the walker, and
charges any path for straying from it over the first second, so a near-tie doesn't flip sides
every frame. **User model** watches the planner's output and the walker's heading and measures
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

## Driving haptics from the path

Nothing in the pipeline vibrates anything yet. This is what a haptics driver could listen to.

Every planned path goes out as the same JSON message, to the phone over the path connection on 9100
and to the web page. A vibration motor, a wristband or the phone's own haptics can be driven from
either. The fields and their exact meaning are in `docs/arcore_wire_format.md`, under *What the
laptop sends back*.

The useful values come in two kinds, and it helps to think of a pair of headphones. Some values
are a level, like the volume control: they say how strongly to buzz, or how fast. Others are a
switch, like the noise cancelling button: they say whether a mode is on.

| Field | Kind | What it says | Range now | What it could drive |
|---|---|---|---|---|
| `first_heading_radians` | level, with a side | Where the path is heading, read 1 s ahead. Positive is right | 0 straight ahead, up to ±0.62 rad (±35.5 degrees) | Which side buzzes, and how strongly |
| `avoidance_surprise_bits` | level | How soon the walker reaches the nearest thing in its way | 0 with nothing in the way, 0.72 at 1 s to contact, rising as contact nears | How urgent the buzz is: its strength or its pulse rate |
| `alarm` | switch | Something in the walker's corridor is under 0.7 s away at walking pace, or was in the last 0.5 s | `true` or `false` | A distinct pattern that means stop or step aside |
| `scene_information_bits` | level, usable as a switch | How much what the camera saw changed the plan from the plan with nothing in view | 0 for an empty scene, at most about 6.3 | Whether to say anything at all. At 0, the scene changed nothing, so the motor can stay quiet |
| `lateral_offsets_meters`, `times_seconds` | the whole plan | Where to be sideways, 0.1 s apart, out to 3.8 s | positive is right | A warning before a turn the plan already contains |

`cumulative_cost_bits` is not on the list. The contract says it is for display and logging, not
for steering, because its size depends on how many costs the planner adds up, not on how close
anything is.

Four things about these values matter for haptics more than for the screen:

- **The heading flattens at its limit.** ±35.5 degrees is how far the walker can sidestep at
  1.0 m/s while walking at 1.4 m/s. With something 3 to 5.32 m ahead, the arrow sat at that limit
  on 30.3 % of the classroom recording's frames and 1.6 % of `pixel_walk_3`'s. Heading strength
  mapped straight onto vibration strength buzzes at full in exactly those moments.
  `avoidance_surprise_bits` is the value that still changes there.
- **The alarm already holds, the heading does not.** Once raised, the alarm stays up for at least
  0.5 s, so it does not flicker from one path to the next. The heading's only steadying is the
  planner's pull toward its previous plan, which took swings from one sidestep limit to the other
  from 76 to 105 a minute down to 0 to 1.3 on the Pixel recordings. That pull lapses when the
  previous plan is more than 0.5 s old, and on the glasses walk 56 % of the gaps between plans
  were longer than that. On the glasses, a driver that ramps its strength over a few paths will
  feel steadier than one that jumps.
- **Paths come about 1.7 times a second on the glasses.** Measured on 2026-10-05 at
  `--process-resolution 336` on the Quadro T2000 laptop, over the school Wi-Fi. The longest gap
  between two paths was 1.4 s. A driver has to hold the last value between paths, and should stop
  once no path has come for longer than that, rather than keep buzzing on old news.
- **A path describes the world as it was when the frame was captured.** On that walk a frame's
  capture to its finished plan took 0.9 s at the median and 2.0 s at the 95th percentile, and the
  arrow on the phone looked another 0.5 to 1.5 s behind to the walker, judged by eye rather than
  measured. `timestamp_seconds` is the frame's own stamp, so a driver on the laptop's clock can
  tell how old a path is. For the glasses, that stamp is on the Neon's clock, 1.27 to 1.33 s behind
  the laptop's on that day.

## Measured on the glasses

One session, 2026-10-05, in a classroom. The conditions apply to every figure below:

| | |
|---|---|
| Laptop | Quadro T2000 with Max-Q, 4 GB, driver 581.95, on mains power |
| Software | Python 3.12.10, torch 2.14.1+cu126, DA3METRIC-LARGE, `--process-resolution 336` unless a row says otherwise |
| Network | the school Wi-Fi, laptop and Companion phone on it, address given with `--neon-address`. Round trip 7 to 8 ms |
| Clocks | laptop 1274 ms ahead of the Neon on the walk, 1286 ms on the second capture, 0.0 ms drift over the replay |
| Not recorded | the Companion phone's model and app version. The walk had no timed warm-up. The pipeline had been run several times earlier in the session |

**Latency on the live walk.** About 5.5 minutes indoors, 570 frames after the first 10 s, from
`timing_report.py`. Median, 95th percentile, worst:

| Share | median | 95th | worst |
|---|---|---|---|
| capture to arrival | 155 ms | 714 ms | 1.6 s |
| arrival to depth ready | 431 ms | 776 ms | 1.3 s |
| depth ready to plan done | 275 ms | 753 ms | 1.5 s |
| capture to plan done | 905 ms | 2.0 s | 2.9 s |

1.68 planned frames a second, longest gap 1.4 s. The decoder fell behind 15 times for a moment.
The arrow on the phone's browser page looked another 0.5 to 1.5 s behind to the walker, by eye.
Standing still for about 30 s, 504 planned 0.69 frames a second and 280 planned 2.34.

**The floor.** The live walk ran before the depth was converted from the model's 300 pixel focal
length, so its figure, `fitted` on 243 of 570 frames (42.6 %), describes the bug and not the
pipeline. The second capture replayed with the conversion, 432 frames: `fitted` on 395 (91 %),
the camera a median of 1.56 m above the floor, 1.41 to 1.66 m between the 10th and 90th
percentiles. The 37 others were walls, refused for leaning a median of 82.6 degrees. The same 432
frames at the old scale fitted 13 (3 %).

**On a replay.** `check_planner numbers` on that replay's frame log printed the same output three
times running. The arrow sat at its sidestep limit on 5.1 % of frames, and the alarm was on for
24.1 %. On the `--neon-replay` run, the scene took a median of 126 ms a frame and the planner 131 ms,
with 95th percentiles of 423 and 371 ms. A raw capture replayed through `--neon-replay` took a median of 86 ms on the laptop
from a packet being fed in to its decoded frame reaching the pipeline, 137 ms at the 95th percentile.

The choice of the glasses assumed about 10 frames a second at reduced resolution. The measured 1.68 is the
figure to plan with on this laptop. The depth model alone tops out at about 3.5 a second at 280.

## Known limits

- ARCore depth is computed from motion. It stops updating when the walker stands still, drops
  out on blank walls, and is unreliable on moving objects. Static obstacles first.
- Estimator depth flickers between frames and its scale can drift. The pose never comes from it.
- Without a position there is no memory of obstacles out of view, and N is approximate because
  the ground grid moves with the walker.
- Motion prediction in the planner is off by default. The scene reports a group's velocity from
  its centroid in the world frame, which needs a source with a position.
- The user model's time constant, b, is a placeholder until a walker is measured.
- The planner's walker sway, how far a person drifts from the line the arrow asks for, is an
  assumed 0.10 m. No recorded walk had anyone steering by the arrow, so it has not been measured.
- The planner's memory of its previous plan was tuned and checked on replays only. It has not run
  live on the phone or the glasses yet.
- The depth conversion from the model's 300 pixel focal was checked on a replayed glasses capture,
  not on a live walk. Every `video_file` figure taken before it was added used depth at the wrong
  scale.
