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

That adds torch, Depth Anything 3, the Pupil Labs client and its recording reader. The torch that pip picks is the CPU
build, which is fine for a recording and too slow for a live walk. For live use install the CUDA
build that matches your driver from pytorch.org first, then run the line above. The first run
downloads the metric depth checkpoint, 1.3 GB.

### The CUDA build, as installed on the Quadro T2000 laptop

On 2026-10-03, with NVIDIA driver 581.95 (which reports CUDA 13.0), the line that worked was:

```
.venv/Scripts/python -m pip install torch==2.14.1+cu126 --index-url https://download.pytorch.org/whl/cu126
.venv/Scripts/python -m pip install -e ".[dev,glasses]"
```

The cu126 build was picked over cu130 because its kernel list includes the T2000's Turing GPU,
sm_75. `torch.cuda.get_arch_list()` prints that list. Installing the glasses extra afterwards kept
the CUDA build in place. Check it did, every
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
| `neon_recording` | a native Neon recording, straightened with its own calibration, through the depth estimator | `--recording-dir`, `--recording-rate` (2 frames a second of recording) |
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

Every sink starts on the main thread before the first frame, so its port is open from the start.
After that, paths reach the sinks from a publisher thread, the moment each one is planned. That
includes the debug window, which OpenCV draws fine from that thread on Windows and Linux. macOS
only allows windows on the main thread, so `debug_window` isn't supported there. Use `web`.

The web page gets every path's arrow, but its plan view and depth picture are drawn on a thread of
their own, at most 10 times a second, from the newest frame. Drawing one takes about 50 ms. Done on
the publisher thread, it held up the phone's next path and slowed the planner.

The page also plays sound, in two modes picked from its "Sound" control, both from numbers the
laptop sends with every path. **Alarm** beeps toward the side the arrow points to, and while the
alarm is up the beep quickens, rises in pitch and moves to the danger's side, which the laptop sends
as `alarm_pan`. **Noise cancellation** plays music instead, a built-in bed or a file picked on the
page, with each ear at the gain the laptop sends: the ear away from the heading goes quieter by the
course's surprise of the heading error, and the danger's ear drops to a floor while the alarm is up.
A label beside it says what noise cancellation would do, "ANC on" or "ANC disabled", because no page
can switch it. When no path has arrived for 1.5 s the beeps stop, the music plays on at full in
both ears, which is no cue, and the label reads "unknown". Browsers refuse sound before a tap, so
the control has to be touched once on the page that should play. The formula is in
`docs/math/09_arrow_and_alarm.md`.

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
  leave the rest dark. Light gray cells are floor outside the camera's view. Dark gray cells are in
  view, but the camera didn't see floor there, because something stands in front of it or the depth
  has no reading. The planner treats both as empty floor.
- **Path.** The planned path as a band the body's width, from a dot at the walker to an arrowhead
  where the plan ends. Its color runs blue to red as something in the way gets closer, fully red
  from the moment the alarm raises, 0.7 s to contact by default. Its fill is more solid the more the scene shaped the plan, and its borders
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
gravity, read from the pose, whenever the source says its orientation is gravity aligned. The
Pixel does, and so does the Neon, which has an IMU and no position at all. It has to be: the
Pixel's depth image arrives in the sensor's landscape orientation however the phone is held, so
with the phone in portrait the image's own up points sideways, and measured against it every floor
leans 90 degrees. A source that doesn't say, a plain video, gets the image's up. The ceiling came from the first Pixel walk, where ARCore handed over a plane 2.3 m
down, a meter below the real floor, and nothing refused it.

A phone held in the hand and pointed at the pavement still leans about 40 degrees from gravity,
so the live runs set `--floor-max-tilt 50`. The metric model returns no camera intrinsics for a
plain video, so `video_file` assumes `--fallback-fov` degrees of horizontal field of view, 100 by
default. A phone is nearer 75.

The metric model answers as if every camera had a 300 pixel focal length, and the depth source
converts its output to meters with the focal it was given. So the assumed field of view only
changes distances along the direction the camera points. Too wide, and they come out short. Up and
sideways don't move, because the focal cancels out of both. For a level camera that leaves the
floor where it was. Tilted down, the floor tilts and moves too. On a camera pitched 40 degrees
down with a real 75 degree view, assuming 100 reads a 1.5 m height as 1.19 m and leans the floor
12.5 degrees. The first outdoor recording read the camera 2.27 m above the floor at 100 degrees
and 1.84 m at 75, but that was before the conversion existed, so those two figures no longer apply.

`neon_live` ignores the flag. It reads the glasses' own calibration when it connects, straightens
every frame and the gaze point with it, and hands the straightened camera matrix on, so there is
nothing to guess.

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

A run that ends with `[winerror 10013] an attempt was made to access a socket in a way forbidden by
its access permissions` is different: nobody holds the port, but Windows reserved a block around it
when it started. That list shows the block without a `*` beside it. On 2026-10-07 it was 8725 to
8824, which swallows 8765. The service that reserves these blocks has to be stopped for the
reservation above to go through, so in an elevated PowerShell:

```powershell
net stop winnat
netsh int ipv4 add excludedportrange protocol=tcp startport=8765 numberofports=1 store=persistent
net start winnat
```

Or, for one run, give the page another port with `--web-port`, outside every block in the list.

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
glasses'. It exits 1 if the glasses do not hand over their calibration. It then watches the IMU for
3 s and prints how many readings arrived and how many were empty. It exits 1 if none arrived, or if
every one was an empty orientation, because then no frame of a walk would have a pose and every
floor would be fitted against a level head. On 2026-10-05 the IMU sent nothing but zeros, timestamps
included, for minutes. Reproduced on 2026-10-08 with the glasses: the phone serves the IMU stream to
one client, and a second client connected while the first is still open gets packets that decode to
nothing, 478 of 478 in the test, while the first keeps receiving. It clears the moment the first
connection is gone, which a crash or a kill does on its own. A run that is still alive and not
reading, such as one hung in another terminal, is what holds it. Close that run, or, as Pupil Labs
answered the same report ([pl-realtime-api issue 71](https://github.com/pupil-labs/pl-realtime-api/issues/71)),
force-stop and restart the Companion app. Recording on the phone is unaffected.
`--neon-replay <capture>` runs the same check on a capture folder. `check_neon.py` and the
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

### The timing log

A run with `--record-to` also writes `timing.jsonl` into that directory, one line for every frame
the laptop received. `--timing-log <file>` writes the same log to that file without recording any
frames, which is what a replay that only measures uses. A run with neither writes none, and the
two together are refused, as is a `--timing-log` file that already holds lines.

Each line says when the frame was captured, when it reached the laptop, when the worker started on
it, how long the scene, the planner and the user model took, when its depth was ready, when its
plan was done, when the phone sink finished writing its path to the phone, and where its floor
came from. Its `outcome` says what became of it:

- `published`: planned, and its path handed to every display. `sent_seconds` is empty when no
  phone was connected or the run has no phone sink.
- `superseded`: planned, and a newer plan went out first. Only the newest path is ever sent.
- `dropped`: replaced by a newer frame before the worker took it.
- `skipped`: refused. A frame the scene refused for having no usable floor has no floor and no
  plan time, so the floor acceptance rate reads back the way it happened. A frame the planner
  refused after the floor was found keeps its floor.
- `in_flight`: still unfinished when the run ended.

A line is written when its frame's fate is settled, so the file is in that order, not in frame
order. Sort by `timestamp_seconds` for frame order. A run that closes cleanly ends the file with
one more line, `{"log_closed": true, "lines_written": n}`. A log without it was cut short: the run
was killed or its writer failed, and the report warns that it may be only the first part. Logs
written before 2026-10-07 never have it.

The times are on the laptop's clock. Capture is only there for a source that measured the offset
between its clock and the laptop's, which today is the Neon. Summarize a run with:

```
.venv/Scripts/python examples/timing_report.py frame_logs/walk
```

It leaves out the first 10 seconds by default, while the network and the GPU warm up, and prints
each share's median, 95th percentile and worst case, the planned frame rate, the longest gap
between plans, and the floor sources. A log with the newer fields also gets the laptop's shares
toward the phone: the queue wait, the processing, the publish wait, and arrival to sent. The
processing is the scene and the planner. The user model runs after the plan is done, so its time
is part of the publish wait, and the three add up to arrival to sent.
`--verbose` prints the same shares per frame while a run is going, along with the observed heading.

The report takes a record directory or a `--timing-log` file. Given the Pixel's own timing log too,
pulled off the phone as `pixel_app/README.md` describes, it joins the two frame by frame and prints
the whole delay from the phone handling a frame to the arrow drawn from it. The shares are the
phone, the network (both hops together), the laptop and the display, with every count the figures
rest on and a verdict on whether the phone's two clocks share a base:

```
.venv/Scripts/python examples/timing_report.py frame_logs/walk --phone timing_from_phone/timing_2026-10-06T10-12-03Z.jsonl --json frame_logs/walk/report.json
```

`--json` saves the figures, along with whether either log was cut short. It refuses to save over
a log it is reading. `--compare` sets saved reports side by side, at least three runs a side,
and flags any share whose run-to-run spreads overlap, because a difference inside that spread is
not a result. A share that fewer than three runs a side carry, such as the sensor shares when a
run's clocks didn't share a base, is listed as not compared, and a run that was cut short is
named in a warning:

```
.venv/Scripts/python examples/timing_report.py --compare --before frame_logs/before_1.json frame_logs/before_2.json frame_logs/before_3.json --after frame_logs/after_1.json frame_logs/after_2.json frame_logs/after_3.json
```

Run it on a live run or a `--neon-replay` run, not on a `--source logged` replay. A logged replay
carries the original run's capture, arrival and depth times beside its own plan times, so its
shares mix two runs. The report says so when it sees one.

To measure the laptop on a recorded Pixel walk, send the walk over the network instead, so the
laptop runs exactly what it runs on a live walk and stamps arrival itself. Three terminals, from
`server/`, started in this order:

```
.venv/Scripts/python -m nav --source arcore_tcp --arcore-accept-timeout 600 --sink phone_app --sink web --floor-max-tilt 50 --timing-log frame_logs/replay_timing_1.jsonl
.venv/Scripts/python tests/fake_path_reader.py --port 9100 --wait 30
.venv/Scripts/python tests/fake_arcore_sender.py --port 9000 --log-dir frame_logs/wifi_run_2 --realtime --wait 30
```

The page is at `http://127.0.0.1:8765` while it runs. The reader stands in for the phone on the
path port, so each published path gets a send time. Use the recording's own `--floor-max-tilt`,
from its `run_config.json`. Each run needs a new `--timing-log` file.

### Recording what the glasses send

A frame log starts after the depth model, so it can't test anything upstream of it. To replay the
glasses from the network up, record their raw stream while they're worn:

```
.venv/Scripts/python examples/capture_neon_stream.py --neon-address 10.0.0.5 --seconds 240 frame_logs/captures/walk
.venv/Scripts/python -m nav --source neon_live --neon-replay frame_logs/captures/walk --sink web --record-to frame_logs/walk_replay
```

The capture keeps the scene video as the compressed packets the glasses sent, with the gaze, the
IMU, the calibration and the clock offset beside it. Nothing is decoded while recording, so the
capture doesn't fall behind the way a decoder can. `meta.json` is written as soon as the
stream description arrives with the first packets, so a capture stopped early can still be replayed. `--neon-replay` feeds the
capture through the same decoder, depth model, scene and planner as a live run, at the pace it was
recorded, with the timestamps moved to now. The run ends when the capture does. A 240 s capture is
about 200 MB.

The glasses' client runs in a process of its own, at above-normal priority, and hands frames over
through shared memory. When it ran in the same process as everything else, frames reached the
planner 4 to 12 s late. Three things stacked up. The depth model and the planner held Python's
global lock while the client decoded on one thread. The client converted all 30 frames a second
to color whether anyone wanted them or not. And the depth model leaked about 7 threads on every
call. Frames go through shared memory rather than the pipe between the two processes, because a
1600 by 1200 frame is 5.8 MB to copy.

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

## Driving haptics and headphones from the path

The laptop decides the stereo cue and sends it with every path: `alarm_pan`, where the danger is
while the alarm is up, and `ear_gain_left` and `ear_gain_right`, how loud each ear should be. The
web page plays them, see *Sinks* above. Nothing vibrates yet, and no phone API switches noise
cancellation on third-party headphones, so the page only shows what it would do. Which values in
the path message are worth listening to, what they mean in numbers, and how they could drive a
vibration motor or a real noise cancelling switch is in
[`../docs/guides/drive_feedback_from_the_path.md`](../docs/guides/drive_feedback_from_the_path.md).

## Measured on the glasses

Latency, frame rate and floor figures from the first session with the Neon glasses, with the
conditions they were taken under and the commands that rerun them, are in
[`../docs/evaluation/neon_glasses_first_session.md`](../docs/evaluation/neon_glasses_first_session.md).

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
  assumed 0.10 m. `wifi_run_2` is the one recorded walk with the arrow shown, six scorable turns,
  and the sway has not been measured from it.
- The planner's memory of its previous plan ran live once, on the glasses walk of 2026-10-05, where
  it lapsed on 56 % of gaps between frames. It now holds across gaps up to 3 s with its spread
  widening as the plan ages. That was tuned on Pixel walks thinned to the glasses' gaps, because
  the glasses replay has no position and no swings to measure. It hasn't run live since.
- The depth conversion from the model's 300 pixel focal was checked on a replayed glasses capture,
  not on a live walk. Every `video_file` figure taken before it was added used depth at the wrong
  scale.
