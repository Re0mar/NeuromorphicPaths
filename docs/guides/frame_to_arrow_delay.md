# Measuring the delay from frame to arrow

quick how-to. results + what they mean: [../evaluation/frame_to_arrow_delay.md](../evaluation/frame_to_arrow_delay.md)

## what it answers

how long from the Pixel grabbing a depth frame to the arrow planned from that frame being on its
screen. and where that time goes:

- **phone**: frame handled to its depth sent
- **network**: both hops together, phone to laptop and back. can't split them without the two
  clocks agreeing, and they don't
- **laptop**: frame arrived to its path sent back, split into queue wait, processing and publish wait
- **display**: path arrived on the phone to the arrow first drawn with it
- **total**: all four added up

no shared clock needed. each side times durations on its own clock, and the laptop hands back the
frame's own timestamp in every path, so the two logs line up frame by frame.

## setup

- the `server/` venv, made the way `server/README.md` says
- the Pixel app built and installed (`pixel_app/README.md`). any build writes the timing log, but
  take numbers from a release build, see below
- phone + laptop on the same Wi-Fi. laptop's firewall rules for ports 9000, 9100 and 8765, and the
  port reservation, both in `server/README.md`
- recordings and logs live in `server/frame_logs/`, **not in git** (gitignored, big)

nothing to switch on. both sides always log: the phone every session, the laptop on any run with
`--record-to` or `--timing-log`.

## run a walk

laptop's address on the Wi-Fi: `ipconfig`, the IPv4 under the Wi-Fi adapter. Git Bash, from the
repo root:

```bash
cd server && .venv/Scripts/python -m nav --source arcore_tcp --arcore-accept-timeout 600 --reconnect --sink phone_app --sink web --floor-max-tilt 50 --record-to frame_logs/<new walk> --verbose
```

- keep `--sink phone_app` before `--sink web`. the phone's path goes out first that way
- the page is at `https://<laptop address>:8765`. accept the certificate warning the first time
- on the phone: open the app, allow the camera and local network access (Android 17 asks), type the
  laptop's address, tap Connect

hold the phone the way you'll walk with it. nothing arrives at the laptop until ARCore sees the room
and starts tracking, and nothing at all while the app is off screen. keep it in front, screen on.

**warm up first.** leave it connected and planning for 2 min before you start walking. the phone
throttles under ARCore after a bit, and a cold start makes the first minute look faster than the
rest. then walk ~3 min. Ctrl-C the laptop when done.

## pull the phone's log

phone on USB. plugging it back in after the walk asks "Allow USB debugging?" on the phone again,
and adb says `unauthorized` until you accept. Git Bash rewrites the device path unless you stop it:

```bash
cd server && MSYS_NO_PATHCONV=1 adb pull /sdcard/Android/data/com.neuromorphicpaths.pixel/files/timing/ frame_logs/<new walk>/phone/
```

one file per app session, named by start time (`timing_2026-10-06T10-12-03Z.jsonl`). pick the one
from your walk. never `adb shell cat > file`, it adds carriage returns.

## run the report

```bash
cd server && .venv/Scripts/python examples/timing_report.py frame_logs/<new walk> --phone frame_logs/<new walk>/phone/<its file> --exclude-first-seconds 120 --json frame_logs/<new walk>/report.json
```

`--exclude-first-seconds 120` drops the warm-up. same number on every walk you compare.

## what it looks like when it worked

a real one, a 2 min walk on 5 GHz on 2026-10-07 (the counts after this are trimmed):

```
joined paths               1525  (581 frames left out as the cold start)
frame to arrow, ms           median     p95   worst   frames   (percentiles linear between ranks)
  total, handled to drawn      128.6   217.2   332.4     1525
  phone, handled to sent        10.0    20.4    32.0     1525
  network, both hops            11.5    23.0    79.0     1524
  laptop, arrival to sent       90.2   185.2   305.2     1525
    queue wait                  25.1    52.0    81.9     1525
    processing                  60.5   149.5   261.6     1525
    publish wait                 1.1     2.3    12.8     1525
  display, received to drawn    11.4    18.6    23.1     1525
  sensor to handled            138.3   150.7   162.9     1525
  total, sensor to drawn       263.5   357.9   465.0     1525
phone clocks               same base: handled minus sensor 106.2 to 200.7 ms over 5256 frames
WARNING network, both hops came out negative 1 times, left out. First: -0.7 ms
network median per 10 s of the walk, ms (paths)
  0 s: 12 (123), 10 s: 12 (101), 20 s: 12 (184), 30 s: 12 (160), 40 s: 11 (157), 50 s: 12 (103), 60 s: 11 (141), 70 s: 11 (154), 80 s: 12 (111), 90 s: 11 (110), 100 s: 11 (180)
counts
  ...
```

the one negative network value is a frame where the round trip and the laptop's time came out
within a millisecond of each other. one or two of those is stamp jitter. lots would be a bug.
this walk's laptop log also got a warning above the table that it has no closing line, since it
was recorded before the closing line existed. a new log only gets that warning when the run was
killed or its writer failed, and then the numbers may cover only part of the walk.

that's a table of the shares (median, 95th percentile, worst, frame count), a line on the phone's clocks,
the network median for every 10 s of the walk, then the counts. check:

- `joined paths` is 100 or more. fewer and there's a warning on top. walk again, longer
- `paths drawn` is close to `paths received`. a big gap means the drawn stamp isn't firing
- `phone clocks`: `same base` adds a sensor-to-arrow total. `different base` or `not enough frames`
  just means the total starts when the phone handled the frame. still fine
- `not joined` counts are small. lots of `phone sent, no laptop line` means the two logs aren't
  from the same walk. `phone handled, never sent or dropped` counts every frame the phone saw with
  no laptop connected, before you tapped Connect and after the run ended, so a few hundred is normal
- the network line jumping between tens of ms and seconds means a queue is filling and draining
  somewhere between the two. one median over the walk hides that. the one time we've seen it, it
  was the laptop reading frames too slowly, not the Wi-Fi (results doc, walks 2 and 3)

## comparing laptop changes without walking

for anything that only changes the laptop, replay a recorded walk instead of walking again. send it
over the network into a normal live run, so the laptop stamps arrival itself. three terminals, from
`server/`, in this order:

```bash
.venv/Scripts/python -m nav --source arcore_tcp --arcore-accept-timeout 600 --sink phone_app --sink web --floor-max-tilt 50 --timing-log frame_logs/replays/before_1.jsonl
```

```bash
.venv/Scripts/python tests/fake_path_reader.py --port 9100 --wait 30
```

```bash
.venv/Scripts/python tests/fake_arcore_sender.py --port 9000 --log-dir frame_logs/<walk> --realtime --wait 30
```

- a new `--timing-log` file every run (`before_1`, `before_2`, …). an old one gets refused
- report each one: `.venv/Scripts/python examples/timing_report.py frame_logs/replays/before_1.jsonl --json frame_logs/replays/before_1.json`.
  no `--exclude-first-seconds 120` here. that's for the phone warming up on a walk. a replay only
  needs the report's default 10 s, and a recorded walk can be shorter than 120 s
- at least 3 runs before the change and 3 after, each from a fresh process. one run proves nothing,
  floor fits don't repeat exactly
- then:

```bash
.venv/Scripts/python examples/timing_report.py --compare --before frame_logs/replays/before_*.json --after frame_logs/replays/after_*.json
```

a share marked `spreads overlap, not a result` didn't move by more than runs vary. don't quote it.

**not** `--source logged`. it keeps the original walk's arrival times, so its laptop shares mix two
runs and the report warns about it.

glasses instead of the Pixel: replay a capture with `--source neon_live --neon-replay frame_logs/captures/<capture>`
in place of `--source arcore_tcp` and skip the sender. keep `--sink phone_app` and the reader even
though the glasses walker reads the web page: only the phone sink stamps a send, so without it
there's no publish wait. needs the `glasses` extra, see `server/README.md`.

## release build

numbers go in the report from a release build only. a debug build draws slower, and the display
share is exactly what that slows. the build commands are in `pixel_app/README.md`.

## where the result goes

`docs/evaluation/frame_to_arrow_delay.md`. every number with how it was taken:

- phone model + Android version, laptop, which Wi-Fi
- date, walk length, warm-up, debug or release
- the commit both sides ran

a number without those can't be compared with the next one.

## when it goes wrong

- **no `received` lines in the phone log**: the path connection never came up. line three on the
  phone says why. usually the firewall on 9100, or the local network permission
- **no `drawn` lines**: the app's screen was off or the app wasn't in front
- **everything in `not joined`**: phone log and laptop log from different walks, or the phone log
  from a different session. check the file's start time
- **`different base`**: fine. the total just starts when the phone handled the frame, not when the
  camera took it. the results doc says which
- **report says the log mixes two runs**: that's a `--source logged` replay. use the sender above
- **the laptop doesn't stop after you close the app**: the laptop never heard the connection close,
  and it waits on that read with no time limit. Ctrl-C waits for the read too. close the terminal,
  or `taskkill //F` the two python processes. the timing log is already written except the last
  frame or two
- **`adb devices` lists nothing and the "Allow USB debugging?" prompt never shows**: the laptop
  can't see the phone on USB at all. a charge-only cable, or the phone set to charging. tap the
  USB notification on the phone and pick File transfer, or swap the cable
- **`[winerror 10013]` starting the web page**: Windows reserved its port. see `server/README.md`,
  or add `--web-port 8700` for this run. nothing measured changes
