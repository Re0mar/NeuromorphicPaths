# From a Pixel frame to the arrow on its screen

How long does it take from the Pixel's camera taking a depth frame to the arrow planned from that
frame being drawn on the phone? And where does that time go: the phone, the network, the laptop, or
the drawing? This measures it on three walks and on replays of recorded walks, before and after a
change to the laptop.

The change: the laptop used to send a finished path only when the next frame arrived, and it drew
the web page's pictures on the same thread that reads frames. It now sends each path the moment
it's planned, and draws the pictures on a thread of their own.

On two walks the same evening, same phone and same Wi-Fi band, only the laptop's code different,
the median from sensor to arrow went from 3,017 ms to 263.5 ms. Almost all of that came out of what
the report had been calling network. It was frames the laptop hadn't read yet.

How to take these numbers yourself is in [the guide](../guides/frame_to_arrow_delay.md). How to
rerun every figure here is at the end.

## The five shares

A frame's trip from the phone and back as an arrow is cut into four pieces, plus the total.

| Share | In plain words | Measured as |
|---|---|---|
| Phone | the app has the frame, until its depth is handed to the network | `sent - handled`, phone clock |
| Network | the frame on its way to the laptop, plus the path on its way back | `(received - sent)` on the phone clock, minus the laptop share |
| Laptop | the frame arrived, until its path is handed back to the network | `sent - arrival`, laptop clock |
| Display | the path arrived on the phone, until the arrow is first drawn with it | `drawn - received`, phone clock |
| Total | all four together | `drawn - handled`, phone clock |

The laptop share splits three ways. **Queue wait** is the frame waiting for the planner to finish
the one before it. **Processing** is the scene and the planner working on it. **Publish wait** is
the finished path waiting to be sent, and it includes the user model, which runs after the plan
is done and took a median of 0.1 ms on walk 2. The three add up to the laptop share. Publish wait is the part
the change was aimed at.

The phone stamps when the app gets the frame, not when the camera took it. ARCore's own timestamp
says when the camera took it, and when the phone's clocks allow it (see *The clock check* below), a
sixth figure, **sensor to handled**, adds that stretch. The total then runs from the sensor.

## How it was measured

Both sides write a log line per frame, each on its own clock. The phone writes when it handled the
frame, sent it, received the path and drew the arrow, on Android's `elapsedRealtimeNanos`. The
laptop writes when the frame arrived, when the planner started on it and finished, and when its
path went back, on a clock built on Python's `time.perf_counter`.

The two logs never compare a phone time with a laptop time. Every path the laptop sends carries
the frame's own ARCore timestamp, so the two lines for one frame are matched on that number. Each
side then only subtracts its own stamps. The network share is the one figure that needs both: the
phone's round trip minus the laptop's time in the middle. That's why the two network hops can't be
told apart. Splitting them would need the two clocks to agree, and they don't.

The first 10 s of every run are left out. A walk is also reported with the first 120 s left out,
since the phone slows down as it warms up under ARCore. A replay is reported with 10 s only.

## Conditions

These apply to every walk.

| | |
|---|---|
| Phone | Pixel 8, Android 17. The same release build of the app on all three walks, not debuggable, one install checked by its sha256 |
| Laptop | Dell Precision 5550, Quadro T2000, on mains power. Python 3.12.10, numpy 1.26.4, OpenCV 4.11.0 |
| Network | home Wi-Fi, one network name. The phone at `10.0.0.185`, the laptop at `10.0.0.184` |
| Laptop run | `--source arcore_tcp --sink phone_app --sink web --floor-max-tilt 50 --record-to`, the phone sink named first so the phone's path goes out before the web page's |
| Warm-up | none. Every walk started straight away |

And per walk:

| Walk | Laptop code | When | Phone on | Battery before / after | Page open in a browser |
|---|---|---|---|---|---|
| 1, before the change | before | 2026-10-06, 11:23:53 to 11:26:29, about 2.6 min | 2.4 GHz, 2,442 MHz | 25.2 / 30.8 °C | yes |
| 2, after the change | after | 2026-10-07, 19:27:04 to 19:29:06, about 2.0 min | 5 GHz, 5,300 MHz | 26.0 / 30.5 °C | no |
| 3, before the change again | before | 2026-10-07, 19:54:19 to 19:57:00, about 2.7 min | 5 GHz, 5,300 MHz | 30.6 / 32.1 °C | no |

The router moves the phone between its bands on its own, and walk 2 came out on 5 GHz where walk 1
had been on 2.4 GHz. Walk 3 repeats the old code on 5 GHz so that walks 2 and 3 differ only in the
laptop's code. Walks 2 and 3 served the page on port 8700 instead of 8765, which Windows had
reserved. That changes nothing measured.

## Walk 1, before the change

2,405 paths joined across both logs with the first 10 s left out, and 563 with the first 120 s
left out. All values in ms.

| Share | First 10 s left out: median / p95 / worst | First 120 s left out: median / p95 / worst |
|---|---|---|
| Total, sensor to drawn | 1,359 / 3,917 / 4,945 | 602 / 3,100 / 4,314 |
| Total, handled to drawn | 1,229 / 3,780 / 4,809 | 465 / 2,973 / 4,176 |
| Phone | 11.6 / 20.9 / 1,523 | 10.9 / 21.0 / 27.2 |
| Network | 1,126 / 3,652 / 4,594 | 372 / 2,877 / 4,080 |
| Laptop | 69.9 / 143.9 / 194.1 | 84.6 / 138.1 / 194.1 |
| of which queue wait | 7.1 / 56.1 / 103.9 | 10.9 / 55.3 / 103.9 |
| of which processing | 45.8 / 84.6 / 129.2 | 50.2 / 86.6 / 129.2 |
| of which publish wait | 15.0 / 34.6 / 70.0 | 9.7 / 35.3 / 70.0 |
| Display | 11.2 / 19.0 / 30.7 | 11.5 / 19.3 / 26.4 |
| Sensor to handled | 133.4 / 146.5 / 172.6 | 132.5 / 146.5 / 172.6 |

**The network share is most of the delay, and it isn't steady.** Its median over the whole walk
is 1,126 ms, against 69.9 ms on the laptop and about 11 ms each on the phone and the display. Cut
into 10 s stretches, its median goes 2,219, 2,994, 3,055, 1,108, 3,396, 1,635, 637, 267, 130,
1,312, 1,030, 1,648, 42, 122 and 1,900 ms. A delay that climbs to over 3 s and falls back to 42 ms
is a queue filling and emptying, not a slow link. The phone stamps `sent` when its write returns,
which is when the bytes reach its own socket buffer. So any wait between there and the laptop
reading the frame counts as network: the phone's buffer, the air, and the laptop's own socket.
Walks 2 and 3 below show which.

The phone share's worst case, 1,523 ms, belongs to the same queue. 17 of the 3,926 frames the
phone sent took over 100 ms to send, all of them 47 to 84 s into the app's session, where the
network median peaks. A write blocks once the socket buffer is full.

The laptop dropped 1,203 of the 3,655 frames it received as stale, because a newer frame arrived
before the planner was free. It planned 16.8 frames a second.

## What the change did, on replays

A replay sends a recorded walk to the laptop over the network, from the same laptop, at the
recorded pace. The laptop runs exactly as it does on a walk, and a stand-in for the phone reads the
paths. Only the laptop's shares exist on a replay, but every run sees the same frames, so before
and after compare directly. Each figure below is the median of each run's median, with the range
across runs in brackets. A difference is only claimed where the two ranges don't overlap.

**Pixel walk** (2,144 frames 33.3 ms apart, one 31.2 s pause), five runs before and five after.

| Laptop share, ms | Before | After |
|---|---|---|
| Publish wait, median | 11.7 (11.2 to 12.3) | 0.5 (0.5 to 0.5) |
| Publish wait, worst | 43.2 to 81.1 | 3.7 to 5.1 |
| Queue wait, median | 3.7 (3.5 to 3.9) | 8.6 (7.7 to 10.1) |
| Processing, median | 38.9 (34.1 to 39.1) | 38.9 (37.7 to 40.0), ranges overlap |
| Arrival to sent, median | 51.2 (45.7 to 53.4) | 54.2 (52.6 to 56.4), ranges overlap |

The aim was a publish wait with a median under 5 ms and a worst case under 20 ms on these
replays. Every after run meets both.

**Glasses walk** (a 240 s capture from the Neon glasses), three runs before and three after.

| Laptop share, ms | Before | After |
|---|---|---|
| Publish wait, median | 221.8 (194.9 to 237.8) | 1.5 (1.5 to 1.6) |
| Publish wait, worst | 831.2 to 942.3 | 7.2 to 20.0 |
| Queue wait, median | 462.6 (409.6 to 483.5) | 557.2 (551.3 to 606.4) |
| Arrival to sent, median | 1,086.8 (945.1 to 1,137.9) | 952.2 (888.6 to 1,096.4), ranges overlap |

On the glasses the depth model runs on every frame before the planner sees it, about 400 ms each
on this laptop's GPU, and that sets the pace. Before the change the loop took a new frame only after
sending the last path, so no frame was ever dropped. After it, 12 to 19 per run are.

### Why the laptop's total didn't drop

The publish wait fell by 11 ms on the Pixel replays, but arrival to sent didn't fall with it.
Before the change, the laptop drew the web page's depth picture between reading one frame and the
next. Drawing takes about 50 ms, and a frame came every 33 ms. So the laptop fell behind reading,
frames waited unread in its socket, and their arrival stamp came only when they were finally read.
That wait sat outside the laptop's share entirely.

On a replay, sender and laptop share one machine, so the gap between a frame's arrival and its
recorded timestamp should stay constant for a frame read at once. Measured against the lowest value
of that gap over the previous 2 s:

| Pixel replays | Frames read in a burst | Arrival late, median / p95 | Recorded time to sent, median / p95 |
|---|---|---|---|
| Before, five runs | 82 to 150 | 83 to 172 / 671 to 817 ms | 204 to 383 / 733 to 914 ms |
| After, five runs | 0 | 70 to 73 / 78 to 98 ms | 123 to 131 / 164 to 189 ms |
| After, phone sink only | 0 | 70 / 81 ms | 105 / 156 ms |

A burst is a frame read under 5 ms after the one before it, where the recording had them over
30 ms apart. The 70 ms in every after run, the one without the web page included, comes from the
method rather than the laptop. The sender sleeps the recorded gap after each send, so it drifts
later as it goes, and the 2 s minimum lags behind that drift. The before runs sit hundreds of ms
above it at the 95th percentile.

So on the replays, the change's main effect is that frames are read as they arrive. The queue wait
rising from 3.7 to 8.6 ms is the same thing from the planner's side: frames now reach it at the
rate they're sent, and more of them find it still busy.

The same drawing ran on the same thread on walk 1, so some of its network share could be frames
waiting for the laptop to read them. Walks 2 and 3 measure how much.

## Walks 2 and 3, the same band with and without the change

Both on 5 GHz, the same evening, the same phone and app. Only the laptop's code differs. All values
in ms, median / p95 / worst, with the first 10 s left out. Walk 2 was too short for the 120 s cut,
which leaves it no paths, so the two are compared at 10 s.

| Share | Walk 2, after the change | Walk 3, before the change |
|---|---|---|
| Joined paths | 1,525 | 1,583 |
| Total, sensor to drawn | 263.5 / 357.9 / 465.0 | 3,017 / 4,527 / 6,090 |
| Total, handled to drawn | 128.6 / 217.2 / 332.4 | 2,873 / 4,387 / 5,951 |
| Phone | 10.0 / 20.4 / 32.0 | 12.0 / 24.2 / 1,943 |
| Network | 11.5 / 23.0 / 79.0 | 2,710 / 4,201 / 5,706 |
| Laptop | 90.2 / 185.2 / 305.2 | 99.1 / 217.7 / 420.7 |
| of which queue wait | 25.1 / 52.0 / 81.9 | 5.9 / 92.1 / 151.9 |
| of which processing | 60.5 / 149.5 / 261.6 | 69.0 / 133.1 / 271.0 |
| of which publish wait | 1.1 / 2.3 / 12.8 | 19.9 / 46.5 / 128.7 |
| Display | 11.4 / 18.6 / 23.1 | 11.2 / 18.6 / 27.7 |
| Sensor to handled | 138.3 / 150.7 / 162.9 | 139.0 / 167.0 / 187.8 |

The network median in each 10 s stretch:

- Walk 2: 12, 12, 12, 12, 11, 12, 11, 11, 12, 11, 11.
- Walk 3: 2,876, 2,716, 2,601, 3,149, 2,856, 2,632, 2,686, 307, 3,740, 2,430, 1,533, 1,659, 2,415,
  3,939, 3,491.

**The queue was the laptop, not the network.** With the old code it comes back on the same band,
larger than on walk 1. With the new code the network share is 11 to 12 ms in every stretch of the
walk. Walk 3 shows the cause directly. When it ended, the phone had sent 2,420 frames and the
laptop had read 2,390. The other 30 were sitting in the laptop's socket, unread. The old laptop
drew the web picture between reading one frame and the next, fell behind, and the backlog filled
its socket and then the phone's send buffer. The phone's write blocked, and the phone dropped
frames rather than queue them: 2,209 on walk 3, against 103 on walk 2.

The laptop's own share is about the same on both walks, 90.2 against 99.1 ms at the median. Its
queue wait went up, from 5.9 to 25.1 ms, for the reason the replays showed: frames now reach the
planner as fast as they're sent, and more of them find it busy. On walk 2 the laptop dropped 1,640
of the 3,180 frames it received as stale. On walk 3 it dropped 763 of 2,390, because the phone had
already dropped most of the rest.

## The clock check

The sensor-to-handled stretch compares ARCore's camera timestamp with the phone's handled stamp,
two different clocks on the phone. They can only be subtracted if they count from the same start.
The report checks this on every walk. On walk 1, across 5,757 frames, the gap ran 80.9 to 254.7 ms,
with a median of 137.1, a 5th percentile of 124.2 and a 95th of 181.3 ms. Walk 2 ran 106.2 to 200.7
ms over 5,256 frames and walk 3 ran 52.9 to 198.0 ms over 5,066. Two clocks counting from different
starts would differ by the phone's whole uptime. So on the Pixel 8 they share a base, and the
totals above include the 133 to 139 ms before the app sees a frame.

The check accepts a gap between 0 and 500 ms with a 5th-to-95th spread of at most 100 ms, over at
least 100 frames. The first frame of the walk carried an ARCore timestamp of 0, from before ARCore
had a clock reading. It's counted and left out of the check.

## What the numbers rest on

On walk 1, with the first 10 s left out:

- The phone handled 5,190 frames, sent 3,655 and dropped 632 whose slot a newer frame took first.
- The laptop received the same 3,655. It sent 2,412 paths back, dropped 1,203 as stale, and had
  40 superseded by a newer result before they were sent.
- The phone received 2,411 paths and drew 2,405. The 6 received and never drawn came at the end,
  when the app left the screen.
- Nothing went unmatched between the logs. No frame was sent with no laptop line, and no path was
  sent to the phone and never received.
- 903 frames were handled and never sent or dropped. These are frames the app saw while no laptop
  was connected: before Connect was tapped, and after the run had ended.
- No phone log lines were lost and none were duplicated.

On walks 2 and 3, the same way:

- Walk 2: the phone sent 3,180 frames and the laptop received all of them. 1,540 paths went back,
  all received, and 1,525 were drawn. No phone log lines were lost.
- Walk 3: the phone sent 2,420, and the laptop received 2,390. The 30 frames sent with no laptop
  line are the ones still unread at the end. 1,600 paths went back, 1,586 were received and 1,583
  drawn. One path was sent and never received, at the end.
- Walk 3's laptop run didn't end on its own. The app was closed, but the laptop never heard the
  connection close, and the old code waits on that read with no time limit. It was stopped by
  hand. Every frame had its laptop log line by then except the last, whose path the old code was
  still holding back for a next frame.
- On both, the frames handled and never sent or dropped (1,393 and 265) are the ones the app saw
  with no laptop connected.
- All three laptop logs predate the line a clean close now writes at the end of the log, so the
  report warns about each that it may be only part of the run. For walks 1 and 2 that's only
  their age: both runs ended normally. Walk 3's really was cut short, by the one frame above.
- Walk 2 has one path whose network share came out at -0.7 ms, a round trip and a laptop share
  within a millisecond of each other. It's left out and counted.

## What this doesn't tell you

- **Which network hop is slow.** The two hops are one figure, for the reason above.
- **When the light leaves the screen.** The display share ends when the app's draw call returns
  with the new arrow. The screen's own refresh comes after that and isn't measured.
- **Anything beyond one phone on one network.** One Pixel 8 on one home Wi-Fi, on both of its
  bands. Walks 2 and 3 settle the fix on 5 GHz. The new code wasn't walked on 2.4 GHz.
- **The phone's warm-up after the change.** Walk 2 was about 2 min, too short for the 120 s cut
  that leaves the phone's warm-up out.
- **The glasses walker's display.** On the glasses the walker reads the laptop's web page, which
  this doesn't time past the moment the path is handed over.

## Known biases

- **Decoding a frame counts as laptop.** The laptop stamps arrival after reading a frame and before
  decoding it. Decoding takes a median of 0.10 ms on 160x90 frames, at most 0.23 ms, so it moves no
  share by a visible amount.
- **The recording itself costs time on a walk.** A recorded walk writes each frame to disk on the
  laptop's main thread before the planner gets it. Before the change, that also held up reading the
  next frame. Its cost wasn't measured separately.
- **The web page's pictures still take CPU.** After the change they're drawn on their own thread, at
  most 10 a second, from the newest frame, and every path still goes out to the phone and the page.
  Drawing one took a median of 49.3 ms, 77.6 ms at the 95th percentile and 142.3 ms at worst, over
  688 pictures on a replay. That thread shares the laptop with the planner.
- **The laptop clock's resolution.** The laptop's stamps come from `time.perf_counter`, resolved to
  well under a microsecond. `time.monotonic` would have moved in 15.6 ms steps on this laptop, about
  the size of the publish wait, which is why it isn't used.
- **Walk 3 came after walk 2.** The phone started it at 30.6 °C against 26.0 °C. Its phone share's
  median was 12.0 ms against 10.0, so the warmer phone added about 2 ms where the network share
  differed by 2,700.
- **The arrival-lateness figures are an upper bound.** They count the sender drifting late as well
  as the laptop reading late. The run without the web page gives the floor that drift alone makes.

## Rerunning it

The recordings and logs aren't committed, because they're large. They live under
`server/frame_logs/` on the machine that took them:

- walk 1: `before_walk_1`, with its phone log in `before_walk_1_phone/`
- walk 2: `after_walk_1`, with its phone log in `after_walk_1_phone/`
- walk 3: `before_walk_2_5ghz`, with its phone log in `before_walk_2_5ghz_phone/`
- the Pixel recording `wifi_run_2`, the glasses capture `captures/neon_walk_2`, and every replay's
  log and report under `replays/`

Each walk's folder holds its two reports, `report_exclude10` and `report_exclude120`, as `.json`
and `.txt`. From `server/`:

```
# A walk's report, here walk 1. Add --exclude-first-seconds 120 for the second cut
.venv/Scripts/python examples/timing_report.py frame_logs/before_walk_1 --phone frame_logs/before_walk_1_phone/timing_2026-10-06T09-23-27Z.jsonl
.venv/Scripts/python examples/timing_report.py frame_logs/after_walk_1 --phone frame_logs/after_walk_1_phone/timing_2026-10-07T17-26-11Z.jsonl
.venv/Scripts/python examples/timing_report.py frame_logs/before_walk_2_5ghz --phone frame_logs/before_walk_2_5ghz_phone/timing_2026-10-07T17-54-02Z.jsonl

# The replay comparisons, from the saved reports
.venv/Scripts/python examples/timing_report.py --compare --before frame_logs/replays/before_*.json --after frame_logs/replays/after_*.json
.venv/Scripts/python examples/timing_report.py --compare --before frame_logs/replays/neon_before_*.json --after frame_logs/replays/neon_after_*.json
```

A new replay follows the three-terminal steps in the guide. The worst publish waits come from each
replay's own report. The arrival-lateness table was a one-off calculation over the replays' timing
logs, using each line's `arrival_seconds` and `timestamp_seconds` as described above. The picture
drawing time was measured by timing the web page's drawing over a `--source logged --realtime`
replay of `wifi_run_2`.
