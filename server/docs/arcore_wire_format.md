# ARCore depth frame wire format

Written for whoever builds the Android side. You send depth frames to the laptop over TCP, and the
laptop sends planned paths back. This document is the contract. If the two sides disagree, this
file is right and the code that does not match it is wrong.

The laptop listens on a port you choose with `--arcore-port`, default 9000. It accepts one
connection at a time. It also listens on a second port, `--phone-port`, default 9100, for the
path connection. You open both, to the one laptop address you already have, and nothing on the
laptop ever needs your address.

---

## Framing

Every message, in both directions, is a 4-byte big-endian unsigned length followed by exactly that
many bytes. In Kotlin that is `DataOutputStream.writeInt`, which is already big-endian, then the
payload.

A depth frame payload has three parts, in this order:

1. A UTF-8 JSON header.
2. A single newline byte, `0x0A`.
3. The raw depth array bytes.

The newline is the separator, so the header must not contain a literal newline. Compact JSON on
one line, which is what every standard serializer produces by default.

The laptop refuses a message claiming more than 64 MB. A length that large almost always means the
stream is out of step rather than that a real frame is that big.

---

## The header

| Field | Type | Unit | Required | Notes |
|---|---|---|---|---|
| `version` | integer | | yes | Must be `1`. See *Versioning* below |
| `timestamp_seconds` | number | seconds | yes | Your clock. The laptop does not reinterpret it, it only uses differences |
| `depth.dtype` | string | | yes | `uint16`, `float16` or `float32`. See *Depth values* |
| `depth.shape` | `[rows, columns]` | pixels | yes | Two positive integers, height first |
| `depth.byte_length` | integer | bytes | yes | Must equal `rows * columns * bytes per value` |
| `intrinsics` | 3 by 3 array | pixels | yes | Row major, at the depth image resolution, not the camera preview resolution |
| `pose.orientation_wxyz` | 4 numbers | | yes | Quaternion, w first. ARCore gives you x, y, z, w, so reorder it |
| `pose.position_xyz` | 3 numbers or `null` | meters | yes | `null` when tracking is lost |
| `pose.has_position` | boolean | | yes | Must agree with whether `position_xyz` is present |
| `ground_plane` | object or `null` | | yes | `null` if you have no plane. The laptop fits one itself in that case |
| `ground_plane.normal` | 3 numbers | | when present | Unit vector |
| `ground_plane.offset_meters` | number | meters | when present | Signed, so that `normal · point + offset == 0` on the plane |
| `gaze_pixel` | 2 numbers or `null` | pixels | yes | `null` for the Pixel. It exists for the eye tracking glasses |

**Every field in that table is a required key, including the nullable ones.** Send `"gaze_pixel":
null` rather than leaving the key out. A key you deliberately have no value for and a key you
forgot look identical otherwise, and only one of those is a bug worth telling you about.

Every number must be finite. A JSON `NaN` or `Infinity` is not valid JSON anyway, and the laptop's
parser rejects it rather than reading it as a number.

---

## Depth values

Send what ARCore gives you. Its depth image is 16-bit millimeters, so `uint16` is the cheapest
option and keeps your side a straight buffer copy.

| `dtype` | Bytes per value | What the value means |
|---|---|---|
| `uint16` | 2 | Millimeters. The laptop divides by 1000 |
| `float16` | 2 | Meters |
| `float32` | 4 | Meters |

The array is row major, `rows * columns` values, no padding and no stride. If ARCore hands you a
buffer with a row stride wider than the image, repack it before sending.

A depth of 0 means ARCore had no estimate for that pixel. Send it as 0 rather than dropping the
pixel. The laptop treats 0 and anything outside 0.1 to 30 meters as invalid and ignores it.

---

## Pose and the frame it is in

`has_position` is the field the laptop branches on, so it matters more than it looks.

- **`has_position` true.** Points are placed in the world frame and can persist across frames.
  Send a real position.
- **`has_position` false.** The laptop rebuilds everything from scratch each frame in the camera's
  own frame. Send this whenever ARCore's tracking state is not `TRACKING`.

Do not send a stale position with `has_position` true when tracking has dropped. A position that
is wrong by a few meters puts the whole point cloud somewhere it is not, and nothing downstream can
tell that happened.

Orientation is always required, even when position is not.

**The world has y up.** The orientation rotates the laptop's camera axes (x right, y down, z
forward) into ARCore's world, where +y is straight up and the other two axes are horizontal. The
laptop reads gravity from that whenever `has_position` is true, and uses it to tell which way is
down when it looks for the floor. It has to, because the depth image arrives in the sensor's
landscape orientation whatever way the phone is held, so with the phone in portrait the image's
own up points sideways. A sender that does not know which way gravity is sends `has_position`
false and the laptop falls back to the image's up.

---

## Versioning

`version` is checked before anything else. The laptop decodes version 1 and refuses anything else
by number.

When the format changes in a way that would break an older reader, the version goes up and both
sides change together. Adding a new optional field does not need a version bump, because an older
reader ignores keys it does not know. Changing the meaning or the unit of an existing field does.

---

## What the laptop sends back

A planned path, framed the same way: a 4-byte big-endian length, then UTF-8 JSON. No header and
newline split, because there is no binary part.

You open this connection, the same way you open the depth one, to the path port. The laptop writes
one message per planned path and never reads from this socket. You never write on it. A phone that
reconnects gets the next path, not a replay of the one it missed: a path is a decision about this
instant, and the next frame produces the next one within a frame interval.

```json
{
  "timestamp_seconds": 12.345,
  "times_seconds": [0.0, 0.1, 0.2],
  "lateral_offsets_meters": [0.0, 0.05, 0.12],
  "first_heading_radians": 0.0423,
  "alarm": false,
  "cumulative_cost_bits": 18.4
}
```

| Field | Meaning |
|---|---|
| `times_seconds` | How far into the future each offset is |
| `lateral_offsets_meters` | Where to be at that time, sideways from straight ahead. Positive is right |
| `first_heading_radians` | Where to point the arrow now. Positive is right |
| `alarm` | Something is under a second from contact. Turn the display red |
| `cumulative_cost_bits` | Total cost of the chosen path. For display and logging, not for steering |

`times_seconds` and `lateral_offsets_meters` always have the same length.

Every key is required and every number is finite. Per field, who produces it and who checks it:

| Field | Produced by | On the wire | Read by | Value domain | Who enforces it |
|---|---|---|---|---|---|
| `timestamp_seconds` | the depth frame the path was planned for | JSON number | display, logging | finite, your clock's seconds handed back | laptop refuses a non-finite path before encoding, you check finite |
| `times_seconds` | planner, its time step and horizon | JSON array of numbers | the arrow, later a ribbon | finite, at least one entry, same length as the offsets | both sides, you refuse a length mismatch or an empty array |
| `lateral_offsets_meters` | planner | JSON array of numbers | the arrow, later a ribbon | finite, positive is right | both sides |
| `first_heading_radians` | planner | JSON number | the arrow | finite, positive is right, within the sidestep limit | both sides |
| `alarm` | planner, time to contact under a second | JSON boolean | display color | `true` or `false`, never a number | you refuse a number where the boolean belongs |
| `cumulative_cost_bits` | planner | JSON number | display, logging | finite, zero or more | both sides |

---

## A worked example

A 2 by 2 depth frame with a known ground plane and no gaze. Produced by the encoder, not written
by hand, so it is exactly what you will receive.

Total message, 401 bytes. Length prefix:

```
00 00 01 8d
```

That is 397, the number of bytes that follow. Then the header, 380 bytes of UTF-8, shown here with
indentation it does not have on the wire:

```json
{
  "version": 1,
  "timestamp_seconds": 12.345,
  "depth": {"dtype": "float32", "shape": [2, 2], "byte_length": 16},
  "intrinsics": [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
  "pose": {"orientation_wxyz": [1.0, 0.0, 0.0, 0.0], "position_xyz": [0.0, 0.0, 0.0], "has_position": true},
  "ground_plane": {"normal": [0.0, 1.0, 0.0], "offset_meters": -1.6},
  "gaze_pixel": null
}
```

Then one newline byte, `0a`. Then 16 bytes of depth, little-endian float32, which is what every
Android and x86 device writes natively:

```
00 00 c0 3f 00 00 00 40 00 00 20 40 00 00 40 40
```

Those are 1.5, 2.0, 2.5 and 3.0 meters, in row major order.

**The depth bytes are little-endian and the length prefix is big-endian.** That looks inconsistent
and it is deliberate. The length prefix is big-endian because that is what `writeInt` and network
byte order give you for free. The array is native-endian because both sides are little-endian and
byte-swapping a depth image per frame would cost real time for nothing.

---

## Per-field trace

Who produces each field, how it travels, who reads it, and what range it may hold. The point of
this table is that nothing is aligned by assuming both sides derive from the same idea.

| Field | Produced by | On the wire | Read by | Value domain | Who enforces it |
|---|---|---|---|---|---|
| `version` | Android, constant | JSON integer | decoder | exactly 1 | decoder, before any other field |
| `timestamp_seconds` | ARCore frame timestamp | JSON number | scene history, replay pacing | finite, increasing | decoder checks finite, scene uses differences only |
| `depth.dtype` | Android, constant per build | JSON string | decoder | one of three names | decoder, against the allowed list |
| `depth.shape` | ARCore depth image size | JSON array of 2 | decoder, unprojection | two positive integers | decoder |
| `depth.byte_length` | Android, computed | JSON integer | decoder | equals shape product times item size | decoder, and again against the bytes that follow |
| depth array | ARCore depth buffer | raw bytes after the newline | scene unprojection | 0 means unknown, 0.1 to 30 m usable | decoder converts units, scene drops the rest |
| `intrinsics` | ARCore camera intrinsics | JSON 3 by 3 | scene unprojection | finite, at the depth resolution | decoder checks shape and finiteness |
| `pose.orientation_wxyz` | ARCore pose, reordered | JSON array of 4 | scene frame transform | finite, non-zero | decoder checks shape, pose construction checks the rest |
| `pose.position_xyz` | ARCore pose | JSON array of 3 or null | scene frame choice | finite when present | decoder, and consistency against `has_position` |
| `pose.has_position` | ARCore tracking state | JSON boolean | scene, chooses body or world frame | true or false | decoder, must match whether position is present |
| `ground_plane` | ARCore plane, when found | JSON object or null | scene floor fit | null means fit one here | decoder, scene falls back |
| `gaze_pixel` | not sent by the Pixel | JSON null | planner goal | null | decoder |

---

## What happens when something is wrong

Every violation below produces one error naming the field, the frame is dropped, and the
connection stays open. The laptop does not guess at a value and does not close the socket on a bad
frame, because one corrupt frame on a busy network is normal and losing the connection for it is
not.

| What you sent | What the laptop says |
|---|---|
| A version other than 1 | names the version it got and the one it speaks |
| A missing key, including a nullable one | names the key |
| A `dtype` outside the list | names the value and lists what is allowed |
| `byte_length` that does not match `shape` and `dtype` | gives both numbers |
| Fewer or more bytes after the newline than `byte_length` | gives both numbers |
| A number where a boolean belongs, or the reverse | names the field and both types |
| `NaN` or `Infinity` anywhere | names the field |
| `has_position` true with a null position, or the reverse | names the disagreement |
| `intrinsics` that is not 3 by 3 | names the shape it got |
| No newline in the payload | says the header is truncated |

Two cases are different, because they mean the stream itself is out of step rather than that one
frame is bad:

- **A length prefix over 64 MB, or zero.** The connection is treated as desynchronised.
- **The connection closing mid-message.** The reader reports how many bytes it was still waiting
  for, rather than blocking forever.

### The path direction

The same rules, applied by you. A path message that breaks a field rule above is dropped on the
phone with the key named, and the connection stays up. A zero length prefix, or one over 1 MB (a
path is a few hundred bytes), means the stream is out of step: close the connection and reconnect,
which is what the laptop does with a bad prefix on the depth side. The laptop never closes the
path connection for a bad message, because it never reads one. It closes it only when the run
ends, and it counts the paths it had nobody to send to.

---

## Testing your side without the laptop

`tests/fake_arcore_sender.py` in this repository sends synthetic frames in this exact format using
the same encoder the laptop decodes with. Point your reader at it, or read its output, to check
your framing before you have a laptop in front of you.

The reverse also works. Run the laptop with `--source arcore_tcp` and connect your app to it. The
laptop logs the first frame's shape and dtype, so a mismatch shows up immediately rather than as a
planner producing nonsense.
