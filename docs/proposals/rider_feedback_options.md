# Proposal: ways to show the surprise signals to the rider

> **This is a proposal for discussion, not a design.** Nothing below has been tried with people.
> It lists options to choose between. Before any of it is built, the questions in *Before building
> any of this* need answers, and some of them need a pilot with real users.

---

## What there is to show

The app computes two signals every frame, following `docs/math/surprise_and_tau.md`.

- **The line signal.** How far the rider is from where they should be, usually the middle of the
  path, measured against how much the path tolerates. It is always present and changes smoothly.
  The app turns it into a number from 0 (on the line) to 1 (at the edge of the tolerance), on a
  logarithmic scale because perception is logarithmic. This number is computed but not shown.
- **The edge alarm.** On when, at the current sideways drift, the rider would leave the path within
  about a second. It switches off only once that time is back above about a second and a half, so
  it doesn't flicker. It is silent while the rider moves parallel to the edge. Today it shows as a
  red banner, "Edge in 0.9 s".

The math document treats these as two channels on purpose, and keeps them apart so the study can
tell which one drove a correction. The display could keep them apart too.

## What limits the choice

- **The project is about distraction.** The report's introduction starts from accidents caused by
  looking at phones. Feedback that asks the rider to look at a screen works against that.
- **The device isn't settled.** The app runs on a phone's camera today. The Neon glasses planned for
  the study track the eyes and have no display. The Meta glasses the first app used can play sound,
  and Meta's toolkit has a display API for models with a screen.
- **Cycling rules out a hand-held phone.** Holding a phone while cycling has been banned in the
  Netherlands since July 2019. A screen only works mounted.
- **Walking and cycling differ.** Street noise masks quiet sounds, and a bike's vibration masks
  light haptics.

## Options for the line signal

| Option | How the signal maps | For | Against |
|---|---|---|---|
| L1. Edge glow on screen | The side of the screen toward the drift glows, brighter as the signal grows | Direct, shows direction | Needs the rider to look at the screen |
| L2. Target line on the camera view | A line drawn where the rider should be, colored by the signal | Shows where to go, not only that something is off | Same, and draws attention to the phone |
| L3. Spatial sound | A soft tone on the side of the drift, louder as the signal grows | Eyes stay on the path, carries direction | Constant sound tires people, masked by traffic, needs earphones |
| L4. Haptic | Vibration that pulses faster as the signal grows | Private, no looking or listening | Carries no direction on one phone, masked on a bike |
| L5. Not shown at all | The line signal is only logged, for the analysis | Nothing competes for attention, and the study can still fit it | The rider gets no help until the alarm |

## Options for the edge alarm

| Option | What happens | For | Against |
|---|---|---|---|
| A1. Banner (current) | A red message with the time to the edge | Already built, shows the number | Needs the rider to look |
| A2. Sound | A short tone when it switches on | Reaches a rider looking elsewhere | Missed in traffic noise, startles |
| A3. Vibration | A distinct pattern when it switches on | Private, works with the phone in a pocket or mount | Weak on a bike, easy to mistake for a notification |
| A4. Two stages | Vibration first, sound if it stays on | Gentle for short drifts, firm for long ones | Needs a second threshold, which the math doesn't define yet |
| A5. Direction cue | Sound or vibration on the side of the edge | Tells which way to steer | Needs stereo sound or two haptic points, a phone has one |

## What the course material says that bears on this

- Perception is logarithmic, which is why the line signal is already scaled that way.
- Vertegaal et al. (section 6.5) showed their participants a fixed-width feedback bar. That is the
  closest published display, and it was visual.
- Murray-Smith et al. (section 5.2) note that users struggle with displays of uncertainty. That is
  an argument for showing a decision (an alarm) rather than a quantity, at least at first.

## Before building any of this

- **Which device the study uses**, since it decides which options exist at all.
- **Whether the alarm is right often enough.** A walk with the debug build, counting false alarms
  and missed edges, should come first. A signal that is often wrong trains people to ignore it,
  whatever form it takes.
- **Reaction time per option.** A small pilot, a handful of people on a quiet path, comparing how
  fast each alarm option gets a correction.
- **Annoyance over a longer walk.** Especially for any continuous sound or vibration.
- **Whether the display itself changes behavior.** The math document plans to test dependence on
  the display by removing it after exposure. The chosen option has to allow that.
- **Accessibility**, for riders with hearing or vision limits.

## A possible starting point, not a decision

If the group wants something to pilot first, one low-cost combination is L5 with A3: log the line
signal without showing it, and vibrate when the alarm switches on. It keeps the rider's eyes free,
needs no earphones, and changes the least while the alarm's accuracy is still unknown. A walk with
the current banner and the debug readout would come before that, to see how often the alarm is
right. This is a suggestion to react to, not a recommendation to build.
