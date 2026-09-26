# Methods notes

Rough explanation of how everything works and how we tested it, so whoever writes the real methods section has stuff to pull from.Actual math is in `docs/math/push_field.md` if you need the real formulas.

## The big idea

- we are NOT finding the sidewalk and then drawing a path through it. we did that first and threw it out
  - thrown out because it was trying "old methods" to do the initial lane borders, and then trying to use surprise only to refine/path. Prof gave us advice to just use surprise for "everything" instead
- instead: the app finds obstacles, gives each one its own "surprise" score, adds them all up for every direction you could walk, and the direction with the lowest total wins
- so the path comes out of the obstacles, there's no path to find first
- walking is the main thing. biking is a maybe-later
- we play back a recorded walk first, and only use the live camera after the math is sorted
- we use models that already exist, nothing gets trained

## Where we started, and why we changed (do we put this in report? maybe just add but keep short?)

- **the proposal**: use the Meta AI glasses to recognize the path you're walking or biking on and keep you on it without running into stuff. the plan was phases:
  - phone app hooked up to the glasses, find the path in single pictures, work out distance and angle to the path's edges, record a run to measure our own noise
  - add GPS for speed and heading, first estimate of Tau (how long until you leave the path)
  - use lots of frames, second Tau estimate from frame to frame, a "precision" per person from how much they drift side to side
  - a warning tied to Tau, and a user study comparing it to a plain distance warning
- **the first version we built**:
  - a sidewalk-finding model (YOLOv8n-seg, trained on 1,000 sidewalk frames from Belgium) that outlines the path
  - worked pretty well on the dataset's own pictures (overlap score 0.892 on 8 stills)
  - worked out the camera's tilt from the path's edges, but that only gave a reliable answer on 1 of 8 dataset pictures and 3 of 6 of our own photos
  - then surprise and Tau got applied to where you are inside that path
  - we also switched cameras along the way, from the Meta glasses to the Pupil Labs Neon, recording with it and looking at the recordings afterwards
- **why we threw it out (25 Sept)**: the professor's feedback was basically that we'd built a normal lane-keeping app with the course's ideas stuck on at the end. almost all our time went into the normal part (finding the path, camera tilt, edges). what he wanted was the other way around: find the obstacles and the places you shouldn't be, give each one its own surprise, and let those add up to the path. so there's no path to find first
- **what carried over**: the surprise and Tau math (Tau is still in there as "time until you'd reach an obstacle"), the idea of a per-person wobble (we measured it, it just turned out not to be the right thing to set turning from), GPS speed as a backup, and the plan to replay recordings before going live
- possible write up: one paragraph like "our first design did X, feedback pointed out Y, so we switched to Z".

## How the app is put together

- Android app in Kotlin
- split into parts: where the pictures come from (camera, a video file, later the glasses), what finds obstacles, what works out where they are, what picks the direction, and what shows it
- any part can be swapped without touching the others
- all the math is in parts that don't need a phone, so it can be tested on a laptop
- if the detector is slower than the camera, the app just skips old frames instead of falling behind
- the whole thing runs in the background now, so you can use another app and the arrow still shows up on top of it

## Finding obstacles

- detector is YOLO-World. you give it the list of things to look for as words, so we didn't need to train anything
- normal detectors only know stuff like person, car, chair. they don't know tree, pole, fence or bin, which we need
- 35 words ("prompts") that map to 11 kinds of obstacle: tree, barrier/fence, person, animal, car, bike, pole, hole, table, chair, trash can
- picture size 640. we tried 320 because it's way faster but it missed the stuff that matters most (pillars, a lamp post, the bike rack). at 320 it found 455 boxes on the walk, at 640 it found 1509
- careful with the word list. adding the plain word "wall" made it find almost no people anymore (381 person boxes dropped to 34). changing one word changes everything else
- so every time we change the words we test it on a few frames first, and now also on the moments we tuned the math on (adding "concrete pillar" randomly added a pole next to the cone)
- the detector's score isn't a real probability. we count 0.5 and up as "definitely there" and trust lower scores less. that came from one cone that scored 0.45 and got basically ignored when we used the score as-is

## Working out where stuff is

- direction to an object comes from where its box is left to right, and how wide the camera sees
- distance comes from where the bottom of the box touches the ground, using how high the camera is and how it's tilted (from the phone's sensors)
- if the bottom of the box is cut off by the edge of the picture, it guesses from how tall that kind of thing usually is
- camera height is still a guess, 1.4 m

## Ground, walls and buildings

- boxes can't describe stuff like a wall, a building or grass, so there's a second model, SegFormer, that labels every patch of the picture
- it runs every third frame to save time, and the app reuses the last result in between
- its 150 labels get grouped into 9: pavement, grass, dirt, road, building, wall, stairs, sky, other
- ground type costs a little per meter you'd walk on it: grass a bit, dirt a bit more, road a lot, pavement nothing
- walls and buildings: it finds where they touch the ground and turns those spots into obstacles, one every half meter so a long wall doesn't count 50 times
- stairs are on the screen but don't count as obstacles yet, because on our outdoor walk the stairs were literally the path

## The math (the push field)

- for one object and one direction, the chance of hitting it = chance it's in your way × chance you reach it soon × chance it's really there × chance it matters if you touch it
- "in your way" depends on how far off your line it is. "soon" depends on how many seconds until you'd get there. every kind of obstacle has its own settings for how much space it needs, how far ahead in time it matters, and whether bumping it is ok (bumping a person is less bad than a car lol)
- surprise = minus log2 of the chance you DON'T hit it, in bits. so if you'd probably miss it the surprise is small
- surprises for different objects just add up
- try every direction from 60 degrees left to 60 right, 1 degree apart (121 directions)
- turning is a little surprising by itself because people mostly walk where they're facing. that's the "prior", set to 30 degrees
- lowest total = the direction the arrow shows
- the warning number = how much worse the way you're actually going is than the best way. zero if you're already doing the best thing
- why not just add up push arrows like forces: two objects on either side would cancel and aim you at a gap that's too small, and something dead ahead pushes you nowhere. scoring every direction fixes both
- picking the direction with the lowest surprise is basically one step of active inference, which is the connection to the lectures

## Things moving between frames

- the app tracks boxes from frame to frame so one missed detection doesn't flip the arrow
- it tries to work out how fast things are coming at you from how their distance changes, but that number is really noisy (sometimes it even says standing things are moving away)
- so it's only allowed to make things MORE urgent, never less

## Walking speed

- counts your steps with the phone's motion sensor, steps per second × 0.7 m
- works indoors and doesn't need location permission, GPS is just the backup
- when you stop, it goes to zero after 1.5 s without a step
- the math treats a stopped walker as going 0.3 m/s so the arrow still works on your first step
- result: when you stop, stuff that's just standing there stops mattering, but someone walking toward you still does

## The projected path (the ribbon)

- the app "walks" forward 6 steps of half a meter, picking the best direction at each step, and draws that as a path on the ground
- drawn as a wide ribbon, a fifth of the screen wide at your feet and getting narrower, like a strip of ground
- color = surprise. blue when things are fine, red from 3 bits up
- see-through-ness = how much what's in view changed the app's mind compared to just walking straight. that's the KL divergence, aka Bayesian surprise. solid when a wall or gap decided the path, lighter when nothing did
- it always fades out at the far end so it looks like it's disappearing, not ending
- we first tried fading it by entropy, but entropy barely moved on the whole walk (6.73 out of max 6.92 bits) so it was useless

## When there's no way through (STOP)

- normal warning can't tell you "everything is blocked" because it only compares your way to the best way
- so the app also checks the least surprising direction there is, but looking twice as far ahead in time (4 s instead of 2)
- we had to double it because at the normal look-ahead a wall 2 m ahead looked the same as squeezing past a post, both about 3 bits
- red from 3 bits, STOP if it stays at 4 bits for a whole second, U-turn once you've actually stopped
- on our walk it never would have said STOP (good, there's no dead end in it). a fake wall across the path makes it say STOP about 1.5 m before you hit it
- first idea was to check along the projected path, but the path just walks right through walls, so that didn't work

## What shows up on screen

- in the app: the camera picture, boxes, the ribbon, the arrow and a bunch of numbers
- over other apps, two options:
  - small: one thick arrow in the corner (straight, slight turn, hard turn, STOP or U-turn), blue to red
  - big: just the ribbon over the whole screen, half see-through normally and 0.8 when things are bad. you can still tap stuff underneath it

## How we tested

- recorded one outdoor walk with the phone's sensors logging next to the video
- play it back on the phone with the tilt the camera really had at each moment, and log everything the app decides
- the log has every obstacle for every frame, so the math can be rerun on a laptop in like a second with different settings, no phone needed
- checked the numbers against two moments we know happened: the traffic cone at ~184 s (walker went left) and the bike rack at ~210 s (walker went right)
- a real turn = the walker's direction changed 15+ degrees in 2 s, not just the path curving. then we see if the app called it
- lots of small automatic tests, 134 right now
- timing numbers only count from a cool phone. it gets hot during a replay and the detector slows from ~550 to ~750 ms

## Stuff we tried that didn't work (maybe useful for "why we did it this way")

- finding the sidewalk first (the whole first version)
- setting how much you can turn from how much you wobble while walking. wobble is about 4 degrees, using it made the app never turn at all, even for the cone. wobble and turning are different things
- trusting the detector score as-is (the cone disappeared)
- "anything moving away doesn't count" (noise made real stuff vanish)
- fading the path by entropy (didn't move)
- checking for "no way through" along the path (walks through walls)
- picture size 320 (missed the important stuff)
