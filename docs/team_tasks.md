# Team tasks

Ok so here's all the stuff that still needs doing. If you want one, put your name next to it in the table and push it. You can do one with someone else too, just put both your names.

| # | Task | What you need | Name |
|---|---|---|---|
| 1 | Check if the app's arrows match where the walker actually turned | laptop | |
| 2 | Make a test that yells at you if the walk's results change | laptop | |
| 3 | Figure out which obstacle settings actually matter | laptop, do 1 first | |
| 4 | Figure out how much we can trust the detector's scores | laptop, some sorting by hand | |
| 5 | Check the app's distances with a tape measure, and fix the camera width on playback | laptop, a phone, a tape measure | |
| 6 | Time how late the arrow shows up | laptop or phone | |
| 7 | Test three things the app just assumes | laptop | |
| 8 | Check if the walking speed is right on a real walk | any phone, a walk | |
| 9 | Try the app on real people | people and a phone, **not yet**, we have to pick how the app tells people where to go first | |

## When you're done

You're done when you've got whatever the task asks for and (ideally) you wrote a quick how-to in `docs/guides/` so somebody else could do it again without texting you. Just say what you were trying to find out, what you had to install, where the files are, what you typed and in what order, what it looks like when it works, and what you found out.

## Getting the recording

The walk we recorded isn't on GitHub because there are random people in the video. Ask in the group chat and someone will send the zip. Inside there's the video, two files from the phone's motion sensors (`Orientation.csv` and `TotalAcceleration.csv`), and a text log from when the app played the walk back on the Pixel. The log has one line for every frame of the video, and under that a line for every obstacle the app saw in it.

The video starts 4.61 seconds after the sensors do. You don't have to do anything about that, the tools already know.

When a command asks for a file, give it the whole path, like starting from `C:/`. Short paths don't work, trust me.

## 1. Check if the app's arrows match where the walker actually turned

Basically most of the app's settings are guesses we agreed on, nobody actually tested them. The big one is how much the app expects you to just keep walking straight (it's set to 30 degrees right now). The motion sensor file shows exactly when the walker turned and which way, and nobody's compared the app to that yet.

Someone already started on this. This command runs the recorded walk through the app a few times with different settings and counts how many of the walker's turns the app got:

`./gradlew :math:priorSweep -Preplay.log=<whole path> -Porientation.csv=<whole path> -Pout.dir=<whole path>`

What you need to do is find every time the walker really turned (like actually changed direction, not just the sidewalk curving). For each one, check if the app pointed the same way and how early. Also count how many times the app said to turn when the walker didn't. Do it once with the normal settings and then a bunch of times with different numbers for the "keep walking straight" setting, and put it all in a table.

## 2. Make a test that yells at you if the walk's results change

Right now if anyone changes a setting, the results for the whole walk change and nobody notices unless they sit there for 15 minutes playing it back on a phone. Which nobody does.

So make a test that runs the recorded walk through the app on your laptop and checks a few numbers: how many times the warning goes over 0.3, what the biggest warning is, how big the warnings are at the traffic cone (around 184 seconds) and at the bike rack (around 210 seconds), and how much the arrow jumps around between frames. Let each number be off by a little, and write a note saying where you got it from. Then if someone changes a setting later, they have to update the numbers too, so nothing changes by accident.

You can put the text log in the repo, it's just text. There's already code that reads it in `math/src/test/kotlin/.../tools/ReplayLog.kt`.

## 3. Figure out which obstacle settings actually matter

Every kind of obstacle (people, trees, cars, etc.) has three settings in `math/.../ObstacleProfile.kt`. There's how much space to give it, how far ahead of time it starts to matter, and whether bumping into it is a big deal or not. They were all guesses. We just want to know which two or three actually change anything so we can stop arguing about the rest lol.

Change one setting at a time, try a few different numbers, and count how many frames end up with a different arrow or warning. Then write down which settings change stuff and which don't do anything. You can reuse whatever you made for task 1.

We already think two things are off. The app thinks about a sixth of the ground right in front of you is road, when a lot of it is actually red brick, and it counts walking on road as bad. Also if you walk next to a wall at about arm's length it freaks out and tells you to turn hard, even though you're totally fine. The log doesn't say what kind of ground the app saw, so to check the road thing you'd have to play the walk back on a phone or add that to the log.

## 4. Figure out how much we can trust the detector's scores

The detector gives every box it draws a score. Right now the app treats anything 0.5 or higher as "100% definitely there" and trusts lower scores less and less. We only picked that because of one traffic cone that scored 0.45.

Run the detector on frames from the video until you have a few hundred boxes. Sort them by score and go through and mark each one as real or not real. Make a graph of the score vs how often the box was actually real. Then use the graph to change `confidenceForCertainty` in `PushFieldParameters`, or change the whole rule if a straight line doesn't really fit.

There's no script for saving the boxes yet so you'll have to write a small one. The README says how to get the detector.

## 5. Check the app's distances with a tape measure, and fix the camera width on playback

The app figures out how far away stuff is using how high the camera is, how it's tilted, and how wide it can see. The height is still just a guess (1.4 m).

**Fix the camera width first please**, because tasks 1 to 4 all use the recorded walk. When the app plays the walk back, it thinks the camera sees 66 degrees wide, but the Pixel's camera actually sees 56. So every distance on playback is off. The 66 is in `AppDefaults.CAMERA_INTRINSICS`. Make playback use the real number, and add a test that checks the distance math with a fake camera where you already know the right answers.

Then do one session inside. Put some stuff at measured distances, hold the phone at a height you measured, and see if the app's distances match the tape measure. Put the real height in `AppDefaults.CAMERA_HEIGHT_METERS`.

## 6. Time how late the arrow shows up

The app can only handle about 2 frames a second, so a warning can pop up almost a whole second after the thing happened. We need that number for the report.

Save the time each result shows up on the screen, and figure out how long it took since the camera took the picture. Report the normal delay and the worst one, and keep how long the detector took separate. You can test everything except the detector on the phone emulator using the fake detector. We already know the detector takes 550 to 800 ms on the Pixel, depending on how hot the phone is.

## 7. Test three things the app just assumes

The app assumes three things that nobody checked:

- It counts every box as a different object, but sometimes the same thing gets two boxes and gets counted twice.
- It never tells you to turn more than 60 degrees either way.
- It acts like nothing is ever closer than 0.2 seconds away.

Write one test for each one, and name it after what's supposed to happen. The same thing shouldn't get counted twice. If something needs more than a 60 degree turn to get around, the app should still give the best direction it can. If something is super close, the warning should be really big but not infinite. If the app gets any of these wrong, fix it while you're at it.

## 8. Check if the walking speed is right on a real walk

When you stop walking, stuff that's just standing there shouldn't count as a warning anymore, but a person walking at you still should. The app figures out how fast you're going by counting your steps with the phone's motion sensor. It works on the recording but nobody's tried it on an actual walk yet.

Go on a walk and stop a few times, with the app's log running (`adb logcat -s Guidance:D`). Compare the app's speed to the GPS speed when you're outside, and to how many steps you counted when you're inside. At one of the stops, stand next to a wall and check that the wall's warning goes down to almost nothing, but someone walking toward you still gets a warning. Any Android phone works.

## 9. Try the app on real people

**Not yet.** We can't start this until the team decides how the app tells people which way to go. The options are in `docs/proposals/rider_feedback_options.md`.

After that, write down how the test is going to work, pick a route with obstacles you already know about, and get a few people to walk it with the app on and then with it off. See how fast they react, how often they go the way the arrow says, and how big the warning has to be before they notice it.
