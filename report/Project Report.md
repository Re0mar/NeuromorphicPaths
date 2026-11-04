# THE PLAN
- Abstract
  
- Introduction (Daniël)
    - Explain the problem
    - General approach idea?

- Background (Kars)
    Why the problem?
    - Motivation
        - Prior knowledge
      - Explain Depth Map (Depth anything 3)
      - Explain Surprise Equations (using it twice (weight and tau)) (docs/math)
      - Hamiltonian path routing
      - Open 3d dense slam
papers roel: mute conversation in cars

- Methods (Nolan & Dirk)
    
    - Formulation of Depth map from surroundings
    - 4 main math formulas from presentation
      - Possible additional content from NeuromorphicPaths\docs\math if relevant & significant
    - Explain pipeline
        - Translating Depth map to Surprise Map
          - Brief mention of height map usage (floor and ceiling so we narrow what points we actually care about from depth map)
        - Choosing the next step of least surprise -> cost/risk
            - Forming Path through Surprise Map
        - Gaining output from the path
        - Collision detection
          - Surprise
          - Alarm
          - Gaze tracking
    
- Results
    
    - We have demo, but no real tests to show it works
    
- Discussion
    
    - Next steps
	    - What is implemented, but not rigurously tested (and not part of the current prototype)
        - Usage of magic numbers
        - Inclusion of accelerometer and gps?
    - Presentation mentions
        
- References
# Introduction
In recent years, the problem of distracted people taking part in trafic has been on the rise, caused for the most part by increased usage of mobile phones and similar devices.
The effects of this have been pretty clear, causing many traffic accidents which would've been preventable [(Rijniers, 2017)](#references).
In an effort to reduce these collisions our group set out to research a potential aid based upon surprise theory, 
where the user is either alerted or has the distractions on the phone muted whilst their environment is to surprising, 
as this increase in surprise should correlate with an increased risk for accidents.
Put into practice, our team will work with an camera device to generate a map of the surroundings,  
after which Fitt's Law is used to identify areas of high surprise.
These high surprise areas will then be used to alert the user, whilst also steering them clear of it, theoretically eliminating the potential for accidents or collisions to occur.

# Background
## Motivation

A 2017 thesis published at Radboud University concluded that reminding drivers to pay attention to the road has a significant effect on subsequent accidents [(Rijniers, 2017)](#references). Many of these accidents are preventable one-sided accidents, caused by drivers, cyclists and pedestrians leaving or crossing the road by accident or at the wrong moment. Our leading theory is that wearable computer vision, as found in commercial products like smart glasses, could detect these dangerous situations and direct the user's attention back to the road before an incident occurs.

Alerting the user is only one option. The distractions on the phone can also be muted while the environment is too surprising, similar to how conversations can be muted in cars when the driver needs to focus. 

>> *[Roels papers on muting conversation in cars and summarize]*

Research on auditory interfaces gives some support for muting as a strategy. Hearing is always available and cannot be turned off, which is why people use headphones to curate their own soundscape. This comes at a cost, since it lowers situational awareness and carries distinct safety risks when used on the move [(Haas, 2023)](#references). Haas also found that when people are asked which sounds in their environment they want to change, they rarely want a sound louder, but often want it muted or reduced, while keeping the information the sound carries. A simple interface with a few discrete states (muted, quieter, unaltered, louder) was preferred over continuous control, although this difference was not statistically significant [(Haas, 2023)](#references). This fits our approach, where distractions are only reduced when the surroundings require it.

The same work also warns about the downside of automation. Systems that alter the soundscape automatically may give the user the illusion of an unaltered environment and an incomplete picture of the situation, and Haas therefore recommends that users keep control over such alterations [(Haas, 2023)](#references). For our system this means that muting should be limited to moments of clear risk, and that the user should always be able to see why something was muted or flagged.

The hardware we use also limits what we can do. The Pupil Labs Neon glasses are monocular, have eye-tracking cameras, and have no other sensing capabilities such as LiDAR. Any sense of distance must therefore be inferred from a single RGB image, which led us to a monocular depth estimation model.

## Surprise

Interactive inference describes human-computer interaction as a form of Bayesian inference, where movement is the result of the sensorimotor system reducing the difference between its predictions and what it observes [[3]](#references). In this view, Fitts' Law (the relationship between the distance to a target, its width, and the time needed to reach it) follows from this inference process. Moving towards a target is described as relaxing along a surprise potential. This potential is approximately the negative log probability of the goal distribution and equals a KL divergence. It has the same form as the potential energy of a spring [[3]](#references): the further the current state is from the goal, relative to the tolerance around it, the higher the surprise.

We apply this idea to the physical environment of the user. An obstacle is a place the user should not end up. The surprise of a point in the environment depends on the signal-to-noise ratio of the gap between the user and that point:

$$U = \tfrac12 \left(\frac{\Delta S}{S}\right)^2$$

where $S$ is the gap and $\Delta S$ is the uncertainty of that gap. We use this formulation twice in the pipeline:

1. **As a weight for every cell.** The floor plan is divided into cells. For each cell the uncertainty is the measured *wobble* $N$, and the gap is $S = d - 0.35$, which gives $U=\frac{1}{2}\left( \frac{N}{S} \right)^2$. Here $d$ is the distance between the centre of the person and the closest point of the object, and 0.35 m accounts for the forward-most part of the user. This value is the cost of a cell in the planner.
2. **As time to impact.** When the uncertainty is one second of movement, the equation becomes $U=\frac{1}{2}\left( \frac{1s}{\tau} \right)^2$, where $\tau$ is the time until impact. This form is used for collision detection, with the nearest obstacle straight ahead: an obstacle that will be reached soon is much more surprising than one far away, and the surprise grows quadratically as $\tau$ shrinks.

Both forms express how reliably the user can predict the distance to an obstacle. The exact definitions of $N$ and the cell grid are given in the [Methodology](#methodology).

## Depth Estimation

Depth Anything 3 is a model that "predicts spatially consistent geometry from an arbitrary number of visual inputs, with or without known camera poses" ([source](https://depth-anything-3.github.io/)). We use the metric large variant, which turns a single frame from the glasses into a depth map in meters. This is important for our application, since the surprise equations above require real distances rather than relative depth. The model only outputs depth, so the ground and the height of points have to be derived afterwards from this depth map and the IMU data of the glasses.

## Path Planning

Once every cell of the environment has a surprise value, the next question is which route through them is best. Following the idea that movement is the relaxation along a surprise potential [[3]](#references), the safest route is the one that accumulates the least surprise over the next few seconds. This is computed as a minimum-cost path through the cost field (Hamiltonian path planning, solved by dynamic programming), which is then reduced to a single heading for the user. The cost of following the resulting path, expressed in bits of surprise, can also be measured afterwards by the user model, which only observes and does not steer anything.

## 3D Reconstruction

[Dense SLAM](https://www.open3d.org/html/tutorial/t_reconstruction_system/dense_slam.html) (**S**imultaneous **L**ocalization **a**nd **M**apping), as implemented in the Open3D library, builds a 3D reconstruction of the environment from RGB-D input while tracking the movement of the camera through it. Because the surprise of a cell depends on how it changes over previous frames, knowing the movement of the camera is useful in principle. In the current prototype this is an optional component that is disabled by default (see the [Methodology](#methodology)).

# Materials
## Neon smart glasses
A pair of [Pupil Labs Neon smart glasses](https://pupil-labs.com/products/neon) were provided to us by the course coordinator and act as the accessory to provide a real-time stream of images to be analyzed. Included with this is a [Motorola Edge 40 Pro](https://mijnmoto.nl/product/motorola-edge-40-pro/) smartphone, which comes paired with the glasses and allows us to control, review and analyse the recordings. 

## Image analysis
Depth Anything 3 is a model that "predicts spatially consistent geometry from an arbitrary number of visual inputs, with or without known camera poses" [Source](https://depth-anything-3.github.io/). It is used to estimate a depth map given an input of a single image.

[RGB-D SLAM](https://www.open3d.org/html/tutorial/t_reconstruction_system/dense_slam.html) is a (dense) implementation of **S**imultaneous **L**ocalization **a**nd **M**apping included in the Open3D library. The depth maps generated by Depth Anything 3 can optionally be fed into SLAM (disabled by default), which builds a 3D reconstruction of the environment and the movement of the camera through it.

# Methodology

## System Overview
- glasses stream video + IMU + gaze to Neon-paired phone Companion app, which then streams over wifi to Server
- background process on Server decodes all 30 fps but keeps only newest frame, older ones get tossed
- frame gets straightened with glasses' calibration, and gets IMU reading closest to when it was captured (within 50 ms, or we can't figure out orientation at all and ignore it)
- Depth Anything 3 (metric large) turns it into a depth picture (meters). Only depth comes out, no height map from model
- *(optional, off by default)* `--position odometry` or `dense_slam` adds a camera position at this step. (see Part 1 of experiment section)
- scene: depth → 3D points → find the ground/floor → only keep points between ankle and head height → chop into 0.25 m floor cells → each cell's distance and wobble (N)
  - <define wobble better here, word is ambiguous>
- planner: cost field over next few seconds, dynamic program for cheapest path, generate arrow heading (not directly shared with user), then alarm + gaze silencing
- "user model" for actual navigation: just measures what each avoidance cost in bits, doesn't steer anything
- planning runs on a worker thread that always grabs newest frame, so when it's busy frames get dropped (not queued! always as "live" as possible)
- output: Server sends plan (arrow, alarm, numbers) to webpage over HTTPS -> page decodes the glasses' video itself and draws extra info on it (for evaluation/documentation, the webpage isn't expected to be shared with user).
    - NOTE: fixes vs old step list: depth only (no height map from the model), no dense SLAM by default, planner is field + dynamic program (the Lagrangian side is the user model measuring afterwards), page draws over the video, local network not internet
    - NOTE: the component diagram still has a "Dense SLAM" box, needs redrawing!

![Component Diagram|700](diagrams/ComponentDiagram.drawio.png)

## Hardware and Data Capture
- camera: Neon scene camera, 1600 × 1200 at 30 fps, IMU (~100 Hz) and eye-tracking/gaze (200 Hz)
- Server: laptop with a Quadro T2000 GPU (4 GB), runs the whole pipeline + serves the webpage
- types of recordings + what we use them for:
  - **Companion Recording:** phone app records everything itself (video + IMU + gaze + calibration), sometimes saved to Server afterward
    - USED FOR: our "raw data" archive, kept for future tests/dev/demos (example: presentation demo)
  - **Stream Capture:** Server saves exactly what glasses send over wifi (compressed video packets + IMU + gaze + calibration). No decoding while recording! This is what we used to get "real" samples that included network behavior, and then use it for test/dev of our decoding afterward
    - USED FOR: what the Server actually receives, so replaying it tests everything (decoder, timing, IMU matching) simulating live behavior. Replays lost packets but not network delay. Used and then archived
  - **Frame Log (`--record-to`):** Server saves depth frames coming out of depth model
    - USED FOR: faster testing for all of the math AFTER the depth model (logs let us skip straight to scene + planner). Can't test anything before depth model. Used and then archived
- replay: capture → exact same live route and Server treats it like glasses are on (demos + live-route tests ran this way)

### Neon Companion
The [Neon Companion](https://pupil-labs.com/products/neon/software) phone app runs on the phone the glasses are connected to and is used to read out and record the data from the camera and eye tracker on the glasses. In the prototype, the phone broadcasts the live feed from the glasses over the local network so that this data can be analyzed on a different device. Apart from a video feed, the real-time API also includes information about the glasses' camera calibration and a clock offset used to straighten every frame and sync measurements respectively (see [Appendix D](d-neon-companion-api)). The live feed moreover includes gaze data from the eye tracker, which is used in the prototype both to optionally silence the alarm when the user looks at detected danger and by the path planner as the preferred direction to walk in.
## Point Cloud Generation
Point cloud generation refers to the repeated creation of a 3D map of points that are most crucial for navigation. It is generated primarily by analyzing the video feed from the glasses, but could alternatively simply be obtained from LiDAR sensors on select smartphones and smart glasses. This would nullify the need for the additional computation discussed in this section. The general flow for point cloud generation is given in below image.

![General flow point cloud generation](DataAnalyzationFlow.drawio.png)

The smart glasses are equipped with a wide-angle camera, which records a barrel distorted (fisheye) image. The first step is to straighten this image using the camera's calibration, to undo the effects of distortion (see [Appendix C](c-point-cloud-details)). 

This image is then piped into Depth Anything 3's metric model to determine a depth (distance from camera) for every point. But because this model treats each image as if it were captured on a 300 pixel focal length camera, the camera's real focal must first be scaled from the resolution at which the model runs:
$$
z = z_{raw}\cdot \frac{\bar{f}}{300} \qquad \bar{f}=\frac{f_{x}+f_{y}}{2}
$$
The scaled (actual) depth $z$ and depth returned by the model $z_{raw}$ are both in meters. The horizontal and vertical focal length values $f_{x}$ and $f_{y}$, which correspond to the dept map produced by the model, are measured in pixels. The average focal length is signified with $\bar{f}$. Every pixel is then sent back along its ray as far as the estimated dept specifies, to generate a 3D image of the visible environment.

To help orient and interpret the meaning of these points, the position of the floor is estimated for each frame. The floor is defined as a plane:
$$
\hat n \cdot \mathbf p + h = 0
$$
With $\hat{n}$ being the unit normal pointing up and $h$ the camera height, which is determined by incorporating data from the on-board Inertial Measurement Unit (see [Appendix A](c-point-cloud-details)). After this, the distance from the floor of every point $e_{i}$ is computed in meters using:
$$e_i = \hat n \cdot \mathbf p_i + h$$
Every point that is either too high or too low to collide with is filtered out. In the prototype, this leaves points $\{e_{i} \mid 0.2 < e_{i} < 2.0\}$.

Finally, the filtered point cloud is projected onto a 2D top-down representation of the environment by using the IMU (Inertial Measurement Unit) data to determine each point's lateral and forward position. 

## Point Cloud Cost
- **chop into cells:** surviving points flattened onto floor -> (lateral, forward) from walker -> 0.25 m squares ("cells"), 3 m each side, 6 m ahead. One cell = one obstacle from here on
  - cells with fewer than 2 points dropped (one point as likely depth noise as a thing)
  - each cell keeps closest point to walker + tallest point. 1.5 m or taller -> "wall"
  - cell id = row and column on grid, so **without a position the grid moves with the walker** and a post changes id every 0.25 m walked. With `--position`, grid pinned to the world and ids stick (see Part 1)
- **gap (S):** distance from edge of walker's footprint to cell's closest point:

  $$S = d - r, \qquad r = 0.35 \text{ m}$$

  - $d$ = distance from walker center to cell's closest point, $r$ = footprint radius (shoulder half-width plus margin), floored at 0 here, at $\epsilon = 0.06$ m in planner
    - NOTE: 0.35 m is a circle around walker, same every direction. Old notes called it "the forward-most part of the user"
- **wobble (N):** how much a cell's gap jumped around lately. SD of $S$ over last 0.5 s, needs at least 3 samples, never under 0.01 m:

  $$N = \max\!\left(\operatorname{std}\{S_{t}\}_{t \in \text{last } 0.5\text{ s}},\; 0.01\right)$$

  - steadily measured cell = small $N$, distance jumping half a meter between frames = big $N$. Paper calls this the noise, same units as $S$
  - **in practice N is almost always 0.01 m on the glasses**: at 2.5 to 3.5 planned frames/s the 0.5 s window holds 1 or 2 samples, under the 3 needed. Measured on about 0% of obstacles, position or not (see Part 1)
    - NOTE: this is why contact term below exists. Widening window to hold 3 samples still open
  - window is our choice (professor's slides use 1 s), 0.01 m floor is our choice so a steady cell still costs something
- **surprise per cell:** half the squared signal-to-noise, capped at $2 \times 10^4$:

  $$U = \tfrac12 \left(\frac{N}{S}\right)^2$$

  - $U$ = surprise (natural-log units, "nats", turned into bits once at end), $S$ floored at $\epsilon = 0.06$ m so touching isn't infinite. Form, cap and $\epsilon$ are the professor's
  - wall's $N$ tripled before this, so walls avoided further out than posts (our addition)
- what scene hands planner per cell: closest point (lateral, forward), $S$, $N$, wall flag, velocity (only with a position, planner doesn't read it yet)
- **wobble in System Overview:** $N$ above, frame-to-frame jitter of a cell's measured gap, not of the camera

## Planning
- **field:** cost-per-second for every future moment and sideways position
  - time: steps of $\Delta t = 0.1$ s out to a 3.8 s horizon (professor's values). Sideways: 0.1 m spacing, 3 m each way (our choice)
  - at step $k$, walker assumed $1.4 \cdot k\Delta t$ m further ahead (walking speed 1.4 m/s, our choice, professor's is 5 m/s for a cyclist) -> every cell that much nearer. Obstacles held still (motion prediction exists but off)
  - two terms added at each (step, position), both as rates per second:
    - **professor's collision surprise:** $\tfrac12 (N/S)^2$ from Point Cloud Cost, $S$ measured from candidate position
    - **our contact term:** how surprising for the body to touch the cell at all, whatever the noise. Professor's term alone lets a steadily measured post (small $N$) cost less to hit than to dodge, so plan walked into things

      $$U_{\text{contact}} = \frac{-\ln \Phi\!\left(S_b / \sigma\right)}{\tau_b}, \qquad S_b = d - 0.30, \qquad \sigma = \sqrt{N^2 + 0.10^2}, \qquad \tau_b = \frac{2 \cdot 0.30}{1.4}$$

      - $S_b$ = gap to body (half-width 0.30 m, narrower than 0.35 m footprint). $\sigma$ = spread of where body really is (cell's $N$ plus 0.10 m walker sway, an assumption, never measured). $\Phi$ = standard normal cumulative, so $\Phi(S_b/\sigma)$ = chance of getting past cleanly, $-\ln$ of it = surprise of that. $\tau_b$ = 0.43 s to walk past something. Capped at 50 per contact
  - within one cell take worst point, across cells add up (professor's rule). Scene sends one point per cell today anyway
  - **goal:** added only to last row: $\tfrac12 ((x - x_{\text{goal}}) / 1.5)^2$, $x_{\text{goal}}$ = 0 (straight ahead) or where gaze lands on floor (`--goal gaze`). Tolerance 1.5 m = half a hallway. Only breaks ties, can't pull plan through anything
  - **previous-plan prior (our addition, on by default):** professor's dynamic program forgets everything between frames, so with something dead ahead the arrow flipped left/right 65 to 127 times a minute on our walks. Fix in the professor's own form: prior pulling toward where last frame's plan said we'd be by now

    $$U_{\text{prior}} = \tfrac12 \frac{(x - x_{\text{prev}})^2}{0.25^2 + 0.05\,\Delta T}$$

    - $x_{\text{prev}}$ = last plan's position at this row, slid forward by $1.4 \cdot \Delta T$ (we assume walker followed it). $\Delta T$ = seconds since that plan. Spread 0.25 m widening with age. Only on first 1 s of rows (later rows free to replan). Dropped after a 3 s gap or when goal moves sideways more than 1.5 m. Kept in walker's frame, so no position needed
    - after the sweep: flips 0 to 1.3 a minute instead of 65 to 127
- **dynamic program:** cheapest sideways path through field (Hamiltonian planning, professor's, written from the slides)
  - each step costs field at position reached + sideways effort $\tfrac12 w v_x^2$, both $\times\, \Delta t$. $v_x$ = sideways speed, max 1 m/s (one 0.1 m cell per step). $w = 6.5$ (our choice, professor's 0.055 made plan sidestep at full speed whenever anything was ahead)
  - cheapest cost to every (step, position) with back-pointers -> cheapest end position wins -> walk pointers back = path
  - cost of chosen path minus what prior charged (that's memory, not scene) -> bits, $\text{bits} = \text{nats} / \ln 2$. User model reads this

## Collision Detection
- separate from plan on purpose: reads obstacles, never the path -> still fires when plan is wrong
- **corridor:** cells strictly ahead, within 0.30 m (body half-width) either side. Furniture being walked past isn't in it, neither are sides of a doorway you fit through
  - NOTE: 0.30 m, not the 0.35 m footprint: at 0.35 + 0.15 a 0.9 m doorway alarmed on 93 % of frames, at 0.30 it's 36 %
- **time to contact:** nearest corridor cell's gap over walking speed, $\tau = S_{\min} / 1.4$. Walking speed, not measured closing rate: that's a two-frame difference, 1 cm depth jitter reads as 0.3 m/s. Cost: someone walking at you looks no more urgent than a wall
- **alarm:** $\tau < 0.7$ s (0.98 m at 1.4 m/s, our choice). Once up, holds 0.5 s on frame timestamps so it doesn't flicker (classroom walk: 108 state changes -> 66)
- **avoidance surprise (the $\tau$ form from Background):** sent with every plan, in bits:

  $$U_{\text{avoid}} = \tfrac12 \left(\frac{1\text{ s}}{\tau}\right)^2 \Big/ \ln 2$$

  - 0 with empty corridor, 0.72 bits at 1 s, 1.47 bits at the 0.7 s alarm point. Page's path goes from normal color to full red at 1.47 bits. **Only the alarm and the page read this form.** Planner's field never does
- **gaze silencing (switch, off by default):** a beep about a chair you just looked at says nothing new
  - looked = any gaze sample within 0.1 s of the frame lands within 2° of the object's outline in the picture (Neon accuracy 1.8°). Look remembered 2 s, slid toward walker at 1.4 m/s (what counts as "the object": Appendix E)
  - looked + switch on -> alarm still red on page, but not heard. Never changes path or alarm itself. No gaze -> counts as not looked, so fails to "heard"

## Overlay
- **stereo cue (decided on the Server, page just applies the numbers):**
  - steering: ear on far side from planned heading $\theta$ (the arrow, Appendix E) gets quieter, walker turns away from quiet ear. Gain = $e^{-\frac12 (\theta / \sigma_\theta)^2}$ with $\sigma_\theta = \arctan(1.5 / 4) = 20.6°$ (goal tolerance seen as an angle). 4° head wobble leaves it at 98 %, 35.5° sidestep limit at 22.5 %. Never under 10 % (silent ear reads as broken headphones)
  - danger: while alarm is heard, ear on danger's side drops to that 10 % floor. Dead ahead ducks both
- **what reaches the walker vs what's for us:** ear gains and alarm sound are the walker's. Arrow, red, rings, plan view, pictures and bits are evaluation/demo (walker isn't looking at a screen)

## Experimental Conditions
- all three parts run on recordings, through the exact live route (`--neon-replay` on a capture), on Server (Quadro T2000, 4 GB), otherwise idle. Same code path as a live walk, so frame rate and delays real, network delay isn't
- recordings (walking) we have (all on glasses, all indoors):
  - 2026-10-05: first live session, two 240 s classroom captures. (only the second works for real, first one had sensor issues)
  - 2026-10-08: walk A (214 s, between classroom desks), walk B (27 s)
  - 2026-10-08, demo room: hallway into classroom (194 s), up and down stairs (121 s), standing on the tiers (263 s)
  - 2026-10-09: demo walk (148 s, starts by a big window)
  - 2026-10-14: attention walk (44 s, scripted: stare at a chair, walk past it looking and not looking, stairs, toward a wall)
    - NOTE: none outdoors, none with pedestrians or other moving obstacles. maybe worth a re-record since he wants us to make a full demo vid anyway? (we know these work, but processing time still an issue)
- **part 1 (position):** attention walk 2026-10-14 (44 s, tiered classroom: chair, stairs, whiteboard wall). One run per setting: `--position` absent, odometry or dense_slam, at 336 or 280 px depth resolution. Measured: planned frames/s, arrival-to-plan delay, share of frames with a position, position error and 3 s drift against reference camera path rebuilt from the video by COLMAP (structure-from-motion software) (first 18.4 s, 8 m, scale from depth model, aligned by rotation + shift only), share of obstacles with measured N
  - NOTE: no ground truth exists, COLMAP is the reference. Report has to state this
- **part 2 (planning):** replays through the evaluation's `check_planner` on frame logs, planner settings swept. Numbers to report: arrow flips per minute, arrow at sidestep limit with something 3 to 5.32 m ahead (band), 90th pct plan disagreement, alarm share. Walks: classroom (2026-10-08 A), pixel_walk_3 and contact_walk_1 for the prior sweep (Pixel-era logs, say so)
  - NOTE: which runs go in the report still to pick. Config comments carry every number above with the walk it came from
- **part 3 (collision):** alarm share and state changes per walk with corridor/hold on vs off, gaze silencing on the scripted attention walk (looks at the chair vs walking past it not looking)
  - NOTE: the 2026-10-09 demo walk hasn't been watched/described yet, and no walk has pedestrians

## Literature
https://www.nature.com/articles/s41598-025-08475-4

(How we did it)

*If you did an experiment, write about the motivation, earlier experiments in literature and why they didn't show what you are showing, used apparatus, participants and methods, results (just plain results in tables with statistical tests described in text), and a discussion of the results in light of the prior literature.*

*If you did a system, follow the same procedure but instead of results you describe 3 scenarios in which the system could be used in the future. When you write the scenario, imagine you are a user using the system and describe all the little details that demonstrate how well your interactions were thought through and how it solves an existing usability problem.*

# Results

At the end of the development process of the product, the product needed to be presented to our peers. For this presentation a demonstration of the product was made, this demonstration had next to a live demonstration also a back-up video in which a user from the development team used the glasses while walking through a lecture room at Radboud university. The image below shows a frame from this back-up video in which we see a multitude of panels further explained below the image. 
<img width="1726" height="1079" alt="NMHCI_demo_screenshot" src="https://github.com/user-attachments/assets/06b57432-3772-48f0-af11-262eb068fab6" />
## Panel: Sound
The panel at the top left is the output panel. This is the visual demonstration of what the product feeds back to the user. The top part shows the left and right speaker of the music device of the user. When encountering an obstacle with high risk it will start emitting a beeping noise from that specific side. This is seen in the five green bars showing the direction the beeping is coming from. There are also options to turn off the sound and turn off the noise cancellation at that side.

## Panel: Heading
The panel at the bottom left is the direction the user should walk towards to follow the safe path. This is not part of the output the user sees but the direction is concluded from the calculated path. The direction heading is not used in the rest of the product at the moment but it gives an entry point for different implementations to take input from to improve user experience by improving the system of guiding the user over the safe path.

## Panel: Video
The central panel is the field of view from the user, recorded by the camera on the neon glasses. This video shows the lecture room environment. The video shows the rows of chairs on the right, a wall with glass on the left and the path between those. In the path is a big black bag placed that functions as an obstacle for the user. This video feed is the input for the product to apply all the earlier mentioned functionalities on.

## Panel: From above
The top right panel shows to determined point cloud. This shows the grid in which all the measured points are placed. Specifically, the yellow and pink dots are the point of interest where a worth mentioning amount of risk is determined. The panel also shows blue and green areas below and around the dots, these are the weighted averages of the points in the point cloud to determine which objects are of any risk. The risk goes from blue to green to orange to red in order of increasing risk. Finally there is a big red line with an arrow to show the safe path for the user, this path is the source for the Sound and Heading panels and is obtained from the Risk panel.

## Panel: Risk
The bottom right panel is the video feed overlayed with the point cloud from the view from above. The panel shows the high risk, dark red, areas such as the chairs on the right and the wall on the left. It determines the floor as completely safe, seen by the dark blue color, except for the black bag placed. The bag has a clear yellow which indicates it has some risk of encountering and would turn darker red when getting closer to it. Next to the bag is the safe path shown with two red lines. These lines are calculated here and then used in the panel from above to show where the path progresses.

Link to the demonstration video
Explain the demonstration video.

argue about the resutls

# Scenarios

1. You want to walk in a straight line but there is an object in the middle of the hallway.

2. The user is walking and another pedestrian is approaching from the front and on route for collision.

3. Visually impaired person will not see an object but the product will help the user avoid it by showing it in a way the person can perceive.

# Discussion

# Conclusion

# References
1. Rijniers, R. (2017). *Ogen op de weg ☺ Een vermindering van het smartphonegebruik op de N36 middels een gedragsinterventie langs de weg* [Radboud Universiteit]. https://theses.ubn.ru.nl/items/c8935b8d-8583-455a-97e4-90905e7adb13/full
2. Haas, G. (2023). *Towards Auditory Interaction: An Analysis of Computer-Based Auditory Interfaces in Three Settings* [Doctoral dissertation, Ulm University]. https://oparu.uni-ulm.de/server/api/core/bitstreams/c45354b6-409b-4444-81e5-d55b99acf3f1/content
3. *[Authors]* (2025). *Interactive Inference: A Neuromorphic Theory of Human-Computer Interaction*. https://arxiv.org/abs/2502.05935 

# Appendix

## A. From IMU to camera up
- **which way is up:** fused IMU orientation says how glasses are turned against gravity. IMU measures in a world with z up -> two fixed turns (from Pupil Labs' docs) move it onto our camera and our world (y up):

  $$R = R_x(-90°)\; R_{\text{IMU}}\; R_x(-102°), \qquad \hat u_{\text{cam}} = R^\top \begin{pmatrix} 0 \\ 1 \\ 0 \end{pmatrix}$$

  - $R_{\text{IMU}}$ = IMU orientation, $R_x(\theta)$ = turn by $\theta$ about x axis, $R$ = turns camera directions into world directions, $\hat u_{\text{cam}}$ = "up" as camera sees it
  - −102° = camera sits 90° plus a 12° downward tilt from the IMU. Both angles from the docs, nobody measured them on our glasses
  - each frame takes IMU reading closest to its capture time (within 50 ms). None -> up falls back to image up, $(0, -1, 0)$, so we assume the head is level
    - NOTE: only glasses' fused orientation is used. Raw accelerometer, gyroscope and magnetometer aren't (old notes listed all three)

## B. Finding the floor
- only points well below camera can be floor: drop $\delta_i = -\mathbf p_i \cdot \hat u_{\text{cam}}$ over 0.5 m, and at least 200 of them
- *with gravity (normal case on glasses):* drops -> histogram with 2 cm bins, smoothed over 3 bins -> each peak = a level surface (floor, desk tops, stair treads, tiers)
  - surface needs at least 200 points and at least 8% of candidates within 5 cm of it. Refit as a plane by least squares, may lean at most 8°
  - deepest surface wins, since everything else stands on it
  - except: recent floor (median of last 10) within 0.25 m of a nearer surface and not of the deepest (like looking down at a lower tier) -> nearer one wins, until the deeper one has been passed over 5 frames in a row
- *without gravity:* one plane fit by RANSAC (300 tries, 5 cm tolerance)
- either way, plane has to pass sanity check: leans at most 35° from up, camera sits more than 0.3 m and at most 2.2 m above it. Nothing passes -> keep last frame's floor
  - NOTE: no height map comes from the model. Heights come from the floor fit here. Old notes' "too high to care @2.2 m" mixed up two numbers: 2.2 m is the highest a camera can sit above a believable floor (find floor). The "too high to care" cut is 2.00 m

## C. Point cloud details
- OpenCV picks new camera matrix that crops away black border (`alpha = 0`), same 1600 × 1200 size out
- different focals per axis (636 vs 754 px) squashed the picture 16% sideways. Both set to the bigger one, $f = 754.4$ px, so pixels stay square (like the model was trained on)
- field of view: 103.4° wide -> $2 \arctan(800 / 754.4) \approx 93.4°$
- Neon Player's depth plugin does the same conversion: its saved maps measured 0.9343 times model's raw output vs 0.9353 predicted, so conversion checks out
  - NOTE: Neon Player only checked this depth conversion. Gaze was never checked with it (old notes said both)

$$X = \frac{(u - c_x)\, z}{f_x}, \qquad Y = \frac{(v - c_y)\, z}{f_y}, \qquad Z = z$$

- $(u, v)$ = pixel column and row, $(c_x, c_y)$ = picture center ("principal point"), $(X, Y, Z)$ = point in camera's frame (x right, y down, z forward, meters)
- only every 2nd pixel each way, only depths 0.1 m (closer = smudged lens) to 30 m (further = sky)
- thinned to one point per 5 cm cube (Open3D voxel downsample, only Open3D call in the scene)

## D. Neon Companion API
- **Discovery:** finds phone on network automatically, or by typed-in address (Radboud wifi blocks discovery)
- **Clock Offset:** used for timing delays from the moment of capture. Capture time on server's clock:

  $$t_{\text{server}} = t_{\text{Neon}} + \Delta$$

  - $t_{\text{Neon}}$ = glasses' timestamp, $\Delta$ = measured offset (median of many round trips)
- **IMU stream only serves ONE client at a time!** (other connections get empty readings, glasses only support one connection at a time)
- **Recording:** app can record on its own (Companion Recording above), server never starts/stops it
- Companion recording → capture conversion takes a few seconds, so recordings and captures replay the same way
- recordings don't store depth (frame logs do), recomputed from video every run (so a newer depth model applies to old recordings automatically)

## E. Planner and alarm details
- **arrow:** angle from where walker stands to where path is 1 s ahead (our choice of 1 s: reading only the first 0.1 s step gave just 3 possible angles, since one step moves at most one cell)

  $$\theta = \arctan\!\left(\frac{x_{1\text{s}} - x_0}{1.4 \cdot 1\text{ s}}\right)$$

- **scene information (bits, for evaluation):** how far what camera saw moved the plan vs a planner seeing nothing (goal + prior only).
  - **how it's computed:** Both fields -> distribution over positions 1 s ahead, $p \propto e^{-\text{cost}}$ -> how far the two differ (KL divergence), in bits. Empty corridor = 0 bits, not "unsure"
- **user model:** measures only. Avoidance episode opens when planner asks for a turn (alarm, or arrow past 3°), closes when planner and walker's actual heading (IMU yaw against 5 s running average) both back under 3°. Work = path cost at open − cost at close, in bits
  - **predicted turn time:** Also predicts how long turn takes: heading error $s$ shrinks as fast as it can without overshooting ("critically damped"), $s'' + (2/b)s' + s/b^2 = 0$, with $b = 0.25$ s/bit how quick this walker is (placeholder, never measured). Episodes go to `episodes.jsonl`, nothing in pipeline reads them
  - NOTE: the Lagrangian side of the professor's math is only here, measuring after the fact. Nothing on the live path uses it
- "danger" = corridor cell that'd be hit first + cells chained to it within 0.4 m on the floor (up to 0.6 m out), so a look at the chair's back counts, not just its feet
- time-to-contact floor under which it's always heard = 0 (set after the 2026-10-15 manual test, where a desk inside the footprint kept a 0.35 s floor sounding the whole run)

## F. Webpage and sounds
- every plan goes out as one JSON message: path offsets, arrow angle, alarm (and whether it's heard), avoidance bits, information bits, path cost bits, danger side (−1 left to +1 right), left and right ear gains. Same message to webpage and phone app socket (phone app not used on the glasses runs)
- **webpage (HTTPS, self-signed cert, needed because the browser's video decoder only runs on a secure origin):**
  - page decodes glasses' compressed video itself (second websocket, one access unit per frame), draws over it: arrow, red on alarm, gaze ring, cell rings
  - top-down plan view: field, path through it, obstacles. Path color goes to red by avoidance bits, fill opacity by information bits
  - depth picture (or "risk" = depth recolored by the field) as PNG, at most 10 a second, about 50 ms each, drawn on its own thread so it never delays a plan
  - sounds: steering tick every 0.7 s (220 Hz, about 8 dB under the alarm) while alarm's down, alarm sound while it's heard. Default "Pulse", a 330 Hz click whose rate goes from 250 ms at 0.7 s or more to contact down to 100 ms at 0.2 s, parking-sensor style. Three volumes 6 dB apart. Optional music bed under the ear gains
  - page can flip: depth/risk picture, live/recording side, gaze silencing on/off. Only things it sends back
  - no path for 1.5 s -> sound stops (a beep would be pointing at a frame that's gone)
- **also written per run:** `timing.jsonl` (per-frame capture-to-plan delays), `episodes.jsonl` (user model), optional frame log (`--record-to`)
