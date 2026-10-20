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
2. **As time to impact.** When the uncertainty is one second of movement, the equation becomes $U=\frac{1}{2}\left( \frac{1s}{\tau} \right)^2$, where $\tau$ is the time until impact. This form is used when evaluating a route: an obstacle that will be reached soon is much more surprising than one far away, and the surprise grows quadratically as $\tau$ shrinks.

Both forms express how reliably the user can predict the distance to an obstacle. The exact definitions of $N$ and the cell grid are given in the [Methodology](#methodology).

> **CHECK:** I wrote the second use as "surprise with $\tau$ when evaluating the route".


## Depth Estimation

Depth Anything 3 is a model that "predicts spatially consistent geometry from an arbitrary number of visual inputs, with or without known camera poses" ([source](https://depth-anything-3.github.io/)). We use the metric large variant, which turns a single frame from the glasses into a depth map in meters. This is important for our application, since the surprise equations above require real distances rather than relative depth. The model only outputs depth, so the ground and the height of points have to be derived afterwards from this depth map and the IMU data of the glasses.

## Path Planning

> **CHECK:** The plan lists "Hamiltonian path routing", but the Methodology describes the planner as a cost field plus dynamic programming, with the Lagrangian side as the user model. Can someone please confirm??

Once every cell of the environment has a surprise value, the next question is which route through them is best. Following the idea that movement is the relaxation along a surprise potential [[3]](#references), the safest route is the one that accumulates the least surprise over the next few seconds. This is computed as a minimum-cost path through the cost field, which is then reduced to a single heading for the user. The cost of following the resulting path, expressed in bits of surprise, can also be measured afterwards by the user model, which only observes and does not steer anything.

## 3D Reconstruction

[Dense SLAM](https://www.open3d.org/html/tutorial/t_reconstruction_system/dense_slam.html) (**S**imultaneous **L**ocalization **a**nd **M**apping), as implemented in the Open3D library, builds a 3D reconstruction of the environment from RGB-D input while tracking the movement of the camera through it. Because the surprise of a cell depends on how it changes over previous frames, knowing the movement of the camera is useful in principle. In the current prototype this is an optional component that is disabled by default (see the [Methodology](#methodology)).

# Materials
## Neon smart glasses
A pair of [Pupil Labs Neon smart glasses](https://pupil-labs.com/products/neon) were provided to us by the course coordinator and act as the accessory to provide a real-time stream of images to be analyzed. Included with this is a [Motorola Edge 40 Pro](https://mijnmoto.nl/product/motorola-edge-40-pro/) smartphone, which comes paired with the glasses and allows us to control, review and analyse the recordings. 

## Image analysis
Depth Anything 3 is a model that "predicts spatially consistent geometry from an arbitrary number of visual inputs, with or without known camera poses" [Source](https://depth-anything-3.github.io/). It is used to estimate both a depth map and a height map given an input of one or more subsequent images.

[RGB-D SLAM](https://www.open3d.org/html/tutorial/t_reconstruction_system/dense_slam.html) is a (dense) implementation of **S**imultaneous **L**ocalization **a**nd **M**apping included in the Open3D library. The depth and height maps generated by Dept Anything 3 are fed into SLAM, which is used in this project to build a 3D reconstruction of the environment and the movement of the camera through it.

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
- Companion recording → capture conversion takes a few seconds, so recordings and captures replay the same way
- replay: capture → exact same live route and Server treats it like glasses are on (demos + live-route tests ran this way)
- recordings don't store depth (frame logs do), recomputed from video every run (so a newer depth model applies to old recordings automatically)
- recordings (walking) we have (all on glasses, all indoors):
  - 2026-10-05: first live session, two 240 s classroom captures. (only the second works for real, first one had sensor issues)
  - 2026-10-08: walk A (214 s, between classroom desks), walk B (27 s)
  - 2026-10-08, demo room: hallway into classroom (194 s), up and down stairs (121 s), standing on the tiers (263 s)
  - 2026-10-09: demo walk (148 s, starts by a big window)
  - 2026-10-14: attention walk (44 s, scripted: stare at a chair, walk past it looking and not looking, stairs, toward a wall)
    - NOTE: none outdoors, none with pedestrians or other moving obstacles. maybe worth a re-record since he wants us to make a full demo vid anyway? (we know these work, but processing time still an issue)

### Neon Companion
- Neon-paired phone app = only thing server talks to (glasses never connect to server directly)
- what server receives from the app (real-time API):
  - **Discovery:** finds phone on network automatically, or by typed-in address (Radboud wifi blocks discovery)
  - **Calibration:** glasses' camera calibration
    - used for straightening every frame
  - **Streams:** scene video + gaze + IMU. Video arrives compressed, server decodes
  - **Clock Offset ("Time Echo"):** gap between phone and server clocks
    - used for timing delays from the moment of capture. Capture time on server's clock:

      $$t_{\text{server}} = t_{\text{Neon}} + \Delta$$

      - $t_{\text{Neon}}$ = glasses' timestamp, $\Delta$ = measured offset (median of many round trips)
- **IMU stream only serves ONE client at a time!** (other connections get empty readings, glasses only support one connection at a time)
- **Recording:** app can record on its own (Companion Recording above), server never starts/stops it
- **Gaze**:
  - `--goal gaze`: planner "aims" where wearer is looking (default straight ahead)
  - gaze silencing: looking at the danger can quiet the alarm (switch, off by default)
  - gaze ring on webpage (evaluation only, not shared with user)

## Point Cloud Generation
> Each frame...
> Dept anything 3
> Video Feed -> Dept map

> Dept map + IMU data (accelerometer, gyroscope, magnetometer) -> Height map (floor detection + sky/too high to care detection (@2.2m))

> Mention that the methods used to infer gaze & dept from the recorded video and gaze metadata are validated by using the Neon Player app

![[DataAnalyzationFlow.drawio.png]]

## Point Cloud Cost
> Project a 2d grid of 25x25cm squares (called cells). 
> The noise and suprise are computed per cell (not per point) while taking into account the surprise values of each cell in previous frame.
> Compute signal-to-noise a.k.a. avoidance form ${\color{purple}{U}} = \tfrac12 (\Delta {\color{teal}{S}} / {\color{teal}{S}})^2$ (which is essentially the same as $U=\frac{1}{2}\left( \frac{1s}{\tau} \right)^2$) where U = surprise, $\tau$ is time till impact and 1s = 1sec. (which is essentially the same as $U=\frac{1}{2}\left( \frac{N}{S} \right)^2$ and $S= d-0.35$) where N = pixel wobble, S = gap (distance between the 'forward most part' of the user and the "top-down" closest point of the object) and d = distance (distance between person centre and closest point of the object).

> Wobble $N$ is essentially the SD of the difference in positition for a given point over the past 3 frames (0.5). Wobble is noise and therefore generates more surprise. Wobble is the lack of consistency for each cell over the past few frames.
## Overlay

## Experimental Conditions
> Pipeline

Our prototype linearly feeds input from the glasses through a pipeline which outputs an optimal path it likes the user to take.
1. A live feed (or prerecorded video) is taken from the glasses which is streamed by the phone over the internet.
2. A laptop taps into this life feed and uses Depth Anything 3 to estimate a dept and height map.
3. Depth and height maps are fed into Open3D dense SLAM to generate point clouds. 
4. Based on these point clouds, a Hamiltonian and subsequently a Lagrangian is computed.
5. Finally, the Lagrangian is overlayed on the original live feed and streamed back to the phone over the internet.

![Component Diagram|700](diagrams/ComponentDiagram.drawio.png)
## Literature
https://www.nature.com/articles/s41598-025-08475-4

(How we did it)

*If you did an experiment, write about the motivation, earlier experiments in literature and why they didn't show what you are showing, used apparatus, participants and methods, results (just plain results in tables with statistical tests described in text), and a discussion of the results in light of the prior literature.*

*If you did a system, follow the same procedure but instead of results you describe 3 scenarios in which the system could be used in the future. When you write the scenario, imagine you are a user using the system and describe all the little details that demonstrate how well your interactions were thought through and how it solves an existing usability problem.*

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