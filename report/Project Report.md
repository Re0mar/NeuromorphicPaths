# THE PLAN
- Abstract
  
- Introduction (Daniël)
    - Explain the problem
    - General approach idea?




        
- Background (Kars)
    Why the problem?
    - Motivation
    - Explain Depth Map
       - Explain Surprise Equations
        
    
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
A 2017 thesis published at the Radboud University concluded that reminding drivers to pay attention to the road has a significant effect on subsequent accidents [1](#references). Many of these are preventable one-sided accidents caused by drivers, cyclists and pedestrians leaving or crossing the road by accident or at the wrong moment. Our leading theory is that using wearable computer vision found in commercial products like smart glasses, we could detect these dangerous situations and subsequently direct the users attention to the road to greatly decrease the likelihood for an incident to occur.

> Mention that the Neon glasses are monocular, have eye tracking cameras and do not have other sensor capabilities (like LiDAR capabilities)

*The background of a study is the [section of a research paper, thesis, or proposal](https://paperguide.ai/blog/how-to-write-a-research-paper/) that presents existing knowledge about the topic and explains why the current study is necessary. It typically appears at the beginning of the paper, either as a distinct section (common in theses and dissertations) or as the opening portion of the introduction (common in journal articles)*

# Motivation
We noticed that there are a lot of visually impaired people ([27% to be precise](https://www.who.int/news-room/fact-sheets/detail/blindness-and-visual-impairment)) who struggle with finding their way, there are some solutions to help those people but they are [heavy duty and cost a lot of resources](https://www.bemyeyes.com/business/blog/essential-accessibility-tools-for-visually-impaired/). Our goal is a more light-weight product that could assist the visually impaired. The system could later be adapted to specific visual conditions by changing how the computed safe route is presented, for example through different visual encodings, contrast settings, or non-visual modalities such as audio. In its current form, the output is a single guidance line that the user follows, which serves as a proof of concept. The long-term objective is to support visually impaired people in their daily lives by reducing the risk of collisions with obstacles, and thereby to increase their mobility and independence.

> Remove the topic of visual impairment
> 

# Materials
## Neon smart glasses
A pair of [Pupil Labs Neon smart glasses](https://pupil-labs.com/products/neon) were provided to us by the course coordinator and act as the accessory to provide a real-time stream of images to be analyzed. Included with this is a [Motorola Edge 40 Pro](https://mijnmoto.nl/product/motorola-edge-40-pro/) smartphone, which comes paired with the glasses and allows us to control, review and analyse the recordings. 

## Image analysis
Depth Anything 3 is a model that "predicts spatially consistent geometry from an arbitrary number of visual inputs, with or without known camera poses" [Source](https://depth-anything-3.github.io/). It is used to estimate both a depth map and a height map given an input of one or more subsequent images.

[RGB-D SLAM](https://www.open3d.org/html/tutorial/t_reconstruction_system/dense_slam.html) is a (dense) implementation of **S**imultaneous **L**ocalization **a**nd **M**apping included in the Open3D library. The depth and height maps generated by Dept Anything 3 are fed into SLAM, which is used in this project to build a 3D reconstruction of the environment and the movement of the camera through it.

# Methodology

## Pipeline Overview
> Diagram and such

## Camera Feed
> What kind of videos/how we record them (i.e. walking up to objects on the ground, walls, people etc.)
## Neon Companion
> Explain what the app does:
> 1. Video recording & Streaming
> 2. Gaze tracking data

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
1. Rijniers, R. (2017). _Ogen op de weg ☺ Een vermindering van het smartphonegebruik op de N36 middels een gedragsinterventie langs de weg_ [Radboud Universiteit]. https://theses.ubn.ru.nl/items/c8935b8d-8583-455a-97e4-90905e7adb13/full