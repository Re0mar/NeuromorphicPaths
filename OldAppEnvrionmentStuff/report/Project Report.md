
# Abstract
# Introduction
In recent years, more and more preventable traffic accidents keep happening on Dutch roads because of distracted driving caused by smartphones [(Rijniers, 2017)](#references). In light of this, our group has set out to research a potential aid that utilizes surprise to get the user to pay attention to the road when they encounter a potentially dangerous situation. To achieve this, we theorized that we could employ smart glasses utilizing computer vision to analyze the environment and infer the relative risk of potential collision using Fitts' law.

# Background
A 2017 thesis published at the Radboud University concluded that reminding drivers to pay attention to the road has a significant effect on subsequent accidents [[1](#references)]. Many of these are preventable one-sided accidents caused by drivers, cyclists and pedestrians leaving or crossing the road by accident or at the wrong moment. Our leading theory is that using wearable computer vision found in commercial products like smart glasses, we could detect these dangerous situations and subsequently direct the users attention to the road to greatly decrease the likelihood for an incident to occur.
# Motivation

# Materials
## Neon smart glasses
A pair of [Pupil Labs Neon smart glasses](https://pupil-labs.com/products/neon) were provided to us by the course coordinator and act as the accesory to provide a realtime stream of images to be analyzed. Included with this is a [Motorola Edge 40 Pro](https://mijnmoto.nl/product/motorola-edge-40-pro/) smartphone, which comes paired with the glasses and allows us to control, review and analyse the recordings. 

## Image analyzation
[YOLO](https://github.com/ultralytics/) is an open-source library that is used among other things to perform object detection in images. It is used in combination with a data set to optimize for detecting sidewalks, by training the model on [a relevant labeled data set](https://universe.roboflow.com/sidewalk/sidewalk-segmentation) that primarily contains urban sidewalks in Belgium. 

# Methodology

(How we did it)

*If you did an experiment, write about the motivation, earlier experiments in literature and why they didn't show what you are showing, used apparatus, participants and methods, results (just plain results in tables with statistical tests described in text), and a discussion of the results in light of the prior literature.*

*If you did a system, follow the same procedure but instead of results you describe 3 scenarios in which the system could be used in the future. When you write the scenario, imagine you are a user using the system and describe all the little details that demonstrate how well your interactions were thought through and how it solves an existing usability problem.*
# Scenarios

# Discussion

# Conclusion

# References
1. Rijniers, R. (2017). _Ogen op de weg ☺ Een vermindering van het smartphonegebruik op de N36 middels een gedragsinterventie langs de weg_ [Radboud Universiteit]. https://theses.ubn.ru.nl/items/c8935b8d-8583-455a-97e4-90905e7adb13/full