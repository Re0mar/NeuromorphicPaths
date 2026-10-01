# SidewalkVision: Surprise-Minimizing Path Planning for Pedestrians with Smart Glasses

Group 9: Glasses Traffic

# Abstract

Walking in busy urban environments requires continuous attention to static obstacles and approaching traffic. We present SidewalkVision, a prototype assistive navigation system for the Pupil Labs Neon smart glasses and a paired Android smartphone. The system estimates metric depth from the monocular scene camera, converts it into a top-down elevation grid, and tracks motion with sparse optical flow to detect looming threats. These cues are combined with the wearer's gaze into a single "surprise" potential field. A discrete Lagrangian path planner then finds the walking route of least surprise, and the route is shown as an augmented-reality overlay. We describe the architecture, the mathematical model, and three usage scenarios. [TODO: add one sentence on evaluation or future work once decided.]

# Introduction

Pedestrians, and especially people who are blind or visually impaired, must constantly decide where it is safe to step. Smart glasses with a forward camera and an eye tracker make it possible to observe both the scene and the wearer's attention. This project asks: *can we compute a safe route by treating unexpected scene elements as "surprise", and can we use the eye tracker to tell when the user has not noticed a danger?*

Our contributions are:

- A real-time pipeline that turns a monocular camera stream into a top-down obstacle and motion map.
- A surprise potential field combining static obstacles, looming motion, and gaze.
- A Lagrangian least-action path planner that produces a smooth route around that field.
- Three usage scenarios illustrating the interaction design.

# Background

[TODO: 1 paragraph each, with citations]

- **Monocular depth estimation:** Depth Anything V2 (Yang et al., 2024).
- **Visual SLAM and optical flow:** Harris corner detection (Harris & Stephens, 1988) and Lucas-Kanade optical flow (Lucas & Kanade, 1981). Note that our tracker is sparse, not a full dense SLAM system (see Discussion).
- **Surprise and attention:** Itti & Baldi (2009) on Bayesian surprise attracting human attention; Vertegaal's work on gaze and attention in human-computer interaction.
- **Assistive navigation for visually impaired users:** [TODO: find 3 to 5 prior systems, e.g. cane-mounted, smartphone-based, wearable].
- **Path planning with variational principles:** least-action formulations and potential-field planners.
- **Gaze as attention signal:** [TODO: Pupil Labs Neon papers; studies of inattention while walking].

# Motivation

[TODO: Short, concrete.] Existing aids either detect obstacles but ignore what the user already sees, or rely on dense cloud processing. A system that knows both the scene and the wearer's gaze can warn only when needed, reducing alarm fatigue. Running on a phone keeps latency low and keeps video data on the device.

# Materials

## Neon smart glasses

A pair of [Pupil Labs Neon smart glasses](https://pupil-labs.com/products/neon) were provided to us by the course coordinator and act as the accessory to provide a realtime stream of images to be analyzed. Included with this is a [Motorola Edge 40 Pro](https://mijnmoto.nl/product/motorola-edge-40-pro/) smartphone, which comes paired with the glasses and allows us to control, review and analyse the recordings.

The Neon provides a scene camera and an eye tracker. The Companion app exposes normalized gaze coordinates over a local REST API, which our application polls.

## Image analyzation



# Methodology

The pipeline runs about every 333 ms for depth and motion analysis, while the UI and gaze stream run at 30 FPS.

```
Camera frame + IMU pitch -> monocular depth -> ground-plane elevation
-> top-down grid -> optical flow (looming threats) + gaze
-> surprise potential field -> Lagrangian path -> AR overlay
```

## 1. Depth and ground-plane elevation

Frames are resized to 364 × 364, normalized with ImageNet statistics, and passed to the depth network. The raw inverse depth is converted to metric depth:

$$Z_{\text{metric}} = \frac{\text{depthScale}}{\max(d_{\text{raw}}, 10^{-3})}$$

The scale is self-calibrated from the median depth of the floor in the lowest 15% of the image. With pitch $\theta_{\text{pitch}} = \arcsin(R_{2,2})$ from the IMU, the expected ground distance at image row $r$ is

$$\alpha_r = \theta_{\text{pitch}} + \arctan\!\left(\frac{r+0.5-H/2}{f_y}\right), \qquad Z_{\text{ground}}(r) = \frac{H_{\text{cam}}}{\sin\alpha_r}$$

and the elevation above the floor is $E(r,c) = (Z_{\text{ground}}(r) - Z_{\text{metric}}(r,c))\sin\alpha_r$. A homography built from the camera intrinsics, pitch and height maps $E$ onto the top-down grid, with bilinear sampling.

## 2. Motion and looming threats

Harris corners (up to 40, on 160 × 120 grayscale frames) are tracked with Lucas-Kanade optical flow in a 7 × 7 window. Flow in the lower 40% of the image estimates ego-motion. Subtracting it from the upper keypoints gives relative velocity; points approaching faster than 0.45 m/s are flagged as looming threats.

## 3. Surprise potential field

$$S(x,z) = S_{\text{static}} + S_{\text{motion}} + S_{\text{gaze}}$$

- **Static:** floor deviations above 0.18 m receive a quadratic penalty (weight 6.0); unobserved cells get a low default of 0.1.
- **Motion:** each looming landmark $k$ adds a Gaussian scaled by excess speed: $\sum_k v_{\text{excess},k}^2 \exp\!\big(-\tfrac{\Delta x_k^2+\Delta z_k^2}{2\sigma_m^2}\big)$.
- **Gaze:** a Gaussian focus region, $1.2\exp(-d^2/12)$, centered on the smoothed gaze point (exponential moving average, $\alpha = 0.25$).

## 4. Lagrangian path planning

The route is the path minimizing the discrete action

$$S = \int \left(\tfrac{1}{2}\|\dot{\mathbf{x}}\|^2 + \alpha\,\text{Surprise}(\mathbf{x})\right)dt$$

An 18-node path from 1 to 5 m is relaxed for 30 gradient-descent passes with potential weight $\alpha = 0.08$, bending penalty $\beta = 0.50$ and straight inertia $\gamma = 0.15$. A symmetry-breaking term deflects the path when an obstacle is dead ahead, and three passes of Laplacian smoothing remove kinks. A lateral displacement above 0.20 m produces the instruction "STEER LEFT" or "STEER RIGHT"; otherwise "STRAIGHT AHEAD".

## 5. Visual output

An AR overlay draws the path as a ribbon on the camera view, red rings on looming threats, and a gaze reticle. A second view shows the top-down surprise map as a coolwarm heat map with flow vectors and the planned path.

**Figure 1 (TODO):** system pipeline diagram.
**Figure 2 (TODO):** screenshot of the AR overlay next to the top-down surprise map.

# Scenarios

[TODO: Rewrite each in first person, with fine-grained interaction detail. Sketches below.]

**Scenario 1: Crossing a bike path.** Sam wears the glasses on the way to the station. A cyclist approaches from the left while Sam looks at a shop window. The looming flow crosses the 0.45 m/s threshold, a red ring appears around the cyclist, and the path ribbon bends away. [TODO: describe the notification, e.g. audio or haptic, and Sam's reaction.]

**Scenario 2: Uneven pavement and temporary obstacles.** Maya, who has low vision, walks down a street with roadworks. Raised tiles and a sign stand more than 18 cm above the floor plane. The planner routes around them, and the instruction changes smoothly from "STRAIGHT" to "STEER RIGHT" and back.

**Scenario 3: Inattention warning.** [TODO: this depends on the gaze-based inattention feature, which is a stated aim but not yet in the technical description.] Jan is walking while reading messages. His gaze stays off the path while an obstacle with high surprise lies ahead. The system notices that the gaze heatmap does not overlap the high-surprise cells and issues a short warning.

# Discussion

- **Relation to prior work:** [TODO: compare with assistive navigation systems from Background.]
- **Limitations:** depth is monocular and scale-calibrated with an assumed 1.2 m camera height, which is not exact for glasses, whose height and pitch vary with head movement; sparse optical flow is noisy; many parameters are hand-tuned; no user study has been done.
- **Terminology:** the tracker is Harris + Lucas-Kanade (sparse) rather than a full dense SLAM system, and the paper should describe it that way, or call it "visual odometry-style motion tracking".
- **Future work:** gaze-based inattention detection, user studies with blind and low-vision participants, audio/haptic output, learned weights instead of hand-tuned ones, comparison with LiDAR ground truth.

# Conclusion

We presented SidewalkVision, a smart-glasses prototype that fuses monocular depth, motion cues and gaze into a surprise field and plans the route of least surprise with a Lagrangian optimizer. [TODO: finish after evaluation.]

# References

(APA style; verify details before submitting)

- Harris, C., & Stephens, M. (1988). A combined corner and edge detector. *Proceedings of the Alvey Vision Conference*, 147-151.
- Itti, L., & Baldi, P. (2009). Bayesian surprise attracts human attention. *Vision Research, 49*(10), 1295-1306.
- Lucas, B. D., & Kanade, T. (1981). An iterative image registration technique with an application to stereo vision. *Proceedings of IJCAI*, 674-679.
- Yang, L., Kang, B., Huang, Z., Zhao, Z., Xu, X., Feng, J., & Zhao, H. (2024). Depth Anything V2. *Advances in Neural Information Processing Systems*.
- Pupil Labs. (n.d.). *Neon*. https://pupil-labs.com/products/neon
- [TODO: Vertegaal, assistive navigation systems, Neon accuracy reports]