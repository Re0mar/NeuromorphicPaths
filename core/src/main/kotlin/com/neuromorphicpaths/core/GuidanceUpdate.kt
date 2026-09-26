package com.neuromorphicpaths.core

/** Everything one pass of the pipeline produced for one frame, handed to every display. */
data class GuidanceUpdate(
    val frame: Frame,
    val detections: List<Detection>,
    /** The boxed objects placed by the locator, followed by the structure samples from the scene in force. */
    val obstacles: List<Obstacle>,
    val guidance: Guidance,
    val walker: WalkerState,
    /** Wall-clock time the detector took on this frame. The one number that says whether the model fits the phone. */
    val detectorNanos: Long = 0L,
    /** The segmenter's map in force for this frame, which may be a few frames old. Null when no segmenter runs. */
    val sceneMap: SceneClassMap? = null,
    /** Wall-clock time the segmenter took, on the frames it ran on. Null on every other frame. */
    val segmenterNanos: Long? = null,
)
