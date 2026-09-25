package com.neuromorphicpaths.app

import com.neuromorphicpaths.core.CameraIntrinsics
import com.neuromorphicpaths.core.Detection
import com.neuromorphicpaths.core.NormalizedBox
import com.neuromorphicpaths.core.ObstacleClass

/**
 * Numbers the app needs that nobody has measured yet.
 *
 * Both placeholders feed the range estimate directly, so ranges are indicative until they are
 * replaced with a tape measure and the camera's own reported field of view.
 */
object AppDefaults {
    /** A phone held at chest height by an adult. A head-worn camera is nearer 1.6 m. */
    const val CAMERA_HEIGHT_METERS = 1.4

    /** A typical phone main camera. The real value comes from the camera characteristics. */
    val CAMERA_INTRINSICS = CameraIntrinsics(horizontalFovRadians = Math.toRadians(66.0))

    /** A chair ahead and slightly right, a person further left. For the scripted detector. */
    val SCRIPTED_DETECTIONS = listOf(
        Detection(ObstacleClass.CHAIR, confidence = 0.9, box = NormalizedBox(0.55, 0.55, 0.75, 0.85)),
        Detection(ObstacleClass.PERSON, confidence = 0.8, box = NormalizedBox(0.15, 0.35, 0.3, 0.7)),
    )
}
