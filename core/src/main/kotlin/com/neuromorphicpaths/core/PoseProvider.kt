package com.neuromorphicpaths.core

/** Answers "where is the camera pointing right now" for a source that stamps frames with a pose. */
fun interface PoseProvider {
    fun currentPose(): CameraPose
}

/** The same pose for every frame. For replay without an orientation log, and for tests. */
class FixedPoseProvider(private val pose: CameraPose) : PoseProvider {
    override fun currentPose(): CameraPose = pose
}
