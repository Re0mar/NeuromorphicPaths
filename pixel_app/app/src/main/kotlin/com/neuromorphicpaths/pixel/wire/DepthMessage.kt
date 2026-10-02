package com.neuromorphicpaths.pixel.wire

/**
 * One depth frame, in the laptop's camera convention, ready to encode.
 *
 * Axes are the laptop's: x right, y down, z forward. ARCore's camera frame is y up and z backward,
 * and [com.neuromorphicpaths.pixel.ar.ArCoreToWire] does the conversion before anything reaches here,
 * so this class never has to know which device produced it.
 *
 * @property timestampSeconds the frame's own clock, in seconds
 * @property depthMillimeters row-major uint16 depth in millimeters, [rows] by [columns], no padding
 * @property intrinsics 3 by 3 camera matrix at the depth image's resolution, row major
 * @property orientationWxyz camera-to-world rotation as a unit quaternion, w first
 * @property positionXyz camera position in meters, or null when tracking is lost
 * @property groundPlane the floor in the camera frame, or null when none is known
 */
data class DepthMessage(
    val timestampSeconds: Double,
    val depthMillimeters: ShortArray,
    val rows: Int,
    val columns: Int,
    val intrinsics: DoubleArray,
    val orientationWxyz: DoubleArray,
    val positionXyz: DoubleArray?,
    val groundPlane: GroundPlane?,
) {
    init {
        require(rows > 0 && columns > 0) { "depth image must have positive size, got ${rows}x$columns" }
        require(depthMillimeters.size == rows * columns) {
            "depth has ${depthMillimeters.size} values for a ${rows}x$columns image"
        }
        require(intrinsics.size == 9) { "intrinsics must be 3 by 3, got ${intrinsics.size} values" }
        require(orientationWxyz.size == 4) { "orientation must be a quaternion, got ${orientationWxyz.size} values" }
        require(positionXyz == null || positionXyz.size == 3) { "position must have 3 values" }
    }

    /** True when the device knows where it is. The laptop branches on this, so it must be honest. */
    val hasPosition: Boolean get() = positionXyz != null
}

/**
 * A plane as the laptop writes it: normal . point + offset == 0 on the plane, normal pointing up.
 *
 * In the laptop's camera axes up is negative y, so a level floor under the camera has a normal
 * near (0, -1, 0) and a positive offset equal to the camera's height above it.
 */
data class GroundPlane(val normal: DoubleArray, val offsetMeters: Double) {
    init {
        require(normal.size == 3) { "normal must have 3 values" }
    }
}
