package com.neuromorphicpaths.pixel.ar

import com.google.ar.core.Camera
import com.google.ar.core.Frame
import com.google.ar.core.Plane
import com.google.ar.core.Pose
import com.google.ar.core.TrackingState
import com.neuromorphicpaths.pixel.wire.DepthMessage
import com.neuromorphicpaths.pixel.wire.GroundPlane
import kotlin.math.sqrt

/**
 * Turns one ARCore frame into the laptop's [DepthMessage], including the change of axes.
 *
 * ARCore's camera frame is OpenGL's: x right, y up, z backward. The laptop's is OpenCV's: x right,
 * y down, z forward. The two differ by a half turn about x. Every vector and every rotation that
 * leaves this file has been through that turn, and nothing downstream knows ARCore existed.
 */
object ArCoreToWire {
    /** DEPTH16 keeps depth in the low 13 bits. If the top bits are ever set they are not depth. */
    private const val DEPTH_MILLIMETER_MASK = 0x1FFF
    private const val NANOSECONDS_PER_SECOND = 1.0e9

    /**
     * Build the message for this frame, or null when ARCore has no depth image yet.
     *
     * @param frame the frame `session.update()` returned
     * @param camera `frame.camera`
     * @param groundPlane the plane chosen as the floor, or null
     */
    fun convert(frame: Frame, camera: Camera, groundPlane: Plane?): DepthMessage? {
        val depth = DepthImageReader.read(frame) ?: return null

        // The depth image covers the view of the GPU camera texture, not the CPU image. The CPU
        // image is 4:3 and the texture 16:9 on the Pixel 8, and scaling the CPU intrinsics onto
        // the 16:9 depth image axis by axis gave fx a third larger than fy on the first run.
        val cameraIntrinsics = camera.textureIntrinsics
        val imageDimensions = cameraIntrinsics.imageDimensions
        val focal = cameraIntrinsics.focalLength
        val principal = cameraIntrinsics.principalPoint
        val scaleX = depth.columns.toDouble() / imageDimensions[0]
        val scaleY = depth.rows.toDouble() / imageDimensions[1]
        val intrinsics = doubleArrayOf(
            focal[0] * scaleX, 0.0, principal[0] * scaleX,
            0.0, focal[1] * scaleY, principal[1] * scaleY,
            0.0, 0.0, 1.0,
        )

        val tracking = camera.trackingState == TrackingState.TRACKING
        val pose = camera.pose
        val orientation = cameraToWorldWxyz(pose)
        // A pose while tracking is lost is a stale number, not a position. The laptop's whole
        // frame choice hangs on has_position being honest.
        val position = if (tracking) doubleArrayOf(pose.tx().toDouble(), pose.ty().toDouble(), pose.tz().toDouble()) else null

        return DepthMessage(
            timestampSeconds = frame.timestamp / NANOSECONDS_PER_SECOND,
            depthMillimeters = depth.millimeters,
            rows = depth.rows,
            columns = depth.columns,
            intrinsics = intrinsics,
            orientationWxyz = orientation,
            positionXyz = position,
            groundPlane = groundPlane?.let { planeInCameraFrame(it.centerPose, pose) },
        )
    }

    /**
     * The camera-to-world rotation for the laptop's camera axes, as (w, x, y, z).
     *
     * ARCore gives q for its own camera axes. Ours are a half turn about x away, so the rotation
     * taking our camera vectors to the world is q composed with that half turn, (0, 1, 0, 0).
     */
    fun cameraToWorldWxyz(pose: Pose): DoubleArray {
        val qw = pose.qw().toDouble()
        val qx = pose.qx().toDouble()
        val qy = pose.qy().toDouble()
        val qz = pose.qz().toDouble()
        // q * (0, 1, 0, 0)
        return normalized(doubleArrayOf(-qx, qw, qz, -qy))
    }

    /**
     * ARCore's plane, expressed in the laptop's camera frame with the laptop's convention.
     *
     * ARCore planes are horizontal with their local y axis up. The world normal is the plane
     * pose's y axis. Both the normal and a point on the plane are moved into the camera frame and
     * flipped into the laptop's axes, and the offset is chosen so normal . p + offset is zero on
     * the plane and positive above it.
     */
    fun planeInCameraFrame(planeCenterPose: Pose, cameraPose: Pose): GroundPlane {
        val worldToCamera = cameraPose.inverse()
        val normalWorld = planeCenterPose.rotateVector(floatArrayOf(0f, 1f, 0f))
        val normalCamera = worldToCamera.rotateVector(normalWorld)
        val centerCamera = worldToCamera.transformPoint(planeCenterPose.translation)

        val normal = flipAxes(normalCamera)
        val center = flipAxes(centerCamera)
        val offset = -(normal[0] * center[0] + normal[1] * center[1] + normal[2] * center[2])
        return GroundPlane(normal = normalized(normal), offsetMeters = offset)
    }

    /** OpenGL camera axes to OpenCV camera axes: keep x, negate y and z. */
    private fun flipAxes(v: FloatArray): DoubleArray = doubleArrayOf(v[0].toDouble(), -v[1].toDouble(), -v[2].toDouble())

    private fun normalized(v: DoubleArray): DoubleArray {
        var sum = 0.0
        for (value in v) sum += value * value
        val length = sqrt(sum)
        return if (length == 0.0) v else DoubleArray(v.size) { v[it] / length }
    }

    /** A depth image copied out of ARCore's buffer, row stride removed, confidence bits masked. */
    class DepthImage(val millimeters: ShortArray, val rows: Int, val columns: Int)

    object DepthImageReader {
        fun read(frame: Frame): DepthImage? {
            val image = try {
                frame.acquireDepthImage16Bits()
            } catch (notYet: com.google.ar.core.exceptions.NotYetAvailableException) {
                // ARCore needs a few frames of motion before the first depth image exists.
                return null
            }
            image.use {
                val plane = it.planes[0]
                return DepthImage(
                    repack(plane.buffer.order(java.nio.ByteOrder.nativeOrder()), it.height, it.width, plane.rowStride, plane.pixelStride),
                    it.height,
                    it.width,
                )
            }
        }

        /**
         * Copy a DEPTH16 buffer into a dense row-major array of millimeters.
         *
         * Rows may be padded and pixels may be strided. The wire format wants rows times columns
         * values and nothing else, so this walks the strides value by value. The top three bits of
         * each value are reserved by the format and are masked off, so a value with them set
         * never reaches the laptop as a 50 meter depth.
         */
        fun repack(buffer: java.nio.ByteBuffer, rows: Int, columns: Int, rowStride: Int, pixelStride: Int): ShortArray {
            require(rows > 0 && columns > 0) { "depth image must have positive size, got ${rows}x$columns" }
            require(pixelStride >= 2) { "DEPTH16 pixels are two bytes, got a pixel stride of $pixelStride" }
            require(rowStride >= columns * pixelStride) { "row stride $rowStride cannot hold $columns pixels of $pixelStride bytes" }
            val out = ShortArray(rows * columns)
            for (row in 0 until rows) {
                val rowStart = row * rowStride
                for (column in 0 until columns) {
                    val raw = buffer.getShort(rowStart + column * pixelStride).toInt() and 0xFFFF
                    out[row * columns + column] = (raw and DEPTH_MILLIMETER_MASK).toShort()
                }
            }
            return out
        }
    }
}
