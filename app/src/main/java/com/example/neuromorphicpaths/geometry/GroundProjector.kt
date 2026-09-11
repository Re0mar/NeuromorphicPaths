package com.example.neuromorphicpaths.geometry

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import kotlin.math.atan2
import kotlin.math.PI

class GroundProjector(private val homography: Mat) {

    /**
     * Projects an image-space point to the ground plane using the homography matrix.
     */
    fun project(imagePoint: Point): Point {
        val src = Mat(3, 1, CvType.CV_64FC1)
        src.put(0, 0, imagePoint.x)
        src.put(1, 0, imagePoint.y)
        src.put(2, 0, 1.0)

        val dst = Mat(3, 1, CvType.CV_64FC1)
        Core.gemm(homography, src, 1.0, Mat(), 0.0, dst)

        val w = dst.get(2, 0)?.get(0) ?: 1.0
        val x = (dst.get(0, 0)?.get(0) ?: 0.0) / w
        val y = (dst.get(1, 0)?.get(0) ?: 0.0) / w

        return Point(x, y)
    }

    /**
     * Computes the sidewalk position metrics.
     * @param leftEdge Image-space line for the left edge.
     * @param rightEdge Image-space line for the right edge.
     * @param imageHeight Height of the image to pick projection points.
     * @return SidewalkPosition containing offsets and heading.
     */
    fun computePosition(leftEdge: Line?, rightEdge: Line?, imageHeight: Int): SidewalkPosition {
        // Project points at the bottom of the visible ROI (close to the user)
        val projectionY = imageHeight * 0.9
        
        val leftOffset = if (leftEdge != null) {
            val xAtY = interpolateX(leftEdge, projectionY.toFloat())
            val p = project(Point(xAtY.toDouble(), projectionY))
            p.x.toFloat()
        } else -1.5f // Default assumption: 1.5m to the left

        val rightOffset = if (rightEdge != null) {
            val xAtY = interpolateX(rightEdge, projectionY.toFloat())
            val p = project(Point(xAtY.toDouble(), projectionY))
            p.x.toFloat()
        } else 1.5f // Default assumption: 1.5m to the right

        // Heading: angle of the centerline between bottom points and further points
        val lookaheadY = imageHeight * 0.6
        var heading = 0f
        
        if (leftEdge != null && rightEdge != null) {
            val pBottomL = project(Point(interpolateX(leftEdge, projectionY.toFloat()).toDouble(), projectionY))
            val pBottomR = project(Point(interpolateX(rightEdge, projectionY.toFloat()).toDouble(), projectionY))
            val pTopL = project(Point(interpolateX(leftEdge, lookaheadY.toFloat()).toDouble(), lookaheadY))
            val pTopR = project(Point(interpolateX(rightEdge, lookaheadY.toFloat()).toDouble(), lookaheadY))
            
            val centerlineBottomX = (pBottomL.x + pBottomR.x) / 2.0
            val centerlineBottomY = (pBottomL.y + pBottomR.y) / 2.0
            val centerlineTopX = (pTopL.x + pTopR.x) / 2.0
            val centerlineTopY = (pTopL.y + pTopR.y) / 2.0
            
            val dx = centerlineTopX - centerlineBottomX
            val dy = centerlineTopY - centerlineBottomY
            
            heading = (atan2(dx, dy) * 180.0 / PI).toFloat()
        }

        return SidewalkPosition(
            leftEdgeOffsetM = leftOffset,
            rightEdgeOffsetM = rightOffset,
            headingDeviationDeg = heading
        )
    }

    private fun interpolateX(line: Line, y: Float): Float {
        if (line.y1 == line.y2) return line.x1
        val t = (y - line.y1) / (line.y2 - line.y1)
        return line.x1 + t * (line.x2 - line.x1)
    }
}
