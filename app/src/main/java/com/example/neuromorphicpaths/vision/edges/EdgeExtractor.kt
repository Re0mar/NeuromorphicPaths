package com.example.neuromorphicpaths.vision.edges

import com.example.neuromorphicpaths.geometry.Line
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import kotlin.math.abs

class EdgeExtractor {

    fun extractEdges(mask: Mat): Pair<Line?, Line?> {
        val contours = mutableListOf<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(mask, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

        if (contours.isEmpty()) return Pair(null, null)

        // Find largest contour
        val largestContour = contours.maxByOrNull { Imgproc.contourArea(it) } ?: return Pair(null, null)

        val imageWidth = mask.width()
        val imageHeight = mask.height()
        val centerX = imageWidth / 2.0
        val bottomRoiY = imageHeight * 0.4 // Only consider points below this (bottom 60%)

        val leftPoints = mutableListOf<Point>()
        val rightPoints = mutableListOf<Point>()

        val points = largestContour.toArray()
        for (point in points) {
            if (point.y < bottomRoiY) continue

            if (point.x < centerX) {
                leftPoints.add(point)
            } else {
                rightPoints.add(point)
            }
        }

        val leftLine = fitLine(leftPoints, imageWidth, imageHeight)
        val rightLine = fitLine(rightPoints, imageWidth, imageHeight)

        return Pair(leftLine, rightLine)
    }

    private fun fitLine(points: List<Point>, width: Int, height: Int): Line? {
        if (points.size < 2) return null

        val matOfPoint2f = MatOfPoint2f()
        matOfPoint2f.fromList(points)
        val lineParams = Mat()
        Imgproc.fitLine(matOfPoint2f, lineParams, Imgproc.DIST_L2, 0.0, 0.01, 0.01)

        val vx = lineParams.get(0, 0)?.get(0) ?: 0.0
        val vy = lineParams.get(1, 0)?.get(0) ?: 0.0
        val x0 = lineParams.get(2, 0)?.get(0) ?: 0.0
        val y0 = lineParams.get(3, 0)?.get(0) ?: 0.0

        // Line equation: y - y0 = (vy/vx) * (x - x0)
        // x = x0 + (y - y0) * (vx/vy)
        
        if (abs(vy) < 0.001) return null

        val y1 = height.toFloat()
        val x1 = (x0 + (y1 - y0) * (vx / vy)).toFloat()

        val y2 = (height * 0.4).toFloat()
        val x2 = (x0 + (y2 - y0) * (vx / vy)).toFloat()

        return Line(x1, y1, x2, y2)
    }
}
