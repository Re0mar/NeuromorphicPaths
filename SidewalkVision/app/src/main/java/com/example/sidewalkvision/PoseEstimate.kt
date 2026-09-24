package com.example.sidewalkvision

import kotlin.math.abs
import kotlin.math.atan
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.math.tan

// Kotlin copy of analysis/src/neuromorphicpaths_analysis/geometry/pov.py. The constants match it,
// so the app and the offline analysis judge the same edges the same way. Change both together.

// The lower part of the frame is nearest the camera, where the path is most likely straight.
private const val MIN_IMAGE_Y = 0.5
private const val MIN_POINTS_PER_EDGE = 4
private const val MIN_SLOPE_DIFFERENCE = 1e-3

// Edge points come from the mask grid, so distances are judged in grid cells. Points further than
// this from the fitted line are stray cells, dropped before refitting.
private const val OUTLIER_DISTANCE_CELLS = 1.5
private const val OUTLIER_REFITS = 3

// Past this scatter, after dropping outliers, the edge is curved or blocked.
private const val MAX_RELIABLE_RMS_CELLS = 0.75

// A line through a handful of rows swings a lot once extended to the horizon.
private const val MIN_RELIABLE_POINTS_PER_EDGE = 12
private const val MAX_OUTLIER_SHARE = 0.4

// An edge within this of the mask region's side over its whole length is the detection box cutting
// the mask off, not the path. It fits a line perfectly, so the checks above would pass it.
private const val BOX_SIDE_TOLERANCE_CELLS = 1.0

// A typical Belgian or Dutch sidewalk. Height is reported for this width until a better one is known.
const val DEFAULT_PATH_WIDTH_METERS = 1.5

/**
 * Focal length as a fraction of the image's long side, so it holds for any resize or rotation of
 * the same sensor image. Focal length in pixels is this times the long side in pixels.
 */
data class CameraIntrinsics(val focalLengthOverLongSide: Double) {
    fun focalLengthPx(widthPx: Int, heightPx: Int): Double = focalLengthOverLongSide * max(widthPx, heightPx)
}

enum class PoseStatus(val description: String) {
    OK("ok"),
    TOO_FEW_POINTS("too few unclipped points on one edge"),
    PARALLEL_EDGES("edges are parallel in the image"),
    VANISHING_POINT_BELOW_PATH("edges meet below the path, not ahead of it"),
}

/** A straight line x = slope * y + offset through one edge, in frame pixels. */
data class EdgeLineFit(
    val slope: Double,
    val offset: Double,
    // Over the points kept after dropping outliers.
    val rmsPx: Double,
    val pointsUsed: Int,
    val pointsOffered: Int,
    val cellPx: Double,
    val topPx: Double,
    val bottomPx: Double,
    val onBoxSide: Boolean = false,
) {
    fun xAt(yPx: Double): Double = slope * yPx + offset

    val rmsCells: Double get() = rmsPx / cellPx

    val straight: Boolean
        get() = rmsPx <= MAX_RELIABLE_RMS_CELLS * cellPx &&
            pointsUsed >= MIN_RELIABLE_POINTS_PER_EDGE &&
            pointsUsed >= (1 - MAX_OUTLIER_SHARE) * pointsOffered &&
            !onBoxSide

    fun runsAlong(xPx: Double): Boolean {
        // The fit is a straight line, so its largest distance from a vertical line is at an end.
        val distance = max(abs(xAt(topPx) - xPx), abs(xAt(bottomPx) - xPx))
        return distance <= BOX_SIDE_TOLERANCE_CELLS * cellPx
    }
}

/**
 * Where the camera is and how it points, from the path's two edges. Angles in degrees. Pitch is
 * positive tilting down. Heading is positive when the path heads off to the right of where the
 * camera points. Position runs from 0 at the left edge to 1 at the right. Fields a missing input
 * can't give are null: pitch, heading and height need a focal length.
 */
data class PoseEstimate(
    val status: PoseStatus,
    val left: EdgeLineFit? = null,
    val right: EdgeLineFit? = null,
    val vanishingPointPx: Pair<Double, Double>? = null,
    val focalLengthPx: Double? = null,
    val pitchDegrees: Double? = null,
    val headingDegrees: Double? = null,
    val positionAcross: Double? = null,
    val cameraHeightMeters: Double? = null,
    val pathWidthMeters: Double? = null,
    val reliable: Boolean = false,
) {
    /** Why an OK estimate isn't reliable, or null if it is. */
    val unreliableReason: String?
        get() = when {
            reliable || status != PoseStatus.OK -> null
            left?.onBoxSide == true || right?.onBoxSide == true -> "an edge is the detection box's side"
            else -> "edge curved, blocked or too short"
        }
}

private fun fitEdge(points: List<Pair<Double, Double>>, cellPx: Double): EdgeLineFit? {
    if (points.size < MIN_POINTS_PER_EDGE) return null
    val ys = DoubleArray(points.size) { points[it].first }
    val xs = DoubleArray(points.size) { points[it].second }
    var kept = BooleanArray(points.size) { true }
    var slope = 0.0
    var offset = 0.0
    for (refit in 0..OUTLIER_REFITS) {
        var count = 0
        var sumY = 0.0
        var sumX = 0.0
        var sumYY = 0.0
        var sumXY = 0.0
        for (i in ys.indices) {
            if (!kept[i]) continue
            count++
            sumY += ys[i]
            sumX += xs[i]
            sumYY += ys[i] * ys[i]
            sumXY += xs[i] * ys[i]
        }
        val denominator = count * sumYY - sumY * sumY
        if (denominator == 0.0) return null
        slope = (count * sumXY - sumY * sumX) / denominator
        offset = (sumX - slope * sumY) / count

        val nextKept = BooleanArray(points.size) { abs(xs[it] - (slope * ys[it] + offset)) <= OUTLIER_DISTANCE_CELLS * cellPx }
        // Too few left to fit a line means the edge isn't straight. Keep the last fit and let the
        // reliability check say so.
        if (nextKept.count { it } < MIN_POINTS_PER_EDGE || nextKept.contentEquals(kept)) break
        kept = nextKept
    }

    var squares = 0.0
    var used = 0
    var top = Double.MAX_VALUE
    var bottom = -Double.MAX_VALUE
    for (i in ys.indices) {
        if (!kept[i]) continue
        val residual = xs[i] - (slope * ys[i] + offset)
        squares += residual * residual
        used++
        top = minOf(top, ys[i])
        bottom = maxOf(bottom, ys[i])
    }
    return EdgeLineFit(slope, offset, sqrt(squares / used), used, points.size, cellPx, top, bottom)
}

/**
 * Estimate the camera's pose from a detection's edge points.
 *
 * Each image row holding two edge points gives one left and one right point, the leftmost and
 * rightmost. Clipped points are skipped, since they are the frame's border, not the path's edge.
 *
 * @param frameWidthPx width of the image the normalized edge points refer to, in pixels.
 * @param cellFraction width of one mask-grid cell as a fraction of that image's width.
 * @param maskBounds left, top, right, bottom of the region the mask was allowed into, normalized.
 * @param intrinsics null when the camera is unknown, as for gallery images.
 */
fun estimatePose(
    edgePoints: List<SidewalkEdgePoint>,
    frameWidthPx: Int,
    frameHeightPx: Int,
    cellFraction: Double,
    maskBounds: FloatArray?,
    intrinsics: CameraIntrinsics?,
    pathWidthMeters: Double? = DEFAULT_PATH_WIDTH_METERS,
    cameraHeightMeters: Double? = null,
): PoseEstimate {
    val cellPx = cellFraction * frameWidthPx
    val rows = edgePoints
        .filter { it.position.y >= MIN_IMAGE_Y }
        .groupBy { it.position.y }
        .values
        .filter { it.size >= 2 }
    if (rows.isEmpty()) return PoseEstimate(PoseStatus.TOO_FEW_POINTS)

    val leftPoints = mutableListOf<Pair<Double, Double>>()
    val rightPoints = mutableListOf<Pair<Double, Double>>()
    for (row in rows) {
        val leftmost = row.minBy { it.position.x }
        val rightmost = row.maxBy { it.position.x }
        val yPx = leftmost.position.y.toDouble() * frameHeightPx
        if (!leftmost.clipped) leftPoints.add(yPx to leftmost.position.x.toDouble() * frameWidthPx)
        if (!rightmost.clipped) rightPoints.add(yPx to rightmost.position.x.toDouble() * frameWidthPx)
    }

    var left = fitEdge(leftPoints, cellPx)
    var right = fitEdge(rightPoints, cellPx)
    if (left == null || right == null) return PoseEstimate(PoseStatus.TOO_FEW_POINTS, left, right)
    if (maskBounds != null) {
        left = left.copy(onBoxSide = left.runsAlong(maskBounds[0].toDouble() * frameWidthPx))
        right = right.copy(onBoxSide = right.runsAlong(maskBounds[2].toDouble() * frameWidthPx))
    }

    val slopeDifference = right.slope - left.slope
    if (abs(slopeDifference) < MIN_SLOPE_DIFFERENCE) return PoseEstimate(PoseStatus.PARALLEL_EDGES, left, right)

    val vanishingY = (left.offset - right.offset) / slopeDifference
    val vanishingX = left.xAt(vanishingY)
    val highestRowPx = rows.minOf { it.first().position.y.toDouble() } * frameHeightPx
    // The path widens toward the camera, so the lines must meet above the rows they came from.
    if (vanishingY >= highestRowPx || slopeDifference < 0) {
        return PoseEstimate(PoseStatus.VANISHING_POINT_BELOW_PATH, left, right, vanishingX to vanishingY)
    }

    val reliable = left.straight && right.straight
    val focalLength = intrinsics?.focalLengthPx(frameWidthPx, frameHeightPx)
        ?: return PoseEstimate(
            status = PoseStatus.OK,
            left = left,
            right = right,
            vanishingPointPx = vanishingX to vanishingY,
            // Exact when either pitch or heading is zero, close when both are small.
            positionAcross = -left.slope / slopeDifference,
            reliable = reliable,
        )

    val pitch = atan((frameHeightPx / 2.0 - vanishingY) / focalLength)
    val heading = atan((vanishingX - frameWidthPx / 2.0) * cos(pitch) / focalLength)
    // Height per meter of path width. The same for every row, since both lines share the vanishing point.
    val heightPerWidth = cos(pitch) / (slopeDifference * cos(heading))
    return PoseEstimate(
        status = PoseStatus.OK,
        left = left,
        right = right,
        vanishingPointPx = vanishingX to vanishingY,
        focalLengthPx = focalLength,
        pitchDegrees = Math.toDegrees(pitch),
        headingDegrees = Math.toDegrees(heading),
        positionAcross = (-left.slope - tan(heading) * sin(pitch)) / slopeDifference,
        cameraHeightMeters = pathWidthMeters?.let { it * heightPerWidth },
        pathWidthMeters = cameraHeightMeters?.let { it / heightPerWidth },
        reliable = reliable,
    )
}

/**
 * Ground distance ahead of the camera to an image row, from the pose. Null without a reliable pose
 * with pitch and height, or for a row at or above the horizon.
 */
fun groundDistanceMeters(pose: PoseEstimate, rowPx: Double, frameHeightPx: Int): Double? {
    if (!pose.reliable) return null
    val pitchDegrees = pose.pitchDegrees ?: return null
    val focalLength = pose.focalLengthPx ?: return null
    val height = pose.cameraHeightMeters ?: return null
    val angleBelowHorizon = Math.toRadians(pitchDegrees) + atan((rowPx - frameHeightPx / 2.0) / focalLength)
    if (angleBelowHorizon <= 0.0) return null
    return height / tan(angleBelowHorizon)
}
