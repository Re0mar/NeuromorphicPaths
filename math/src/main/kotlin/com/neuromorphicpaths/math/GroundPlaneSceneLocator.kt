package com.neuromorphicpaths.math

import com.neuromorphicpaths.core.Detection
import com.neuromorphicpaths.core.Frame
import com.neuromorphicpaths.core.GroundSurfaceMap
import com.neuromorphicpaths.core.LocatedScene
import com.neuromorphicpaths.core.NormalizedBox
import com.neuromorphicpaths.core.Obstacle
import com.neuromorphicpaths.core.ObstacleClass
import com.neuromorphicpaths.core.SceneClassMap
import com.neuromorphicpaths.core.SceneLocator
import com.neuromorphicpaths.core.SurfaceClass
import kotlin.math.atan2
import kotlin.math.hypot

/**
 * Projects a scene map through the ground plane into surfaces and structure samples.
 *
 * Surfaces go the other way round from the box locator: the field asks about a point on the
 * ground, the point is projected back into the frame, and the cell it lands in answers. Ground
 * outside the frame, above the horizon, or under something that is not ground is unknown, so
 * the field charges nothing for it and a structure is costed once, as an obstacle, rather than
 * again as a surface.
 *
 * Structures come from their feet. A cell of wall, building or stairs with ground in one of its
 * four neighbors is where the structure meets the ground, and the midpoint of that shared edge
 * is projected to a ground point. Those points are then thinned to one per [sampleSpacingMeters]
 * of the same class, nearest first, so a wall that spans fifty cells contributes a handful of
 * obstacles a clearance apart rather than fifty. Each sample becomes an obstacle with the
 * structure's class, full confidence, no track and no closing speed, so the field treats it as
 * a standing thing at the walker's own speed.
 */
class GroundPlaneSceneLocator(
    private val sampleSpacingMeters: Double = DEFAULT_SAMPLE_SPACING_METERS,
    private val maxRangeMeters: Double = DEFAULT_MAX_RANGE_METERS,
) : SceneLocator {

    init {
        require(sampleSpacingMeters > 0.0) { "sampleSpacingMeters must be positive" }
        require(maxRangeMeters > 0.0) { "maxRangeMeters must be positive" }
    }

    override fun locate(map: SceneClassMap, frame: Frame): LocatedScene {
        val projection = GroundPlaneProjection(frame.pose, frame.intrinsics, frame.width, frame.height)
        return LocatedScene(
            surfaces = surfacesOf(map, projection),
            structures = structuresOf(map, projection),
        )
    }

    private fun surfacesOf(map: SceneClassMap, projection: GroundPlaneProjection): GroundSurfaceMap =
        GroundSurfaceMap { forwardMeters, rightMeters ->
            val point = projection.frameAt(GroundPoint(forwardMeters, rightMeters))
            point?.let { map.classAtFraction(it.x, it.y).surface } ?: SurfaceClass.UNKNOWN
        }

    private fun structuresOf(map: SceneClassMap, projection: GroundPlaneProjection): List<Obstacle> {
        val cellWidth = 1.0 / map.width
        val cellHeight = 1.0 / map.height
        val samples = mutableListOf<StructureSample>()
        for (row in 0 until map.height) {
            for (column in 0 until map.width) {
                val structure = map.classAt(column, row).structure ?: continue
                val centerX = (column + HALF) * cellWidth
                val centerY = (row + HALF) * cellHeight
                for (neighbor in NEIGHBORS) {
                    val neighborColumn = column + neighbor.columnStep
                    val neighborRow = row + neighbor.rowStep
                    if (neighborColumn !in 0 until map.width || neighborRow !in 0 until map.height) continue
                    if (!map.classAt(neighborColumn, neighborRow).isGround) continue
                    val edge = FramePoint(
                        x = centerX + neighbor.columnStep * HALF * cellWidth,
                        y = centerY + neighbor.rowStep * HALF * cellHeight,
                    )
                    val ground = projection.groundAt(edge) ?: continue
                    val range = hypot(ground.forwardMeters, ground.rightMeters)
                    if (range > maxRangeMeters) continue
                    samples += StructureSample(structure, ground, range, edge, cellWidth, cellHeight)
                }
            }
        }
        return thin(samples).map { it.toObstacle() }
    }

    /** Nearest first, keeping a sample only when no kept sample of its class is within the spacing. */
    private fun thin(samples: List<StructureSample>): List<StructureSample> {
        val kept = mutableListOf<StructureSample>()
        for (sample in samples.sortedBy { it.rangeMeters }) {
            val crowded = kept.any { other ->
                other.structure == sample.structure &&
                    hypot(
                        other.ground.forwardMeters - sample.ground.forwardMeters,
                        other.ground.rightMeters - sample.ground.rightMeters,
                    ) < sampleSpacingMeters
            }
            if (!crowded) kept += sample
        }
        return kept
    }

    private class StructureSample(
        val structure: ObstacleClass,
        val ground: GroundPoint,
        val rangeMeters: Double,
        val edge: FramePoint,
        val cellWidth: Double,
        val cellHeight: Double,
    ) {
        /** One cell's worth of box around the foot point, so a display has something to draw. */
        fun toObstacle(): Obstacle = Obstacle(
            detection = Detection(
                obstacleClass = structure,
                confidence = 1.0,
                box = NormalizedBox(
                    left = (edge.x - HALF * cellWidth).coerceIn(0.0, 1.0),
                    top = (edge.y - HALF * cellHeight).coerceIn(0.0, 1.0),
                    right = (edge.x + HALF * cellWidth).coerceIn(0.0, 1.0),
                    bottom = (edge.y + HALF * cellHeight).coerceIn(0.0, 1.0),
                ),
                trackId = null,
            ),
            bearingRadians = atan2(ground.rightMeters, ground.forwardMeters),
            rangeMeters = rangeMeters,
            closingSpeedMetersPerSecond = null,
        )
    }

    private class Neighbor(val columnStep: Int, val rowStep: Int)

    companion object {
        /** One sample per clearance width, the smallest clearance any class gets. */
        const val DEFAULT_SAMPLE_SPACING_METERS = 0.5

        /** Beyond this a foot point is several cells past where the map can place anything, and the field would not care. */
        const val DEFAULT_MAX_RANGE_METERS = 20.0

        private const val HALF = 0.5
        private val NEIGHBORS = listOf(Neighbor(0, 1), Neighbor(0, -1), Neighbor(1, 0), Neighbor(-1, 0))
    }
}
