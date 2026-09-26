package com.neuromorphicpaths.core

/**
 * What one scene map says about the world around the walker, in the walker's frame.
 *
 * The surfaces answer the field's per-meter question about the ground on any heading. The
 * structures are the feet of walls, buildings and stairs, placed as obstacles so the field
 * steers around them the way it steers around a boxed object. A region contributes one sample
 * per clearance width along its foot, never one per cell, so a long wall does not outweigh a
 * person by sample count.
 */
class LocatedScene(
    val surfaces: GroundSurfaceMap,
    val structures: List<Obstacle>,
) {
    companion object {
        /** No map yet, or none in this pipeline: unknown ground everywhere and nothing to walk around. */
        val NOTHING = LocatedScene(GroundSurfaceMap.UNKNOWN_EVERYWHERE, emptyList())
    }
}

/**
 * Third math contract. A scene map in a frame becomes surfaces and structures around the walker.
 *
 * Pure, like [ObstacleLocator]. Reads the frame's pose and intrinsics to project cells through
 * the ground plane, never its pixels.
 */
interface SceneLocator {
    fun locate(map: SceneClassMap, frame: Frame): LocatedScene
}
