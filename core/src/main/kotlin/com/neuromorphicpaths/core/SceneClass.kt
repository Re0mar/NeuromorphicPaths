package com.neuromorphicpaths.core

/**
 * What one cell of a segmenter's map can be, as far as the field cares.
 *
 * A segmenter maps whatever labels its model produces onto this set once, at the boundary, the
 * way a detector maps its labels onto [ObstacleClass]. Four members are ground the walker can be
 * on, and [surface] names the [SurfaceClass] the field charges for each. Three rise out of the
 * ground and get walked around, and [structure] names the [ObstacleClass] each becomes. SKY and
 * OTHER are neither. OTHER covers what the detector already boxes, people, cars, trees, and
 * anything nobody has decided about, so it costs nothing and produces no obstacle.
 */
enum class SceneClass(val surface: SurfaceClass?, val structure: ObstacleClass?) {
    PAVEMENT(SurfaceClass.PAVEMENT, null),
    GRASS(SurfaceClass.GRASS, null),
    DIRT(SurfaceClass.DIRT, null),
    ROAD(SurfaceClass.ROAD, null),
    BUILDING(null, ObstacleClass.BUILDING),
    WALL(null, ObstacleClass.WALL),
    STAIRS(null, ObstacleClass.STAIRS),
    SKY(null, null),
    OTHER(null, null),
    ;

    /** True for ground the walker can stand on, whatever it costs. */
    val isGround: Boolean get() = surface != null

    /** True for something that rises out of the ground and is walked around. */
    val isStructure: Boolean get() = structure != null
}
