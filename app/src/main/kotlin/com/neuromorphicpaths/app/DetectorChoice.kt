package com.neuromorphicpaths.app

/** Which detector the next pipeline start uses. */
enum class DetectorChoice(val label: String) {
    NONE("None"),
    SCRIPTED("Scripted"),
    YOLO_WORLD("YOLO-World"),
}
