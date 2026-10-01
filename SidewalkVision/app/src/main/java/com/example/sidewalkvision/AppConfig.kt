package com.example.sidewalkvision

/**
 * Centralized configuration object containing parameters for Geometry, Depth Inference,
 * Visual SLAM, Vertegaal Surprise Potential model, Lagrangian Trajectory Optimization,
 * and AR/Map Overlay UI rendering.
 */
object AppConfig {

    object CameraGeometry {
        const val CAMERA_HEIGHT_M = 1.2f
        const val VFOV_DEG = 45.0f
        const val FALLBACK_PITCH_DEG = 35.0f
        const val GRID_ROWS = 24
        const val GRID_COLS = 20
        const val SIGMA = 0.08f
        const val ANALYSIS_INTERVAL_MS = 333L

        const val GRID_NEAR_M = 1.0f
        const val GRID_FAR_M = 5.0f
        const val GRID_HALF_WIDTH_M = 1.0f

        // Unknown/unobserved cells outside FOV get low surprise so open space detours are favored
        const val UNKNOWN_CELL_SURPRISE = 0.1f

        // High multiplier for physical obstacles (chairs, walls, boxes, steps) rising above ground
        const val OBSTACLE_SURPRISE_WEIGHT = 6.0f
        // Set to 0.18m (18 cm) to filter out floor noise, carpet patterns, and depth model uncertainty
        const val OBSTACLE_ELEVATION_THRESHOLD_M = 0.18f

        const val REFERENCE_EMA_ALPHA = 0.2f
        const val REFERENCE_MIN_CELLS = 30
        const val CALIBRATION_BAND_FRACTION = 0.15f
    }

    object DepthModel {
        const val MODEL_INPUT_SIZE = 364
        const val MODEL_ASSET_NAME = "depth_model.onnx"
        val IMAGENET_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        val IMAGENET_STD = floatArrayOf(0.229f, 0.224f, 0.225f)

        const val SYNTH_W = 160
        const val SYNTH_H = 120

        const val PREFS_NAME = "surprise_track_prefs"
        const val PREF_DEPTH_SCALE = "depth_scale"
        const val DEFAULT_DEPTH_SCALE = 1.0f
        const val MIN_RAW_DEPTH = 1e-3f
    }

    object VisualSlam {
        const val TARGET_W = 160
        const val TARGET_H = 120
        const val GROUND_Y_FRACTION = 0.6f
        const val CORNER_STEP_PX = 16
        const val HARRIS_SCORE_THRESHOLD = 2000f
        const val MAX_KEYPOINTS = 40
        const val LK_WINDOW_RADIUS = 3
        const val LK_MIN_DETERMINANT = 1e-4f
        const val LK_MAX_FLOW_PX = 15f

        const val GROUND_FLOW_SCALE = 0.05f
        const val REL_FLOW_SCALE = 0.1f
        // High tolerance for moving objects (~1.6 km/h excess approaching speed) before flagging as threat
        const val MIN_EXCESS_SPEED = 0.45f
        const val MAX_LANDMARKS = 12
    }

    object VertegaalSurprise {
        const val MOTION_WEIGHT = 1.0f
        const val LANDMARK_SIGMA_M = 0.30f
    }

    object LagrangianPlanner {
        const val NUM_NODES = 18
        const val NUM_RELAX_STEPS = 30
        const val ALPHA_POTENTIAL = 0.08f       // Smooth avoidance push for real obstacles
        const val BETA_BENDING = 0.50f          // High bending penalty to enforce rounded, smooth curves
        const val STRAIGHT_INERTIA = 0.15f      // Strong attraction to keep path straight down clear hallways
        const val SYMMETRY_BREAK_REPELLER = 0.50f // Lateral push force to detour around centered obstacles
        const val OBSTACLE_DETOUR_THRESHOLD = 3.5f // Requires significant real obstacle before detouring
        const val SMOOTHING_PASSES = 3          // Post-processing Laplacian curve smoothing passes
        const val STEP_SIZE = 0.05f
        const val FINITE_DIFF_DELTA_X = 0.1f
        const val LATERAL_CLAMP_RATIO = 0.85f
        const val STEER_THRESHOLD_M = 0.20f      // Lateral threshold before calling STEER LEFT/RIGHT
    }

    object GuidanceUi {
        const val PATH_RIBBON_WIDTH_PX = 14f
        const val PATH_GLOW_WIDTH_PX = 28f
        const val ARROW_STROKE_WIDTH_PX = 6f
        const val WARNING_STROKE_WIDTH_PX = 5f

        const val TEXT_SIZE_TITLE = 42f
        const val TEXT_SIZE_SUB = 32f

        const val COLOR_PATH_CYAN = "#00E5FF"
        const val COLOR_PATH_GLOW = "#8000E5FF"
        const val COLOR_STEER_AHEAD = "#00E676"
        const val COLOR_STEER_LEFT = "#FF9100"
        const val COLOR_STEER_RIGHT = "#00E5FF"
        const val COLOR_HUD_YELLOW = "#FFCC00"
        const val COLOR_HUD_BG_SLATE = 200 // Alpha
    }
}
