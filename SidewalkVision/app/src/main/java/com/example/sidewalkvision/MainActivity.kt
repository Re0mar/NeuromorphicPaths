package com.example.sidewalkvision

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.*
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import android.text.InputType
import android.util.AttributeSet
import android.view.Gravity
import android.view.View
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.io.ByteArrayOutputStream
import java.nio.FloatBuffer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.*
import kotlin.random.Random

/**
 * Data class for tracked visual SLAM landmarks and dynamic approaching objects.
 */
data class SlamLandmark(
    val id: Int,
    var x: Float, // Metric x (meters laterally, 0 = center)
    var z: Float, // Metric z (meters forward)
    var vx: Float, // Velocity x in m/s
    var vz: Float, // Velocity z in m/s
    var vTowards: Float, // Approaching velocity towards camera in m/s
    var excessApproachingSpeed: Float, // vTowards - vGroundTowards
    var confidence: Float
)

/**
 * Visual SLAM module for frame-to-frame feature tracking, optical flow estimation,
 * and dynamic approaching object identification.
 */
class VisualSlamTracker {
    private var lastGray: IntArray? = null
    private var lastW = 0
    private var lastH = 0
    private var lastTimeMs = 0L
    private var nextId = 1

    private val trackedLandmarks = mutableListOf<SlamLandmark>()

    fun processFrame(
        bitmap: Bitmap,
        pitchDeg: Float,
        aspect: Float,
        cameraHeight: Float = AppConfig.CameraGeometry.CAMERA_HEIGHT_M
    ): List<SlamLandmark> {
        val now = System.currentTimeMillis()
        val dt = if (lastTimeMs > 0) (now - lastTimeMs) / 1000f else 0.33f
        lastTimeMs = now

        val targetW = AppConfig.VisualSlam.TARGET_W
        val targetH = AppConfig.VisualSlam.TARGET_H
        val scaled = Bitmap.createScaledBitmap(bitmap, targetW, targetH, true)
        val pixels = IntArray(targetW * targetH)
        scaled.getPixels(pixels, 0, targetW, 0, 0, targetW, targetH)

        val gray = IntArray(targetW * targetH) { i ->
            val px = pixels[i]
            val r = (px shr 16) and 0xFF
            val g = (px shr 8) and 0xFF
            val b = px and 0xFF
            (0.299f * r + 0.587f * g + 0.114f * b).toInt()
        }

        val prevGray = lastGray
        lastGray = gray
        lastW = targetW
        lastH = targetH

        if (prevGray == null || prevGray.size != gray.size) {
            return trackedLandmarks.toList()
        }

        val keypoints = detectCorners(prevGray, targetW, targetH)

        var sumGroundFlowX = 0f
        var sumGroundFlowY = 0f
        var groundCount = 0

        val flowVectors = mutableListOf<Pair<PointF, PointF>>()

        for (kp in keypoints) {
            val flow = lucasKanadeFlow(prevGray, gray, targetW, targetH, kp.x.toInt(), kp.y.toInt())
            if (flow != null) {
                flowVectors.add(Pair(kp, flow))
                if (kp.y > targetH * AppConfig.VisualSlam.GROUND_Y_FRACTION) {
                    sumGroundFlowX += flow.x
                    sumGroundFlowY += flow.y
                    groundCount++
                }
            }
        }

        val groundFlowX = if (groundCount > 0) sumGroundFlowX / groundCount else 0f
        val groundFlowY = if (groundCount > 0) sumGroundFlowY / groundCount else 0f
        val groundVTowards = -groundFlowY * AppConfig.VisualSlam.GROUND_FLOW_SCALE / max(0.05f, dt)

        val updatedList = mutableListOf<SlamLandmark>()

        for ((kp, flow) in flowVectors) {
            val relFlowX = flow.x - groundFlowX
            val relFlowY = flow.y - groundFlowY

            val metricPos = screenToMetricGround(kp.x, kp.y, targetW, targetH, pitchDeg, aspect, cameraHeight)
            if (metricPos != null) {
                val (mx, mz) = metricPos
                val vx = relFlowX * AppConfig.VisualSlam.REL_FLOW_SCALE / max(0.05f, dt)
                val vz = -relFlowY * AppConfig.VisualSlam.REL_FLOW_SCALE / max(0.05f, dt)

                val dist = sqrt(mx * mx + mz * mz)
                val rHatX = -mx / max(0.1f, dist)
                val rHatZ = -mz / max(0.1f, dist)
                val vTowards = vx * rHatX + vz * rHatZ
                val excess = max(0f, vTowards - groundVTowards)

                if (excess > AppConfig.VisualSlam.MIN_EXCESS_SPEED) {
                    updatedList.add(
                        SlamLandmark(
                            id = nextId++,
                            x = mx,
                            z = mz,
                            vx = vx,
                            vz = vz,
                            vTowards = vTowards,
                            excessApproachingSpeed = excess,
                            confidence = 0.85f
                        )
                    )
                }
            }
        }

        trackedLandmarks.clear()
        trackedLandmarks.addAll(updatedList.take(AppConfig.VisualSlam.MAX_LANDMARKS))

        return trackedLandmarks.toList()
    }

    private fun detectCorners(gray: IntArray, w: Int, h: Int): List<PointF> {
        val corners = mutableListOf<PointF>()
        val step = AppConfig.VisualSlam.CORNER_STEP_PX
        for (y in step until h - step step step) {
            for (x in step until w - step step step) {
                var sumIx2 = 0f
                var sumIy2 = 0f
                var sumIxIy = 0f
                for (dy in -2..2) {
                    for (dx in -2..2) {
                        val idx = (y + dy) * w + (x + dx)
                        val ix = (gray[idx + 1] - gray[idx - 1]) / 2f
                        val iy = (gray[idx + w] - gray[idx - w]) / 2f
                        sumIx2 += ix * ix
                        sumIy2 += iy * iy
                        sumIxIy += ix * iy
                    }
                }
                val det = sumIx2 * sumIy2 - sumIxIy * sumIxIy
                val trace = sumIx2 + sumIy2
                val harrisScore = det - 0.04f * trace * trace
                if (harrisScore > AppConfig.VisualSlam.HARRIS_SCORE_THRESHOLD) {
                    corners.add(PointF(x.toFloat(), y.toFloat()))
                }
            }
        }
        return corners.take(AppConfig.VisualSlam.MAX_KEYPOINTS)
    }

    private fun lucasKanadeFlow(
        img1: IntArray,
        img2: IntArray,
        w: Int,
        h: Int,
        px: Int,
        py: Int
    ): PointF? {
        val win = AppConfig.VisualSlam.LK_WINDOW_RADIUS
        if (px - win < 1 || px + win >= w - 1 || py - win < 1 || py + win >= h - 1) return null

        var sumIx2 = 0f
        var sumIy2 = 0f
        var sumIxIy = 0f
        var sumIxIt = 0f
        var sumIyIt = 0f

        for (dy in -win..win) {
            for (dx in -win..win) {
                val idx1 = (py + dy) * w + (px + dx)
                val ix = (img1[idx1 + 1] - img1[idx1 - 1]) / 2f
                val iy = (img1[idx1 + w] - img1[idx1 - w]) / 2f
                val it = (img2[idx1] - img1[idx1]).toFloat()

                sumIx2 += ix * ix
                sumIy2 += iy * iy
                sumIxIy += ix * iy
                sumIxIt += ix * it
                sumIyIt += iy * it
            }
        }

        val det = sumIx2 * sumIy2 - sumIxIy * sumIxIy
        if (abs(det) < AppConfig.VisualSlam.LK_MIN_DETERMINANT) return null

        val u = (-sumIy2 * sumIxIt + sumIxIy * sumIyIt) / det
        val v = (sumIxIy * sumIxIt - sumIx2 * sumIyIt) / det

        val maxFlow = AppConfig.VisualSlam.LK_MAX_FLOW_PX
        return if (abs(u) < maxFlow && abs(v) < maxFlow) PointF(u, v) else null
    }

    private fun screenToMetricGround(
        u: Float, v: Float, w: Int, h: Int,
        pitchDeg: Float, aspect: Float, cameraHeight: Float
    ): Pair<Float, Float>? {
        val tanHalfV = tan(Math.toRadians(AppConfig.CameraGeometry.VFOV_DEG.toDouble()) / 2)
        val fy = (h / 2.0) / tanHalfV
        val fx = (w / 2.0) / (tanHalfV * aspect)
        val cx = w / 2.0
        val cy = h / 2.0

        val theta = Math.toRadians(pitchDeg.toDouble())

        val rowOffset = v - cy
        val colOffset = u - cx

        val rayAngle = theta + atan(rowOffset / fy)
        if (rayAngle <= 0.01 || rayAngle >= Math.PI / 2 - 0.01) return null

        val z = (cameraHeight / sin(rayAngle)).toFloat()
        val x = (colOffset * z / fx).toFloat()

        if (z < AppConfig.CameraGeometry.GRID_NEAR_M || z > AppConfig.CameraGeometry.GRID_FAR_M * 1.5f) return null
        return Pair(x, z)
    }
}

/**
 * Computes the surprise landscape based on static depth elevation,
 * Roel Vertegaal's work on visual attention and approaching motion surprise potential,
 * and user gaze focus points from smartglasses.
 */
class VertegaalSurpriseField {

    fun computeSurpriseGrid(
        grid: Array<FloatArray>,
        reference: Float,
        sigma: Float,
        landmarks: List<SlamLandmark>,
        gridRows: Int = AppConfig.CameraGeometry.GRID_ROWS,
        gridCols: Int = AppConfig.CameraGeometry.GRID_COLS,
        gazePoint: GazePoint? = null
    ): Array<FloatArray> {
        val near = AppConfig.CameraGeometry.GRID_NEAR_M
        val far = AppConfig.CameraGeometry.GRID_FAR_M
        val halfW = AppConfig.CameraGeometry.GRID_HALF_WIDTH_M

        return Array(gridRows) { r ->
            val z = far + (near - far) * r / (gridRows - 1)
            FloatArray(gridCols) { c ->
                val x = -halfW + 2 * halfW * c / (gridCols - 1)
                val elevVal = grid[r][c]

                val staticSurprise = if (elevVal.isNaN()) {
                    AppConfig.CameraGeometry.UNKNOWN_CELL_SURPRISE
                } else {
                    val diff = elevVal - reference
                    if (diff > AppConfig.CameraGeometry.OBSTACLE_ELEVATION_THRESHOLD_M) {
                        AppConfig.CameraGeometry.OBSTACLE_SURPRISE_WEIGHT * (diff / sigma).pow(2)
                    } else {
                        (diff / sigma).pow(2) / 2f
                    }
                }

                var motionSurprise = 0f
                for (lm in landmarks) {
                    if (lm.excessApproachingSpeed > AppConfig.VisualSlam.MIN_EXCESS_SPEED) {
                        val dx = x - lm.x
                        val dz = z - lm.z
                        val distSq = dx * dx + dz * dz
                        val sigmaM = AppConfig.VertegaalSurprise.LANDMARK_SIGMA_M
                        val spatialGaussian = exp(-distSq / (2 * sigmaM * sigmaM))
                        val speedFactor = lm.excessApproachingSpeed.pow(2)
                        motionSurprise += AppConfig.VertegaalSurprise.MOTION_WEIGHT * speedFactor * spatialGaussian
                    }
                }

                var gazeAttention = 0f
                if (gazePoint != null) {
                    val gazeR = (gazePoint.y * (gridRows - 1)).roundToInt().coerceIn(0, gridRows - 1)
                    val gazeC = (gazePoint.x * (gridCols - 1)).roundToInt().coerceIn(0, gridCols - 1)
                    val distSq = ((r - gazeR) * (r - gazeR) + (c - gazeC) * (c - gazeC)).toFloat()
                    gazeAttention = exp(-distSq / 12.0f) * 1.2f
                }

                staticSurprise + motionSurprise + gazeAttention
            }
        }
    }
}

/**
 * Solves discrete Euler-Lagrange trajectory optimization in the surprise potential field.
 */
data class LagrangianPathResult(
    val metricPath: List<Pair<Float, Float>>,
    val gridPath: List<Pair<Int, Int>>,
    val actionCost: Float,
    val steerAngleRad: Float,
    val steerInstruction: String
)

class LagrangianPathPlanner {

    fun planOptimalRoute(
        surprise: Array<FloatArray>,
        startGrid: Pair<Int, Int> = Pair(AppConfig.CameraGeometry.GRID_ROWS - 1, AppConfig.CameraGeometry.GRID_COLS / 2),
        goalRow: Int = 0
    ): LagrangianPathResult {
        val rows = surprise.size
        val cols = surprise[0].size

        val near = AppConfig.CameraGeometry.GRID_NEAR_M
        val far = AppConfig.CameraGeometry.GRID_FAR_M
        val halfW = AppConfig.CameraGeometry.GRID_HALF_WIDTH_M

        val startX = 0f
        val startZ = near
        val goalX = 0f
        val goalZ = far

        val numNodes = AppConfig.LagrangianPlanner.NUM_NODES
        val nodesX = FloatArray(numNodes)
        val nodesZ = FloatArray(numNodes)

        for (i in 0 until numNodes) {
            val t = i.toFloat() / (numNodes - 1)
            nodesX[i] = startX + t * (goalX - startX)
            nodesZ[i] = startZ + t * (goalZ - startZ)
        }

        repeat(AppConfig.LagrangianPlanner.NUM_RELAX_STEPS) {
            for (i in 1 until numNodes - 1) {
                val z = nodesZ[i]
                val x = nodesX[i]

                val gradS_x = computeSurpriseGradX(surprise, x, z, rows, cols, near, far, halfW)
                val tensionX = 2f * x - nodesX[i - 1] - nodesX[i + 1]
                val bendX = x - 0.5f * (nodesX[i - 1] + nodesX[i + 1])
                val straightForce = x - 0f

                val alpha = AppConfig.LagrangianPlanner.ALPHA_POTENTIAL
                val beta = AppConfig.LagrangianPlanner.BETA_BENDING
                val gamma = AppConfig.LagrangianPlanner.STRAIGHT_INERTIA

                val centerSurprise = sampleSurpriseAt(surprise, 0f, z, rows, cols, near, far, halfW)
                var repelledX = 0f
                if (centerSurprise > AppConfig.LagrangianPlanner.OBSTACLE_DETOUR_THRESHOLD && abs(x) < 0.25f) {
                    val sLeft = sampleSurpriseAt(surprise, -0.3f, z, rows, cols, near, far, halfW)
                    val sRight = sampleSurpriseAt(surprise, +0.3f, z, rows, cols, near, far, halfW)
                    val detourDirection = if (sLeft < sRight) -1f else +1f
                    repelledX = detourDirection * AppConfig.LagrangianPlanner.SYMMETRY_BREAK_REPELLER
                }

                val deltaX = tensionX + alpha * gradS_x + beta * bendX + gamma * straightForce - repelledX

                val clampRatio = AppConfig.LagrangianPlanner.LATERAL_CLAMP_RATIO
                nodesX[i] = (x - AppConfig.LagrangianPlanner.STEP_SIZE * deltaX).coerceIn(-halfW * clampRatio, halfW * clampRatio)
            }
        }

        // Post-processing Laplacian smoothing pass for rounded, fluid curves without sharp corners
        repeat(AppConfig.LagrangianPlanner.SMOOTHING_PASSES) {
            val smoothedX = nodesX.clone()
            for (i in 1 until numNodes - 1) {
                smoothedX[i] = 0.25f * nodesX[i - 1] + 0.5f * nodesX[i] + 0.25f * nodesX[i + 1]
            }
            for (i in 1 until numNodes - 1) {
                nodesX[i] = smoothedX[i]
            }
        }

        var totalAction = 0f
        val metricPath = mutableListOf<Pair<Float, Float>>()
        val gridPath = mutableListOf<Pair<Int, Int>>()

        for (i in 0 until numNodes) {
            val x = nodesX[i]
            val z = nodesZ[i]
            metricPath.add(Pair(x, z))

            val r = ((far - z) / (far - near) * (rows - 1)).roundToInt().coerceIn(0, rows - 1)
            val c = ((x + halfW) / (2 * halfW) * (cols - 1)).roundToInt().coerceIn(0, cols - 1)
            gridPath.add(Pair(r, c))

            val sVal = surprise[r][c]
            val kinetic = if (i < numNodes - 1) {
                val dx = nodesX[i + 1] - x
                val dz = nodesZ[i + 1] - z
                0.5f * (dx * dx + dz * dz)
            } else 0f

            totalAction += kinetic + AppConfig.LagrangianPlanner.ALPHA_POTENTIAL * sVal
        }

        val firstStepX = nodesX[2] - nodesX[0]
        val firstStepZ = nodesZ[2] - nodesZ[0]
        val steerAngleRad = atan2(firstStepX, firstStepZ)

        val threshold = AppConfig.LagrangianPlanner.STEER_THRESHOLD_M
        val steerInstruction = when {
            firstStepX < -threshold -> "STEER LEFT ◄"
            firstStepX > threshold -> "STEER RIGHT ►"
            else -> "STRAIGHT AHEAD ▲"
        }

        return LagrangianPathResult(
            metricPath = metricPath,
            gridPath = gridPath,
            actionCost = totalAction,
            steerAngleRad = steerAngleRad,
            steerInstruction = steerInstruction
        )
    }

    private fun computeSurpriseGradX(
        surprise: Array<FloatArray>,
        x: Float, z: Float,
        rows: Int, cols: Int,
        near: Float, far: Float, halfW: Float
    ): Float {
        val deltaX = AppConfig.LagrangianPlanner.FINITE_DIFF_DELTA_X
        val xPlus = (x + deltaX).coerceIn(-halfW, halfW)
        val xMinus = (x - deltaX).coerceIn(-halfW, halfW)

        val r = ((far - z) / (far - near) * (rows - 1)).roundToInt().coerceIn(0, rows - 1)
        val cPlus = ((xPlus + halfW) / (2 * halfW) * (cols - 1)).roundToInt().coerceIn(0, cols - 1)
        val cMinus = ((xMinus + halfW) / (2 * halfW) * (cols - 1)).roundToInt().coerceIn(0, cols - 1)

        val sPlus = surprise[r][cPlus]
        val sMinus = surprise[r][cMinus]

        return (sPlus - sMinus) / (2f * deltaX)
    }

    private fun sampleSurpriseAt(
        surprise: Array<FloatArray>,
        x: Float, z: Float,
        rows: Int, cols: Int,
        near: Float, far: Float, halfW: Float
    ): Float {
        val r = ((far - z) / (far - near) * (rows - 1)).roundToInt().coerceIn(0, rows - 1)
        val c = ((x + halfW) / (2 * halfW) * (cols - 1)).roundToInt().coerceIn(0, cols - 1)
        return surprise[r][c]
    }
}

/**
 * AR Camera Overlay View rendering projected Lagrangian path ribbons, warning highlights,
 * eye-tracking gaze target reticles from Pupil Neon smartglasses, and guidance arrows.
 */
class CameraGuidanceOverlayView(context: Context, attrs: AttributeSet? = null) : View(context, attrs) {

    private var metricPath: List<Pair<Float, Float>> = emptyList()
    private var landmarks: List<SlamLandmark> = emptyList()
    private var steerInstruction: String = "INITIALIZING"
    private var steerAngleRad: Float = 0f
    private var pitchDeg: Float = AppConfig.CameraGeometry.FALLBACK_PITCH_DEG
    private var aspect: Float = 4f / 3f
    private var currentGaze: GazePoint? = null
    private var isGlassesMode: Boolean = false
    private var gazePulseAnim = 0f

    private val pathRibbonPaint = Paint().apply {
        color = Color.parseColor(AppConfig.GuidanceUi.COLOR_PATH_CYAN)
        strokeWidth = AppConfig.GuidanceUi.PATH_RIBBON_WIDTH_PX
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
        strokeJoin = Paint.Join.ROUND
        pathEffect = CornerPathEffect(20f)
    }

    private val pathGlowPaint = Paint().apply {
        color = Color.parseColor(AppConfig.GuidanceUi.COLOR_PATH_GLOW)
        strokeWidth = AppConfig.GuidanceUi.PATH_GLOW_WIDTH_PX
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
        pathEffect = CornerPathEffect(20f)
    }

    private val arrowFillPaint = Paint().apply {
        color = Color.parseColor(AppConfig.GuidanceUi.COLOR_STEER_AHEAD)
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    private val arrowStrokePaint = Paint().apply {
        color = Color.WHITE
        strokeWidth = AppConfig.GuidanceUi.ARROW_STROKE_WIDTH_PX
        style = Paint.Style.STROKE
        isAntiAlias = true
    }

    private val warningPaint = Paint().apply {
        color = Color.RED
        strokeWidth = AppConfig.GuidanceUi.WARNING_STROKE_WIDTH_PX
        style = Paint.Style.STROKE
        pathEffect = DashPathEffect(floatArrayOf(15f, 10f), 0f)
    }

    private val badgeBgPaint = Paint().apply {
        color = Color.argb(AppConfig.GuidanceUi.COLOR_HUD_BG_SLATE, 15, 23, 42)
        style = Paint.Style.FILL
    }

    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = AppConfig.GuidanceUi.TEXT_SIZE_TITLE
        typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
        isAntiAlias = true
    }

    private val subTextPaint = Paint().apply {
        color = Color.parseColor(AppConfig.GuidanceUi.COLOR_HUD_YELLOW)
        textSize = AppConfig.GuidanceUi.TEXT_SIZE_SUB
        typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
        isAntiAlias = true
    }

    private val gazeDotPaint = Paint().apply {
        color = Color.parseColor(AppConfig.PupilNeon.COLOR_GAZE_NEON)
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    private val gazeGlowPaint = Paint().apply {
        color = Color.parseColor(AppConfig.PupilNeon.COLOR_GAZE_GLOW)
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    private val gazeRingPaint = Paint().apply {
        color = Color.parseColor(AppConfig.PupilNeon.COLOR_GAZE_RING)
        strokeWidth = 4f
        style = Paint.Style.STROKE
        isAntiAlias = true
    }

    private val gazeTextPaint = Paint().apply {
        color = Color.parseColor(AppConfig.PupilNeon.COLOR_GAZE_TEXT)
        textSize = 28f
        typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
        isAntiAlias = true
    }

    fun update(
        path: List<Pair<Float, Float>>,
        instruction: String,
        steerAngle: Float,
        slamLandmarks: List<SlamLandmark>,
        pitch: Float,
        cameraAspect: Float,
        gaze: GazePoint? = null,
        glassesActive: Boolean = false
    ) {
        metricPath = path
        steerInstruction = instruction
        steerAngleRad = steerAngle
        landmarks = slamLandmarks
        pitchDeg = pitch
        aspect = cameraAspect
        currentGaze = gaze
        isGlassesMode = glassesActive
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val w = width.toFloat()
        val h = height.toFloat()

        if (w <= 0 || h <= 0) return

        if (metricPath.size > 1) {
            val screenPath = Path()
            var first = true

            for ((mx, mz) in metricPath) {
                val pt = projectToScreen(mx, mz, w, h)
                if (pt != null) {
                    if (first) {
                        screenPath.moveTo(pt.x, pt.y)
                        first = false
                    } else {
                        screenPath.lineTo(pt.x, pt.y)
                    }
                }
            }

            canvas.drawPath(screenPath, pathGlowPaint)
            canvas.drawPath(screenPath, pathRibbonPaint)
        }

        for (lm in landmarks) {
            if (lm.excessApproachingSpeed > AppConfig.VisualSlam.MIN_EXCESS_SPEED) {
                val pt = projectToScreen(lm.x, lm.z, w, h)
                if (pt != null) {
                    val radius = (60f / max(1f, lm.z)).coerceIn(25f, 90f)
                    canvas.drawCircle(pt.x, pt.y, radius, warningPaint)
                    val label = "▲ LOOMING +%.1fm/s".format(lm.excessApproachingSpeed)
                    canvas.drawText(label, pt.x - 80f, pt.y - radius - 10f, subTextPaint)
                }
            }
        }

        if (isGlassesMode && currentGaze != null) {
            drawGazeReticle(canvas, w, h)
        }

        drawGuidanceArrow(canvas, w, h)
        drawHudBanner(canvas, w)
    }

    private fun drawGazeReticle(canvas: Canvas, w: Float, h: Float) {
        val gaze = currentGaze ?: return
        val px = gaze.x * w
        val py = gaze.y * h

        gazePulseAnim = (gazePulseAnim + 0.12f) % (2f * Math.PI.toFloat())
        val pulseRadius = AppConfig.PupilNeon.GAZE_RETICLE_RADIUS_PX + 6f * sin(gazePulseAnim)

        // Outer aura
        canvas.drawCircle(px, py, pulseRadius * 1.6f, gazeGlowPaint)

        // Target Ring
        canvas.drawCircle(px, py, pulseRadius, gazeRingPaint)

        // Reticle Crosshairs
        val chSize = pulseRadius * 0.75f
        canvas.drawLine(px - chSize, py, px + chSize, py, gazeRingPaint)
        canvas.drawLine(px, py - chSize, px, py + chSize, gazeRingPaint)

        // Center Dot
        canvas.drawCircle(px, py, 10f, gazeDotPaint)

        // Gaze Position Label
        val simTag = if (gaze.isSimulated) " (Simulated)" else " (Live)"
        val label = "GAZE Target: (%.2f, %.2f)%s".format(gaze.x, gaze.y, simTag)
        val textWidth = gazeTextPaint.measureText(label)
        val textX = (px + pulseRadius + 14f).coerceIn(10f, w - textWidth - 10f)
        val textY = (py - pulseRadius - 10f).coerceIn(40f, h - 20f)

        val textBgRect = RectF(textX - 10f, textY - 26f, textX + textWidth + 10f, textY + 8f)
        canvas.drawRoundRect(textBgRect, 10f, 10f, badgeBgPaint)
        canvas.drawText(label, textX, textY, gazeTextPaint)
    }

    private fun drawGuidanceArrow(canvas: Canvas, w: Float, h: Float) {
        val arrowCenterX = w / 2f
        val arrowCenterY = h - 140f

        canvas.save()
        canvas.translate(arrowCenterX, arrowCenterY)
        canvas.rotate(Math.toDegrees(steerAngleRad.toDouble()).toFloat())

        val arrowPath = Path().apply {
            moveTo(0f, -70f)
            lineTo(35f, 30f)
            lineTo(15f, 20f)
            lineTo(15f, 60f)
            lineTo(-15f, 60f)
            lineTo(-15f, 20f)
            lineTo(-35f, 30f)
            close()
        }

        arrowFillPaint.color = when {
            steerInstruction.contains("LEFT") -> Color.parseColor(AppConfig.GuidanceUi.COLOR_STEER_LEFT)
            steerInstruction.contains("RIGHT") -> Color.parseColor(AppConfig.GuidanceUi.COLOR_STEER_RIGHT)
            else -> Color.parseColor(AppConfig.GuidanceUi.COLOR_STEER_AHEAD)
        }

        canvas.drawPath(arrowPath, arrowFillPaint)
        canvas.drawPath(arrowPath, arrowStrokePaint)
        canvas.restore()

        val instructionWidth = textPaint.measureText(steerInstruction)
        canvas.drawText(steerInstruction, (w - instructionWidth) / 2f, h - 30f, textPaint)
    }

    private fun drawHudBanner(canvas: Canvas, w: Float) {
        val rect = RectF(20f, 20f, w - 20f, 130f)
        canvas.drawRoundRect(rect, 20f, 20f, badgeBgPaint)

        val activeLooming = landmarks.count { it.excessApproachingSpeed > AppConfig.VisualSlam.MIN_EXCESS_SPEED }
        val modeTitle = if (isGlassesMode) "NEON GLASSES" else "SURPRISE"
        val titleText = "$modeTitle: $steerInstruction"

        val gazeInfo = if (isGlassesMode && currentGaze != null) {
            val g = currentGaze!!
            "Gaze: (%.2f, %.2f) %s".format(g.x, g.y, if (g.isSimulated) "[SIM]" else "[LIVE]")
        } else {
            "SLAM Landmarks: ${landmarks.size}"
        }

        val statusText = "$gazeInfo | Looming Threats: $activeLooming"

        canvas.drawText(titleText, 40f, 70f, textPaint)
        canvas.drawText(statusText, 40f, 110f, subTextPaint)
    }

    private fun projectToScreen(mx: Float, mz: Float, viewW: Float, viewH: Float): PointF? {
        val tanHalfV = tan(Math.toRadians(AppConfig.CameraGeometry.VFOV_DEG.toDouble()) / 2)
        val fy = (viewH / 2.0) / tanHalfV
        val fx = (viewW / 2.0) / (tanHalfV * aspect)
        val cx = viewW / 2.0
        val cy = viewH / 2.0

        val theta = Math.toRadians(pitchDeg.toDouble())
        val sinT = sin(theta)
        val cosT = cos(theta)
        val camH = AppConfig.CameraGeometry.CAMERA_HEIGHT_M.toDouble()

        val xc = mx.toDouble()
        val yc = mz * sinT - camH * cosT
        val zc = mz * cosT + camH * sinT

        if (zc <= 0.1) return null

        val u = cx + fx * (xc / zc)
        val v = cy - fy * (yc / zc)

        if (u < -100 || u > viewW + 100 || v < -100 || v > viewH + 100) return null

        return PointF(u.toFloat(), v.toFloat())
    }
}

/**
 * Main Activity integrating Visual SLAM, Depth-Anything-V2 ONNX inference,
 * Roel Vertegaal's approaching motion surprise landscape, discrete Lagrangian trajectory optimization,
 * Pupil Labs Neon smartglasses eye-gaze tracking, and live AR guidance rendering.
 */
class MainActivity : AppCompatActivity() {

    companion object {
        const val CAMERA_HEIGHT = AppConfig.CameraGeometry.CAMERA_HEIGHT_M
        const val VFOV_DEG = AppConfig.CameraGeometry.VFOV_DEG
        const val FALLBACK_PITCH_DEG = AppConfig.CameraGeometry.FALLBACK_PITCH_DEG
        const val GRID_ROWS = AppConfig.CameraGeometry.GRID_ROWS
        const val GRID_COLS = AppConfig.CameraGeometry.GRID_COLS
        const val SIGMA = AppConfig.CameraGeometry.SIGMA
        const val ANALYSIS_INTERVAL_MS = AppConfig.CameraGeometry.ANALYSIS_INTERVAL_MS

        const val GRID_NEAR_M = AppConfig.CameraGeometry.GRID_NEAR_M
        const val GRID_FAR_M = AppConfig.CameraGeometry.GRID_FAR_M
        const val GRID_HALF_WIDTH_M = AppConfig.CameraGeometry.GRID_HALF_WIDTH_M
        const val UNKNOWN_CELL_SURPRISE = AppConfig.CameraGeometry.UNKNOWN_CELL_SURPRISE

        const val REFERENCE_EMA_ALPHA = AppConfig.CameraGeometry.REFERENCE_EMA_ALPHA
        const val REFERENCE_MIN_CELLS = AppConfig.CameraGeometry.REFERENCE_MIN_CELLS

        const val MODEL_INPUT_SIZE = AppConfig.DepthModel.MODEL_INPUT_SIZE
        const val MODEL_ASSET_NAME = AppConfig.DepthModel.MODEL_ASSET_NAME
        val IMAGENET_MEAN = AppConfig.DepthModel.IMAGENET_MEAN
        val IMAGENET_STD = AppConfig.DepthModel.IMAGENET_STD

        const val SYNTH_W = AppConfig.DepthModel.SYNTH_W
        const val SYNTH_H = AppConfig.DepthModel.SYNTH_H

        const val PREFS_NAME = AppConfig.DepthModel.PREFS_NAME
        const val PREF_DEPTH_SCALE = AppConfig.DepthModel.PREF_DEPTH_SCALE
        const val DEFAULT_DEPTH_SCALE = AppConfig.DepthModel.DEFAULT_DEPTH_SCALE
        const val CALIBRATION_BAND_FRACTION = AppConfig.CameraGeometry.CALIBRATION_BAND_FRACTION
    }

    private class RawFrame(val depth: Array<FloatArray>, val pitchDeg: Float)

    private lateinit var previewView: PreviewView
    private lateinit var cameraGuidanceOverlay: CameraGuidanceOverlayView
    private lateinit var overlayView: SurpriseOverlayView
    private lateinit var scaleLabel: TextView
    private lateinit var pitchLabel: TextView
    private lateinit var sourceButton: Button
    private lateinit var neonConfigButton: Button
    private lateinit var hintLabel: TextView

    private lateinit var cameraExecutor: ExecutorService
    private lateinit var sensorManager: SensorManager
    private lateinit var pupilNeonManager: PupilNeonManager

    private var visionSource = VisionSource.ATTACHED_CAMERA
    private var ortEnvironment: OrtEnvironment? = null
    private var session: OrtSession? = null
    private var lastAnalysisTimeMs = 0L

    private val slamTracker = VisualSlamTracker()
    private val vertegaalSurpriseField = VertegaalSurpriseField()
    private val lagrangianPlanner = LagrangianPathPlanner()

    private var depthScale = DEFAULT_DEPTH_SCALE

    @Volatile private var lastRaw: RawFrame? = null
    @Volatile private var sensorPitchDeg: Float? = null
    private var referenceElevation: Float? = null

    private val pitchListener = object : SensorEventListener {
        private val rotationMatrix = FloatArray(9)

        override fun onSensorChanged(event: SensorEvent) {
            SensorManager.getRotationMatrixFromVector(rotationMatrix, event.values)
            val r8 = rotationMatrix[8].coerceIn(-1f, 1f).toDouble()
            sensorPitchDeg = Math.toDegrees(asin(r8)).toFloat()
        }

        override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
    }

    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startCamera() else finish()
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE)
        depthScale = prefs.getFloat(PREF_DEPTH_SCALE, DEFAULT_DEPTH_SCALE)
        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager

        pupilNeonManager = PupilNeonManager(this)
        val savedSourceOrdinal = prefs.getInt(AppConfig.PupilNeon.PREF_VISION_SOURCE, VisionSource.ATTACHED_CAMERA.ordinal)
        visionSource = VisionSource.entries.getOrElse(savedSourceOrdinal) { VisionSource.ATTACHED_CAMERA }
        pupilNeonManager.host = prefs.getString(AppConfig.PupilNeon.PREF_NEON_HOST, AppConfig.PupilNeon.DEFAULT_HOST) ?: AppConfig.PupilNeon.DEFAULT_HOST
        pupilNeonManager.port = prefs.getInt(AppConfig.PupilNeon.PREF_NEON_PORT, AppConfig.PupilNeon.DEFAULT_PORT)

        previewView = PreviewView(this)
        cameraGuidanceOverlay = CameraGuidanceOverlayView(this)
        overlayView = SurpriseOverlayView(this)

        scaleLabel = TextView(this).apply {
            text = "Scale: %.3f".format(depthScale)
            setPadding(12, 0, 0, 0)
        }
        pitchLabel = TextView(this).apply {
            text = "Pitch: --"
            setPadding(12, 0, 0, 0)
        }
        sourceButton = Button(this).apply {
            setOnClickListener { toggleVisionSource() }
        }
        neonConfigButton = Button(this).apply {
            text = "⚙️ Neon"
            setOnClickListener { showNeonConfigDialog() }
        }
        val calibrateButton = Button(this).apply {
            text = "Calibrate"
            setOnClickListener { calibrateDepthScale() }
        }

        val controlBar = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(12, 12, 12, 12)
            addView(sourceButton)
            addView(neonConfigButton)
            addView(calibrateButton)
            addView(scaleLabel)
            addView(pitchLabel)
        }

        hintLabel = TextView(this).apply {
            setPadding(24, 0, 24, 16)
            textSize = 12f
        }

        updateSourceModeUI()

        val cameraContainer = FrameLayout(this).apply {
            addView(previewView, FrameLayout.LayoutParams(FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT))
            addView(cameraGuidanceOverlay, FrameLayout.LayoutParams(FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT))
        }

        val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        root.addView(controlBar)
        root.addView(hintLabel)
        root.addView(
            cameraContainer,
            LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 0, 0.55f)
        )
        root.addView(
            overlayView,
            LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 0, 0.45f)
        )
        setContentView(root)

        cameraExecutor = Executors.newSingleThreadExecutor()
        val env = OrtEnvironment.getEnvironment()
        ortEnvironment = env
        session = loadOrtSession(env)

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun toggleVisionSource() {
        visionSource = if (visionSource == VisionSource.ATTACHED_CAMERA) {
            VisionSource.PUPIL_NEON_GLASSES
        } else {
            VisionSource.ATTACHED_CAMERA
        }
        getSharedPreferences(PREFS_NAME, MODE_PRIVATE).edit()
            .putInt(AppConfig.PupilNeon.PREF_VISION_SOURCE, visionSource.ordinal)
            .apply()

        updateSourceModeUI()
    }

    private fun updateSourceModeUI() {
        if (visionSource == VisionSource.PUPIL_NEON_GLASSES) {
            sourceButton.text = "Source: Neon Glasses 👓"
            neonConfigButton.visibility = View.VISIBLE
            pupilNeonManager.start()
            hintLabel.text = "Pupil Neon Smartglasses Active | Gaze Tracking Enabled"
            Toast.makeText(this, "Switched to Pupil Neon Smartglasses", Toast.LENGTH_SHORT).show()
        } else {
            sourceButton.text = "Source: Camera 📷"
            neonConfigButton.visibility = View.GONE
            pupilNeonManager.stop()
            hintLabel.text = "Attached Camera Active | Hold phone ~%.1f m up".format(CAMERA_HEIGHT)
            Toast.makeText(this, "Switched to Attached Camera", Toast.LENGTH_SHORT).show()
        }
    }

    private fun showNeonConfigDialog() {
        val layout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(40, 20, 40, 10)
        }

        val hostLabel = TextView(this).apply { text = "Pupil Companion IP / Host:" }
        val hostInput = EditText(this).apply {
            setText(pupilNeonManager.host)
            hint = AppConfig.PupilNeon.DEFAULT_HOST
        }

        val portLabel = TextView(this).apply { text = "API Port:" }
        val portInput = EditText(this).apply {
            setText(pupilNeonManager.port.toString())
            inputType = InputType.TYPE_CLASS_NUMBER
        }

        val simCheckBox = CheckBox(this).apply {
            text = "Use Simulated Gaze Feed (Demo / Test)"
            isChecked = pupilNeonManager.isSimulated()
        }

        val statusText = TextView(this).apply {
            text = "Status: ${pupilNeonManager.getStatusText()}"
            setPadding(0, 16, 0, 16)
        }

        layout.addView(hostLabel)
        layout.addView(hostInput)
        layout.addView(portLabel)
        layout.addView(portInput)
        layout.addView(simCheckBox)
        layout.addView(statusText)

        AlertDialog.Builder(this)
            .setTitle("Pupil Neon Smartglasses Settings")
            .setView(layout)
            .setPositiveButton("Apply") { _, _ ->
                val newHost = hostInput.text.toString().trim().ifEmpty { AppConfig.PupilNeon.DEFAULT_HOST }
                val newPort = portInput.text.toString().toIntOrNull() ?: AppConfig.PupilNeon.DEFAULT_PORT
                val useSim = simCheckBox.isChecked

                pupilNeonManager.host = newHost
                pupilNeonManager.port = newPort
                pupilNeonManager.setSimulationMode(useSim)

                getSharedPreferences(PREFS_NAME, MODE_PRIVATE).edit()
                    .putString(AppConfig.PupilNeon.PREF_NEON_HOST, newHost)
                    .putInt(AppConfig.PupilNeon.PREF_NEON_PORT, newPort)
                    .apply()

                Toast.makeText(this, "Neon settings updated", Toast.LENGTH_SHORT).show()
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    override fun onResume() {
        super.onResume()
        val sensor = sensorManager.getDefaultSensor(Sensor.TYPE_GAME_ROTATION_VECTOR)
            ?: sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
        if (sensor != null) {
            sensorManager.registerListener(pitchListener, sensor, SensorManager.SENSOR_DELAY_GAME)
        }
        if (visionSource == VisionSource.PUPIL_NEON_GLASSES) {
            pupilNeonManager.start()
        }
    }

    override fun onPause() {
        super.onPause()
        sensorManager.unregisterListener(pitchListener)
        sensorPitchDeg = null
        pupilNeonManager.stop()
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        session?.close()
        pupilNeonManager.stop()
    }

    private fun loadOrtSession(env: OrtEnvironment): OrtSession? = try {
        val modelBytes = assets.open(MODEL_ASSET_NAME).use { it.readBytes() }
        env.createSession(modelBytes, OrtSession.SessionOptions())
    } catch (e: Exception) {
        null
    }

    private fun calibrateDepthScale() {
        val raw = lastRaw
        if (session == null) {
            Toast.makeText(this, "No depth model loaded -- add depth_model.onnx first", Toast.LENGTH_LONG).show()
            return
        }
        if (raw == null) {
            Toast.makeText(this, "No frame yet -- try again in a moment", Toast.LENGTH_SHORT).show()
            return
        }
        cameraExecutor.execute {
            val depth = raw.depth
            val outH = depth.size
            val bandStart = (outH * (1f - CALIBRATION_BAND_FRACTION)).toInt().coerceIn(0, outH - 1)

            val rawValues = mutableListOf<Float>()
            for (r in bandStart until outH) rawValues.addAll(depth[r].toList())
            val rawMedian = medianOf(rawValues)

            val (_, expectedDepth) = groundPlaneExpectedDepth(outH, CAMERA_HEIGHT, VFOV_DEG, raw.pitchDeg)
            var expectedSum = 0f
            for (r in bandStart until outH) expectedSum += expectedDepth[r]
            val expectedMean = expectedSum / (outH - bandStart)

            val newScale = expectedMean * rawMedian
            depthScale = newScale
            referenceElevation = null
            getSharedPreferences(PREFS_NAME, MODE_PRIVATE).edit()
                .putFloat(PREF_DEPTH_SCALE, newScale)
                .apply()

            runOnUiThread {
                scaleLabel.text = "Scale: %.3f".format(newScale)
                Toast.makeText(this, "Calibrated", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val provider = providerFuture.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }
            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
            analysis.setAnalyzer(cameraExecutor) { imageProxy -> analyzeFrame(imageProxy) }

            provider.unbindAll()
            provider.bindToLifecycle(
                this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis
            )
        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyzeFrame(imageProxy: ImageProxy) {
        val now = System.currentTimeMillis()
        if (now - lastAnalysisTimeMs < ANALYSIS_INTERVAL_MS) {
            imageProxy.close()
            return
        }
        lastAnalysisTimeMs = now

        try {
            val bitmap = imageProxyToBitmap(imageProxy)

            val model = session
            val depth: Array<FloatArray>
            val pitchDeg: Float
            val aspect: Float
            var usingSensor = false
            if (model != null) {
                val sensorPitch = sensorPitchDeg
                usingSensor = sensorPitch != null
                pitchDeg = sensorPitch ?: FALLBACK_PITCH_DEG
                aspect = bitmap.width.toFloat() / bitmap.height
                depth = runDepthModel(model, bitmap, pitchDeg)
            } else {
                pitchDeg = FALLBACK_PITCH_DEG
                aspect = SYNTH_W.toFloat() / SYNTH_H
                depth = makeSyntheticDepth()
            }

            val slamLandmarks = slamTracker.processFrame(bitmap, pitchDeg, aspect)
            val elevation = depthToElevation(depth, pitchDeg)
            val grid = toTopDownGrid(elevation, pitchDeg, aspect)
            val reference = updateReferenceElevation(grid)

            val gaze = if (visionSource == VisionSource.PUPIL_NEON_GLASSES) pupilNeonManager.getCurrentGaze() else null
            val surprise = vertegaalSurpriseField.computeSurpriseGrid(grid, reference, SIGMA, slamLandmarks, gazePoint = gaze)
            val start = Pair(grid.size - 1, grid[0].size / 2)
            val lagrangianResult = lagrangianPlanner.planOptimalRoute(surprise, start)

            runOnUiThread {
                overlayView.update(
                    newGrid = grid,
                    newPath = lagrangianResult.gridPath,
                    newStart = start,
                    slamLandmarks = slamLandmarks,
                    gaze = gaze,
                    glassesActive = (visionSource == VisionSource.PUPIL_NEON_GLASSES)
                )
                cameraGuidanceOverlay.update(
                    path = lagrangianResult.metricPath,
                    instruction = lagrangianResult.steerInstruction,
                    steerAngle = lagrangianResult.steerAngleRad,
                    slamLandmarks = slamLandmarks,
                    pitch = pitchDeg,
                    cameraAspect = aspect,
                    gaze = gaze,
                    glassesActive = (visionSource == VisionSource.PUPIL_NEON_GLASSES)
                )
                pitchLabel.text = "Pitch: %.1f°%s".format(pitchDeg, if (usingSensor) "" else " (fixed)")
                scaleLabel.text = "Scale: %.3f | Act: %.1f".format(depthScale, lagrangianResult.actionCost)
            }
        } finally {
            imageProxy.close()
        }
    }

    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap {
        val yBuffer = imageProxy.planes[0].buffer
        val uBuffer = imageProxy.planes[1].buffer
        val vBuffer = imageProxy.planes[2].buffer
        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()
        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, imageProxy.width, imageProxy.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, imageProxy.width, imageProxy.height), 90, out)
        val bytes = out.toByteArray()
        val bmp = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
        val matrix = Matrix().apply { postRotate(imageProxy.imageInfo.rotationDegrees.toFloat()) }
        return Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, matrix, true)
    }

    private fun runDepthModel(session: OrtSession, bitmap: Bitmap, pitchDeg: Float): Array<FloatArray> {
        val env = ortEnvironment ?: OrtEnvironment.getEnvironment()
        val size = MODEL_INPUT_SIZE
        val scaled = Bitmap.createScaledBitmap(bitmap, size, size, true)
        val pixels = IntArray(size * size)
        scaled.getPixels(pixels, 0, size, 0, 0, size, size)

        val channelSize = size * size
        val inputData = FloatArray(3 * channelSize)
        for (i in 0 until channelSize) {
            val px = pixels[i]
            val r = ((px shr 16) and 0xFF) / 255f
            val g = ((px shr 8) and 0xFF) / 255f
            val b = (px and 0xFF) / 255f
            inputData[i] = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
            inputData[channelSize + i] = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
            inputData[2 * channelSize + i] = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2]
        }
        val inputBuffer = FloatBuffer.wrap(inputData)

        val inputName = session.inputNames.iterator().next()
        val depthRaw: Array<FloatArray>
        OnnxTensor.createTensor(env, inputBuffer, longArrayOf(1, 3, size.toLong(), size.toLong())).use { inputTensor ->
            session.run(mapOf(inputName to inputTensor)).use { result ->
                val outputTensor = result[0] as OnnxTensor
                val shape = outputTensor.info.shape
                val outH = shape[1].toInt()
                val outW = shape[2].toInt()
                val outBuf = outputTensor.floatBuffer
                depthRaw = Array(outH) { r -> FloatArray(outW) { c -> outBuf.get(r * outW + c) } }
            }
        }
        lastRaw = RawFrame(depthRaw, pitchDeg)

        return Array(depthRaw.size) { r ->
            FloatArray(depthRaw[0].size) { c ->
                depthScale / max(depthRaw[r][c], AppConfig.DepthModel.MIN_RAW_DEPTH)
            }
        }
    }

    private fun makeSyntheticDepth(h: Int = SYNTH_H, w: Int = SYNTH_W): Array<FloatArray> {
        val (_, expectedDepth) = groundPlaneExpectedDepth(h, CAMERA_HEIGHT, VFOV_DEG, FALLBACK_PITCH_DEG)
        val depth = Array(h) { r -> FloatArray(w) { expectedDepth[r] } }
        for (r in 40 until 56) for (c in 0 until w) depth[r][c] -= 0.35f
        for (r in 70 until 85) for (c in 60 until 100) depth[r][c] += 0.6f
        for (r in 30 until 55) for (c in 120 until 145) depth[r][c] -= 0.8f
        val rng = Random(7)
        for (r in 0 until h) for (c in 0 until w) {
            depth[r][c] = max(0.05f, depth[r][c] + (rng.nextFloat() - 0.5f) * 2f * 0.015f)
        }
        return depth
    }

    private fun groundPlaneExpectedDepth(
        h: Int, cameraHeight: Float, vfovDeg: Float, pitchDeg: Float
    ): Pair<FloatArray, FloatArray> {
        val fy = (h / 2.0) / tan(Math.toRadians(vfovDeg.toDouble()) / 2)
        val pitch = Math.toRadians(pitchDeg.toDouble())
        val rayAngle = FloatArray(h)
        val expectedDepth = FloatArray(h)
        for (r in 0 until h) {
            val rowOffsetPx = (r + 0.5) - h / 2.0
            val angle = pitch + atan(rowOffsetPx / fy)
            val safeAngle = angle.coerceIn(Math.toRadians(2.0), Math.toRadians(88.0))
            rayAngle[r] = safeAngle.toFloat()
            expectedDepth[r] = (cameraHeight / sin(safeAngle)).toFloat()
        }
        return Pair(rayAngle, expectedDepth)
    }

    private fun depthToElevation(
        depth: Array<FloatArray>,
        pitchDeg: Float,
        cameraHeight: Float = CAMERA_HEIGHT,
        vfovDeg: Float = VFOV_DEG
    ): Array<FloatArray> {
        val h = depth.size
        val w = depth[0].size
        val (rayAngle, expectedDepth) = groundPlaneExpectedDepth(h, cameraHeight, vfovDeg, pitchDeg)
        return Array(h) { r ->
            FloatArray(w) { c ->
                val deltaDepth = expectedDepth[r] - depth[r][c]
                deltaDepth * sin(rayAngle[r].toDouble()).toFloat()
            }
        }
    }

    private fun toTopDownGrid(
        elevation: Array<FloatArray>,
        pitchDeg: Float,
        aspect: Float,
        gridRows: Int = GRID_ROWS,
        gridCols: Int = GRID_COLS
    ): Array<FloatArray> {
        val h = elevation.size
        val w = elevation[0].size
        val tanHalfV = tan(Math.toRadians(VFOV_DEG.toDouble()) / 2)
        val fy = (h / 2.0) / tanHalfV
        val fx = (w / 2.0) / (tanHalfV * aspect)
        val cx = w / 2.0
        val cy = h / 2.0
        val theta = Math.toRadians(pitchDeg.toDouble())
        val sinT = sin(theta)
        val cosT = cos(theta)
        val camH = CAMERA_HEIGHT.toDouble()

        val hm = doubleArrayOf(
            fx, cx * cosT, cx * camH * sinT,
            0.0, -fy * sinT + cy * cosT, fy * camH * cosT + cy * camH * sinT,
            0.0, cosT, camH * sinT
        )

        val near = GRID_NEAR_M.toDouble()
        val far = GRID_FAR_M.toDouble()
        val half = GRID_HALF_WIDTH_M.toDouble()
        return Array(gridRows) { i ->
            val z = far + (near - far) * i / (gridRows - 1)
            FloatArray(gridCols) { j ->
                val x = -half + 2 * half * j / (gridCols - 1)
                val a = hm[0] * x + hm[1] * z + hm[2]
                val b = hm[3] * x + hm[4] * z + hm[5]
                val c = hm[6] * x + hm[7] * z + hm[8]
                if (c <= 1e-6) {
                    Float.NaN
                } else {
                    sampleBilinear(elevation, a / c - 0.5, b / c - 0.5)
                }
            }
        }
    }

    private fun sampleBilinear(map: Array<FloatArray>, px: Double, py: Double): Float {
        val h = map.size
        val w = map[0].size
        if (px < 0 || py < 0 || px > w - 1 || py > h - 1) return Float.NaN
        val x0 = px.toInt()
        val y0 = py.toInt()
        val x1 = min(x0 + 1, w - 1)
        val y1 = min(y0 + 1, h - 1)
        val tx = (px - x0).toFloat()
        val ty = (py - y0).toFloat()
        val top = map[y0][x0] * (1 - tx) + map[y0][x1] * tx
        val bottom = map[y1][x0] * (1 - tx) + map[y1][x1] * tx
        return top * (1 - ty) + bottom * ty
    }

    private fun medianOf(values: MutableList<Float>): Float {
        values.sort()
        val mid = values.size / 2
        return if (values.size % 2 == 0) (values[mid - 1] + values[mid]) / 2f else values[mid]
    }

    private fun updateReferenceElevation(grid: Array<FloatArray>): Float {
        val values = mutableListOf<Float>()
        for (r in grid.size - 1 downTo 0) {
            for (v in grid[r]) if (!v.isNaN()) values.add(v)
            if (values.size >= REFERENCE_MIN_CELLS) break
        }
        val previous = referenceElevation
        if (values.isEmpty()) return previous ?: 0f

        val frameMedian = medianOf(values)
        val updated = if (previous == null) frameMedian
        else previous + REFERENCE_EMA_ALPHA * (frameMedian - previous)
        referenceElevation = updated
        return updated
    }
}

/**
 * Top-down surprise landscape map view overlay with active SLAM feature velocities,
 * elevation + Vertegaal motion surprise heat map, Pupil Neon gaze target, and Lagrangian path curves.
 */
class SurpriseOverlayView(context: Context, attrs: AttributeSet? = null) : View(context, attrs) {

    private var grid: Array<FloatArray>? = null
    private var path: List<Pair<Int, Int>> = emptyList()
    private var start: Pair<Int, Int>? = null
    private var landmarks: List<SlamLandmark> = emptyList()
    private var currentGaze: GazePoint? = null
    private var isGlassesMode: Boolean = false

    private val cellPaint = Paint()
    private val pathPaint = Paint().apply {
        color = Color.CYAN
        strokeWidth = 8f
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
    }
    private val arrowPaint = Paint().apply {
        color = Color.YELLOW
        strokeWidth = 5f
        style = Paint.Style.STROKE
    }
    private val landmarkPaint = Paint().apply {
        color = Color.MAGENTA
        style = Paint.Style.FILL
    }
    private val startPaint = Paint().apply { color = Color.GREEN }

    private val gazeDotPaint = Paint().apply {
        color = Color.parseColor(AppConfig.PupilNeon.COLOR_GAZE_NEON)
        style = Paint.Style.FILL
        isAntiAlias = true
    }
    private val gazeRingPaint = Paint().apply {
        color = Color.parseColor(AppConfig.PupilNeon.COLOR_GAZE_RING)
        strokeWidth = 3f
        style = Paint.Style.STROKE
        isAntiAlias = true
    }
    private val gazeTextPaint = Paint().apply {
        color = Color.parseColor(AppConfig.PupilNeon.COLOR_GAZE_TEXT)
        textSize = 22f
        typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
        isAntiAlias = true
    }

    fun update(
        newGrid: Array<FloatArray>,
        newPath: List<Pair<Int, Int>>,
        newStart: Pair<Int, Int>,
        slamLandmarks: List<SlamLandmark> = emptyList(),
        gaze: GazePoint? = null,
        glassesActive: Boolean = false
    ) {
        grid = newGrid
        path = newPath
        start = newStart
        landmarks = slamLandmarks
        currentGaze = gaze
        isGlassesMode = glassesActive
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val g = grid ?: return
        val rows = g.size
        val cols = g[0].size
        val cellW = width.toFloat() / cols
        val cellH = height.toFloat() / rows
        val vmax = 2.5f

        for (r in 0 until rows) {
            for (c in 0 until cols) {
                cellPaint.color = coolwarm(g[r][c], vmax)
                canvas.drawRect(c * cellW, r * cellH, (c + 1) * cellW, (r + 1) * cellH, cellPaint)
            }
        }

        val near = AppConfig.CameraGeometry.GRID_NEAR_M
        val far = AppConfig.CameraGeometry.GRID_FAR_M
        val halfW = AppConfig.CameraGeometry.GRID_HALF_WIDTH_M

        for (lm in landmarks) {
            val r = ((far - lm.z) / (far - near) * (rows - 1)).roundToInt().coerceIn(0, rows - 1)
            val c = ((lm.x + halfW) / (2 * halfW) * (cols - 1)).roundToInt().coerceIn(0, cols - 1)

            val px = (c + 0.5f) * cellW
            val py = (r + 0.5f) * cellH

            canvas.drawCircle(px, py, 10f, landmarkPaint)

            if (abs(lm.vx) > 0.01f || abs(lm.vz) > 0.01f) {
                val vxPx = lm.vx * cellW * 3f
                val vzPx = -lm.vz * cellH * 3f
                canvas.drawLine(px, py, px + vxPx, py + vzPx, arrowPaint)
            }
        }

        if (isGlassesMode && currentGaze != null) {
            val gaze = currentGaze!!
            val gazeR = (gaze.y * (rows - 1)).roundToInt().coerceIn(0, rows - 1)
            val gazeC = (gaze.x * (cols - 1)).roundToInt().coerceIn(0, cols - 1)
            val px = (gazeC + 0.5f) * cellW
            val py = (gazeR + 0.5f) * cellH

            canvas.drawCircle(px, py, 16f, gazeRingPaint)
            canvas.drawCircle(px, py, 7f, gazeDotPaint)
            canvas.drawText("GAZE", px + 12f, py + 6f, gazeTextPaint)
        }

        if (path.size > 1) {
            val pts = FloatArray((path.size - 1) * 4)
            for (i in 0 until path.size - 1) {
                val (r0, c0) = path[i]
                val (r1, c1) = path[i + 1]
                pts[i * 4] = (c0 + 0.5f) * cellW
                pts[i * 4 + 1] = (r0 + 0.5f) * cellH
                pts[i * 4 + 2] = (c1 + 0.5f) * cellW
                pts[i * 4 + 3] = (r1 + 0.5f) * cellH
            }
            canvas.drawLines(pts, pathPaint)

            val last = path.last()
            val secondLast = path[path.size - 2]
            val px = (last.second + 0.5f) * cellW
            val py = (last.first + 0.5f) * cellH
            val prevPx = (secondLast.second + 0.5f) * cellW
            val prevPy = (secondLast.first + 0.5f) * cellH

            val angle = atan2(py - prevPy, px - prevPx)
            val arrowLength = 25f
            canvas.drawLine(
                px, py,
                px - arrowLength * cos(angle - 0.5f),
                py - arrowLength * sin(angle - 0.5f),
                pathPaint
            )
            canvas.drawLine(
                px, py,
                px - arrowLength * cos(angle + 0.5f),
                py - arrowLength * sin(angle + 0.5f),
                pathPaint
            )
        }

        start?.let { (r, c) ->
            canvas.drawCircle((c + 0.5f) * cellW, (r + 0.5f) * cellH, min(cellW, cellH) * 0.35f, startPaint)
        }
    }

    private fun coolwarm(value: Float, vmax: Float): Int {
        if (value.isNaN()) return Color.DKGRAY
        val t = (value / vmax).coerceIn(-1f, 1f)
        return if (t < 0) {
            val f = 1f + t
            Color.rgb((f * 255).toInt(), (f * 255).toInt(), 255)
        } else {
            val f = 1f - t
            Color.rgb(255, (f * 255).toInt(), (f * 255).toInt())
        }
    }
}
