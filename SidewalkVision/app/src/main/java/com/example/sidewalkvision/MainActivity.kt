package com.example.sidewalkvision

import android.Manifest
import android.content.pm.ApplicationInfo
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.graphics.Matrix
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.PhotoLibrary
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import kotlin.math.max
import kotlin.math.min
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.example.sidewalkvision.ui.theme.SidewalkVisionTheme
import java.util.Locale
import java.util.concurrent.Executors
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference

sealed interface AppScreen {
    object Menu : AppScreen
    object Camera : AppScreen
    data class ImageResult(val uri: Uri, val bitmap: Bitmap, val result: DetectionResult) : AppScreen
}

class MainActivity : ComponentActivity() {
    private lateinit var pathDetector: PathDetector

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        pathDetector = PathDetector(this)

        enableEdgeToEdge()
        setContent {
            SidewalkVisionTheme {
                var currentScreen by remember { mutableStateOf<AppScreen>(AppScreen.Menu) }
                var hasCameraPermission by remember {
                    mutableStateOf(
                        ContextCompat.checkSelfPermission(
                            this,
                            Manifest.permission.CAMERA
                        ) == PackageManager.PERMISSION_GRANTED
                    )
                }

                val permissionLauncher = rememberLauncherForActivityResult(
                    ActivityResultContracts.RequestPermission()
                ) { granted ->
                    hasCameraPermission = granted
                    if (granted) {
                        currentScreen = AppScreen.Camera
                    }
                }

                val context = LocalContext.current
                val coroutineScope = rememberCoroutineScope()
                var isAnalyzingImage by remember { mutableStateOf(false) }
                val imagePickerLauncher = rememberLauncherForActivityResult(
                    ActivityResultContracts.GetContent()
                ) { uri: Uri? ->
                    uri?.let { selectedUri ->
                        isAnalyzingImage = true
                        coroutineScope.launch {
                            // Decoding a full-size photo and running the model take long enough to
                            // freeze the screen, so both run off the main thread.
                            val analyzed = withContext(Dispatchers.Default) {
                                try {
                                    val source = ImageDecoder.createSource(context.contentResolver, selectedUri)
                                    val bitmap = ImageDecoder.decodeBitmap(source) { decoder, _, _ ->
                                        decoder.isMutableRequired = true
                                    }
                                    val argbBitmap = if (bitmap.config != Bitmap.Config.ARGB_8888) {
                                        bitmap.copy(Bitmap.Config.ARGB_8888, true)
                                    } else {
                                        bitmap
                                    }
                                    val intrinsics = photoIntrinsics(context.contentResolver, selectedUri, argbBitmap.width, argbBitmap.height)
                                    argbBitmap to pathDetector.detect(argbBitmap, intrinsics)
                                } catch (e: Exception) {
                                    e.printStackTrace()
                                    null
                                }
                            }
                            isAnalyzingImage = false
                            analyzed?.let { (argbBitmap, detectionResult) ->
                                currentScreen = AppScreen.ImageResult(selectedUri, argbBitmap, detectionResult)
                            }
                        }
                    }
                }

                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    Box(modifier = Modifier.padding(innerPadding).fillMaxSize()) {
                        when (val screen = currentScreen) {
                            is AppScreen.Menu -> {
                                MenuScreen(
                                    onSelectCamera = {
                                        if (hasCameraPermission) {
                                            currentScreen = AppScreen.Camera
                                        } else {
                                            permissionLauncher.launch(Manifest.permission.CAMERA)
                                        }
                                    },
                                    onSelectImage = {
                                        imagePickerLauncher.launch("image/*")
                                    }
                                )
                            }
                            is AppScreen.Camera -> {
                                CameraScreen(
                                    pathDetector = pathDetector,
                                    hasCameraPermission = hasCameraPermission,
                                    onRequestPermission = {
                                        permissionLauncher.launch(Manifest.permission.CAMERA)
                                    },
                                    onBack = {
                                        currentScreen = AppScreen.Menu
                                    }
                                )
                            }
                            is AppScreen.ImageResult -> {
                                ImageResultScreen(
                                    bitmap = screen.bitmap,
                                    detectionResult = screen.result,
                                    onPickAnother = {
                                        imagePickerLauncher.launch("image/*")
                                    },
                                    onBack = {
                                        currentScreen = AppScreen.Menu
                                    }
                                )
                            }
                        }
                        if (isAnalyzingImage) {
                            CircularProgressIndicator(modifier = Modifier.align(Alignment.Center))
                        }
                    }
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        pathDetector.close()
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MenuScreen(
    onSelectCamera: () -> Unit,
    onSelectImage: () -> Unit
) {
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("SidewalkVision") }
            )
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(24.dp),
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                text = "Choose Input Mode",
                style = MaterialTheme.typography.headlineMedium,
                textAlign = TextAlign.Center,
                modifier = Modifier.padding(bottom = 32.dp)
            )

            Button(
                onClick = onSelectCamera,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(56.dp)
            ) {
                Icon(
                    imageVector = Icons.Default.CameraAlt,
                    contentDescription = null,
                    modifier = Modifier.padding(end = 8.dp)
                )
                Text("Use Camera", fontSize = 16.sp)
            }

            Spacer(modifier = Modifier.height(16.dp))

            OutlinedButton(
                onClick = onSelectImage,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(56.dp)
            ) {
                Icon(
                    imageVector = Icons.Default.PhotoLibrary,
                    contentDescription = null,
                    modifier = Modifier.padding(end = 8.dp)
                )
                Text("Select Image from Gallery", fontSize = 16.sp)
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CameraScreen(
    pathDetector: PathDetector,
    hasCameraPermission: Boolean,
    onRequestPermission: () -> Unit,
    onBack: () -> Unit
) {
    if (!hasCameraPermission) {
        Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Text("Camera permission is required for live detection.", modifier = Modifier.padding(16.dp))
                Button(onClick = onRequestPermission) {
                    Text("Grant Camera Permission")
                }
                Spacer(modifier = Modifier.height(8.dp))
                TextButton(onClick = onBack) {
                    Text("Back to Menu")
                }
            }
        }
        return
    }

    val lifecycleOwner = LocalLifecycleOwner.current
    var detectionResult by remember { mutableStateOf<DetectionResult?>(null) }
    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }
    val mainHandler = remember { Handler(Looper.getMainLooper()) }

    val context = LocalContext.current
    // Written on the main thread once the camera is bound, read on the analysis thread.
    val intrinsics = remember { AtomicReference<CameraIntrinsics?>(null) }
    // Debug builds, the ones Android Studio installs, show the pose readout.
    val showPoseReadout = remember { (context.applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE) != 0 }
    val overlayHold = remember { OverlayHold() }
    val surpriseMonitor = remember { SurpriseMonitor() }
    var surprise by remember { mutableStateOf<SurpriseReading?>(null) }
    // The button's state for the screen, and a copy the analysis thread can read.
    var detectionEnabled by remember { mutableStateOf(true) }
    val detectionEnabledForAnalysis = remember { AtomicBoolean(true) }
    val groundSpeedTracker = remember { GroundSpeedTracker(context) }
    val groundSpeed by groundSpeedTracker.metersPerSecond.collectAsState()
    var hasLocationPermission by remember { mutableStateOf(groundSpeedTracker.hasPermission()) }
    val locationPermissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { grants ->
        hasLocationPermission = grants.values.any { it }
        if (hasLocationPermission) {
            groundSpeedTracker.start()
        }
    }

    DisposableEffect(Unit) {
        // Speed is optional. Without location permission the camera screen still works.
        if (!groundSpeedTracker.start()) {
            locationPermissionLauncher.launch(
                arrayOf(Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.ACCESS_COARSE_LOCATION)
            )
        }
        onDispose {
            groundSpeedTracker.stop()
            cameraExecutor.shutdown()
            detectionResult?.maskBitmap?.recycle()
        }
    }

    Box(modifier = Modifier.fillMaxSize()) {
        AndroidView(
            factory = { ctx ->
                val previewView = PreviewView(ctx).apply {
                    implementationMode = PreviewView.ImplementationMode.COMPATIBLE
                    scaleType = PreviewView.ScaleType.FILL_CENTER
                }
                val cameraProviderFuture = ProcessCameraProvider.getInstance(ctx)
                cameraProviderFuture.addListener({
                    val cameraProvider = cameraProviderFuture.get()
                    val preview = Preview.Builder().build().also {
                        it.surfaceProvider = previewView.surfaceProvider
                    }

                    val imageAnalysis = ImageAnalysis.Builder()
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build()

                    var frameCount = 0
                    imageAnalysis.setAnalyzer(cameraExecutor) { imageProxy ->
                        frameCount++
                        if (frameCount % 2 == 0 && detectionEnabledForAnalysis.get()) {
                            try {
                                val bitmap = imageProxy.toBitmap()
                                val rotatedBitmap = rotateBitmap(bitmap, imageProxy.imageInfo.rotationDegrees)
                                val result = pathDetector.detect(rotatedBitmap, intrinsics.get())
                                    .copy(captureTimeNanos = imageProxy.imageInfo.timestamp)
                                mainHandler.post {
                                    // Dropped if detection was switched off while this frame ran,
                                    // or if it's an empty result inside the hold after a path.
                                    // Every frame counts toward surprise, shown or not. Only a
                                    // reliable pose moves it, the rest only age the reading.
                                    if (detectionEnabledForAnalysis.get()) {
                                        result.captureTimeNanos?.let { time ->
                                            val position = result.pose?.takeIf { it.reliable }?.positionAcross
                                            surprise = surpriseMonitor.update(time, position)
                                        }
                                    }
                                    val hasPath = result.maskBitmap != null
                                    val show = detectionEnabledForAnalysis.get() &&
                                        overlayHold.shouldShow(hasPath, SystemClock.elapsedRealtime())
                                    if (show) {
                                        val oldMask = detectionResult?.maskBitmap
                                        detectionResult = result
                                        oldMask?.recycle()
                                    } else {
                                        result.maskBitmap?.recycle()
                                    }
                                }
                                if (rotatedBitmap != bitmap) {
                                    rotatedBitmap.recycle()
                                }
                                bitmap.recycle()
                            } catch (e: Exception) {
                                Log.e("CameraScreen", "Detection error", e)
                                e.printStackTrace()
                            }
                        }
                        imageProxy.close()
                    }

                    val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

                    try {
                        cameraProvider.unbindAll()
                        val camera = cameraProvider.bindToLifecycle(
                            lifecycleOwner,
                            cameraSelector,
                            preview,
                            imageAnalysis
                        )
                        intrinsics.set(cameraIntrinsics(camera))
                    } catch (e: Exception) {
                        e.printStackTrace()
                    }
                }, ContextCompat.getMainExecutor(ctx))
                previewView
            },
            modifier = Modifier.fillMaxSize()
        )

        // Overlay canvas and status text
        detectionResult?.let { res ->
            val box = res.box
            val mask = res.maskBitmap

            Canvas(modifier = Modifier.fillMaxSize()) {
                val canvasWidth = size.width
                val canvasHeight = size.height

                if (mask != null) {
                    val srcWidth = mask.width.toFloat()
                    val srcHeight = mask.height.toFloat()
                    val scale = max(canvasWidth / srcWidth, canvasHeight / srcHeight)
                    val scaledWidth = srcWidth * scale
                    val scaledHeight = srcHeight * scale
                    val left = (canvasWidth - scaledWidth) / 2f
                    val top = (canvasHeight - scaledHeight) / 2f

                    drawImage(
                        image = mask.asImageBitmap(),
                        dstOffset = IntOffset(left.toInt(), top.toInt()),
                        dstSize = IntSize(scaledWidth.toInt(), scaledHeight.toInt())
                    )

                    box?.let { b ->
                        val boxLeft = left + b[0] * scaledWidth
                        val boxTop = top + b[1] * scaledHeight
                        val boxRight = left + b[2] * scaledWidth
                        val boxBottom = top + b[3] * scaledHeight

                        drawRect(
                            color = Color.Yellow,
                            topLeft = Offset(boxLeft, boxTop),
                            size = Size(boxRight - boxLeft, boxBottom - boxTop),
                            style = Stroke(width = 10f)
                        )
                    }

                    if (showPoseReadout) {
                        res.pose?.let { pose ->
                            drawPoseWorking(pose, FrameOnScreen(mask.width, mask.height, left, top, scaledWidth, scaledHeight))
                        }
                    }
                }
            }

            Box(
                modifier = Modifier
                    .align(Alignment.TopCenter)
                    .padding(top = 48.dp)
            ) {
                val scoreText = if (res.score >= PathDetector.CONFIDENCE_THRESHOLD) {
                    "Walkway Detected: ${(res.score * 100).toInt()}%"
                } else {
                    "No Walkway (${String.format(Locale.US, "%.2f", res.score)})"
                }
                Surface(
                    color = Color.Black.copy(alpha = 0.6f),
                    shape = MaterialTheme.shapes.medium
                ) {
                    Text(
                        text = scoreText,
                        color = Color.White,
                        fontSize = 18.sp,
                        modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp)
                    )
                }
            }
        }

        Surface(
            color = Color.Black.copy(alpha = 0.6f),
            shape = MaterialTheme.shapes.medium,
            modifier = Modifier
                .align(Alignment.TopEnd)
                .padding(16.dp)
        ) {
            val speed = groundSpeed
            val speedText = when {
                !hasLocationPermission -> "Speed: no location permission"
                speed == null -> "Speed: waiting for GPS"
                else -> "Speed: ${String.format(Locale.US, "%.2f", speed)} m/s"
            }
            Text(
                text = speedText,
                color = Color.White,
                fontSize = 14.sp,
                modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp)
            )
        }

        if (surprise?.alarmOn == true) {
            EdgeAlarm(
                reading = surprise!!,
                modifier = Modifier
                    .align(Alignment.TopCenter)
                    .padding(top = 104.dp)
            )
        }

        if (showPoseReadout) {
            Column(
                verticalArrangement = Arrangement.spacedBy(8.dp),
                modifier = Modifier
                    .align(Alignment.BottomStart)
                    .padding(16.dp)
            ) {
                surprise?.let { reading ->
                    SurpriseReadout(reading, detectionResult?.pose, groundSpeed)
                }
                detectionResult?.pose?.let { pose -> PoseReadout(pose = pose) }
            }
        }

        Button(
            onClick = {
                detectionEnabled = !detectionEnabled
                detectionEnabledForAnalysis.set(detectionEnabled)
                if (!detectionEnabled) {
                    detectionResult?.maskBitmap?.recycle()
                    detectionResult = null
                    surpriseMonitor.reset()
                    surprise = null
                    overlayHold.reset()
                }
            },
            modifier = Modifier
                .align(Alignment.BottomEnd)
                .padding(16.dp)
        ) {
            Text(if (detectionEnabled) "Detection on" else "Detection off")
        }

        // Back button
        IconButton(
            onClick = onBack,
            modifier = Modifier
                .padding(16.dp)
                .align(Alignment.TopStart)
        ) {
            Surface(
                shape = MaterialTheme.shapes.small,
                color = Color.Black.copy(alpha = 0.6f)
            ) {
                Icon(
                    imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                    contentDescription = "Back",
                    tint = Color.White,
                    modifier = Modifier.padding(8.dp)
                )
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ImageResultScreen(
    bitmap: Bitmap,
    detectionResult: DetectionResult,
    onPickAnother: () -> Unit,
    onBack: () -> Unit
) {
    val context = LocalContext.current
    // Debug builds show how the pose was worked out, as on the camera screen.
    val showPoseWorking = remember { (context.applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE) != 0 }
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Image Detection Result") },
                navigationIcon = {
                    IconButton(onClick = {
                        bitmap.recycle()
                        detectionResult.maskBitmap?.recycle()
                        onBack()
                    }) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back")
                    }
                }
            )
        },
        bottomBar = {
            Surface(
                tonalElevation = 3.dp,
                modifier = Modifier.fillMaxWidth()
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(16.dp),
                    horizontalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    OutlinedButton(
                        onClick = {
                            bitmap.recycle()
                            detectionResult.maskBitmap?.recycle()
                            onPickAnother()
                        },
                        modifier = Modifier.weight(1f)
                    ) {
                        Text("Pick Another")
                    }
                    Button(
                        onClick = {
                            bitmap.recycle()
                            detectionResult.maskBitmap?.recycle()
                            onBack()
                        },
                        modifier = Modifier.weight(1f)
                    ) {
                        Text("Main Menu")
                    }
                }
            }
        }
    ) { padding ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding),
            contentAlignment = Alignment.Center
        ) {
            Box(
                modifier = Modifier.fillMaxSize()
            ) {
                Image(
                    bitmap = bitmap.asImageBitmap(),
                    contentDescription = "Selected Image",
                    modifier = Modifier.fillMaxSize()
                )

                val box = detectionResult.box
                val mask = detectionResult.maskBitmap
                val srcWidth = bitmap.width.toFloat()
                val srcHeight = bitmap.height.toFloat()

                Canvas(modifier = Modifier.fillMaxSize()) {
                    val canvasWidth = size.width
                    val canvasHeight = size.height

                    val scale = min(canvasWidth / srcWidth, canvasHeight / srcHeight)
                    val scaledWidth = srcWidth * scale
                    val scaledHeight = srcHeight * scale
                    val left = (canvasWidth - scaledWidth) / 2f
                    val top = (canvasHeight - scaledHeight) / 2f

                    mask?.let {
                        drawImage(
                            image = it.asImageBitmap(),
                            dstOffset = IntOffset(left.toInt(), top.toInt()),
                            dstSize = IntSize(scaledWidth.toInt(), scaledHeight.toInt())
                        )
                    }

                    box?.let { b ->
                        val boxLeft = left + b[0] * scaledWidth
                        val boxTop = top + b[1] * scaledHeight
                        val boxRight = left + b[2] * scaledWidth
                        val boxBottom = top + b[3] * scaledHeight

                        drawRect(
                            color = Color.Yellow,
                            topLeft = Offset(boxLeft, boxTop),
                            size = Size(boxRight - boxLeft, boxBottom - boxTop),
                            style = Stroke(width = 10f)
                        )
                    }

                    if (showPoseWorking && mask != null) {
                        detectionResult.pose?.let { pose ->
                            drawPoseWorking(pose, FrameOnScreen(mask.width, mask.height, left, top, scaledWidth, scaledHeight))
                        }
                    }
                }

                Box(
                    modifier = Modifier
                        .align(Alignment.TopCenter)
                        .padding(top = 16.dp)
                ) {
                    val scoreText = if (detectionResult.score >= PathDetector.CONFIDENCE_THRESHOLD) {
                        "Walkway Detected: ${(detectionResult.score * 100).toInt()}%"
                    } else {
                        "No Walkway (${String.format(Locale.US, "%.2f", detectionResult.score)})"
                    }
                    Surface(
                        color = Color.Black.copy(alpha = 0.6f),
                        shape = MaterialTheme.shapes.medium
                    ) {
                        Text(
                            text = scoreText,
                            color = Color.White,
                            fontSize = 18.sp,
                            modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp)
                        )
                    }
                }

                if (showPoseWorking) {
                    detectionResult.pose?.let { pose ->
                        PoseReadout(
                            pose = pose,
                            modifier = Modifier
                                .align(Alignment.BottomStart)
                                .padding(16.dp)
                        )
                    }
                }
            }
        }
    }
}

fun rotateBitmap(source: Bitmap, angle: Int): Bitmap {
    if (angle == 0) return source
    val matrix = Matrix().apply { postRotate(angle.toFloat()) }
    return Bitmap.createBitmap(source, 0, 0, source.width, source.height, matrix, true)
}

/** Debug readout of the camera pose estimated from the sidewalk's edges. */
@Composable
fun PoseReadout(pose: PoseEstimate, modifier: Modifier = Modifier) {
    fun format(value: Double?, pattern: String): String =
        value?.let { String.format(Locale.US, pattern, it) } ?: "n/a"

    val lines = buildList {
        add("pitch ${format(pose.pitchDegrees, "%.1f")}°   heading ${format(pose.headingDegrees, "%.1f")}°")
        add("height ${format(pose.cameraHeightMeters, "%.2f")} m, for a ${DEFAULT_PATH_WIDTH_METERS} m wide path")
        add("position across ${format(pose.positionAcross, "%.2f")}   focal ${format(pose.focalLengthPx, "%.0f")} px")
        add(
            when {
                pose.status != PoseStatus.OK -> "no pose: ${pose.status.description}"
                pose.reliable -> "reliable"
                else -> "UNRELIABLE: ${pose.unreliableReason}"
            }
        )
    }
    Surface(
        color = Color.Black.copy(alpha = 0.6f),
        shape = MaterialTheme.shapes.medium,
        modifier = modifier
    ) {
        Column(modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp)) {
            for (line in lines) {
                Text(text = line, color = Color.White, fontSize = 12.sp)
            }
        }
    }
}

/** The warning itself: shown in every build while edge surprise is over the alarm threshold. */
@Composable
fun EdgeAlarm(reading: SurpriseReading, modifier: Modifier = Modifier) {
    val seconds = reading.secondsToEdge
    val text = if (seconds == null || seconds <= 0.0) {
        "Edge of the path"
    } else {
        "Edge in ${String.format(Locale.US, "%.1f", seconds)} s"
    }
    Surface(
        color = Color(0xFFD32F2F),
        shape = MaterialTheme.shapes.medium,
        modifier = modifier
    ) {
        Text(
            text = text,
            color = Color.White,
            fontSize = 22.sp,
            modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp)
        )
    }
}

/**
 * Debug readout of both surprise channels. Sideways speed is shown twice: from the camera alone,
 * which the channels use, and from GPS speed and heading, v = -u sin(heading), as a cross-check.
 */
@Composable
fun SurpriseReadout(reading: SurpriseReading, pose: PoseEstimate?, groundSpeedMetersPerSecond: Double?) {
    fun format(value: Double?, pattern: String): String =
        value?.let { if (it.isInfinite()) "past" else String.format(Locale.US, pattern, it) } ?: "n/a"

    val headingDegrees = pose?.takeIf { it.reliable }?.headingDegrees
    val pathWidthMeters = pose?.pathWidthMeters ?: DEFAULT_PATH_WIDTH_METERS
    val headingSpeed = if (headingDegrees != null && groundSpeedMetersPerSecond != null) {
        -groundSpeedMetersPerSecond / pathWidthMeters * kotlin.math.sin(Math.toRadians(headingDegrees))
    } else {
        null
    }
    val lines = listOf(
        "line ${format(reading.lineSurpriseBits, "%.2f")} bits   edge ${format(reading.edgeSurpriseBits, "%.2f")} bits   " +
            "alarm ${if (reading.alarmOn) "ON" else "off"}",
        "to edge ${format(reading.secondsToEdge, "%.1f")} s   gap ${format(reading.gapToEdge, "%.2f")} widths",
        "sideways ${format(reading.sidewaysSpeed, "%.2f")} widths/s camera, ${format(headingSpeed, "%.2f")} GPS",
        "position ${format(reading.positionFromCenter, "%.2f")}   mean ${format(reading.meanPosition, "%.2f")}   " +
            "wobble ${format(reading.wobble, "%.3f")}",
    )
    Surface(
        color = Color.Black.copy(alpha = 0.6f),
        shape = MaterialTheme.shapes.medium
    ) {
        Column(modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp)) {
            for (line in lines) {
                Text(text = line, color = Color.White, fontSize = 12.sp)
            }
        }
    }
}
