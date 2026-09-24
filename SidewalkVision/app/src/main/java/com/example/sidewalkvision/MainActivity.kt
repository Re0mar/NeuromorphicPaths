package com.example.sidewalkvision

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.graphics.Matrix
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
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
                val imagePickerLauncher = rememberLauncherForActivityResult(
                    ActivityResultContracts.GetContent()
                ) { uri: Uri? ->
                    uri?.let { selectedUri ->
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
                            val detectionResult = pathDetector.detect(argbBitmap)
                            currentScreen = AppScreen.ImageResult(selectedUri, argbBitmap, detectionResult)
                        } catch (e: Exception) {
                            e.printStackTrace()
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
                        if (frameCount % 2 == 0) {
                            try {
                                val bitmap = imageProxy.toBitmap()
                                val rotatedBitmap = rotateBitmap(bitmap, imageProxy.imageInfo.rotationDegrees)
                                val result = pathDetector.detect(rotatedBitmap)
                                mainHandler.post {
                                    val oldMask = detectionResult?.maskBitmap
                                    detectionResult = result
                                    oldMask?.recycle()
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
                        cameraProvider.bindToLifecycle(
                            lifecycleOwner,
                            cameraSelector,
                            preview,
                            imageAnalysis
                        )
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
                }
            }

            Box(
                modifier = Modifier
                    .align(Alignment.TopCenter)
                    .padding(top = 48.dp)
            ) {
                val scoreText = if (res.score >= 0.001f) {
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
                }

                Box(
                    modifier = Modifier
                        .align(Alignment.TopCenter)
                        .padding(top = 16.dp)
                ) {
                    val scoreText = if (detectionResult.score >= 0.001f) {
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
            }
        }
    }
}

fun rotateBitmap(source: Bitmap, angle: Int): Bitmap {
    if (angle == 0) return source
    val matrix = Matrix().apply { postRotate(angle.toFloat()) }
    return Bitmap.createBitmap(source, 0, 0, source.width, source.height, matrix, true)
}
