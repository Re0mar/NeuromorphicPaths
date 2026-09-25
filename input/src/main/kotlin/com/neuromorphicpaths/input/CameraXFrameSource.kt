package com.neuromorphicpaths.input

import android.content.Context
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.google.common.util.concurrent.ListenableFuture
import com.neuromorphicpaths.core.CameraIntrinsics
import com.neuromorphicpaths.core.Frame
import com.neuromorphicpaths.core.FrameSource
import com.neuromorphicpaths.core.PoseProvider
import java.util.concurrent.ExecutionException
import java.util.concurrent.Executor
import java.util.concurrent.Executors
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext

/**
 * Streams the back camera through CameraX image analysis.
 *
 * Frames arrive as RGBA at the analysis resolution, rotated upright, stamped with whatever the
 * pose provider says at that moment. The field of view comes from the camera itself once it is
 * bound, and falls back to [fallbackIntrinsics] on a camera that does not report one. CameraX
 * keeps only the latest image while the analyzer is busy, which matches what the pipeline wants.
 */
class CameraXFrameSource(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner,
    private val poseProvider: PoseProvider,
    private val fallbackIntrinsics: CameraIntrinsics,
) : FrameSource {

    override val name: String = "camera"

    override fun frames(): Flow<Frame> = callbackFlow {
        val mainExecutor = ContextCompat.getMainExecutor(context)
        val analysisExecutor = Executors.newSingleThreadExecutor()
        val provider = ProcessCameraProvider.getInstance(context).await(mainExecutor)
        val analysis = ImageAnalysis.Builder()
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
        // CameraX insists on binding from the main thread. Binding first also gives the camera
        // info the field of view is read from, before the first frame needs it.
        val camera = withContext(Dispatchers.Main) {
            provider.unbindAll()
            provider.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, analysis)
        }
        val rotationDegrees = camera.cameraInfo.getSensorRotationDegrees(analysis.targetRotation)
        val intrinsics = horizontalFieldOfView(camera.cameraInfo, rotationDegrees) ?: fallbackIntrinsics
        analysis.setAnalyzer(analysisExecutor) { image ->
            image.use { proxy ->
                val frame = proxy.toRgbaImage().toFrame(proxy.imageInfo.timestamp, poseProvider.currentPose(), intrinsics)
                trySend(frame)
            }
        }
        awaitClose {
            mainExecutor.execute { provider.unbind(analysis) }
            analysisExecutor.shutdown()
        }
    }

    override fun close() = Unit

    private suspend fun <T> ListenableFuture<T>.await(executor: Executor): T =
        suspendCancellableCoroutine { continuation ->
            addListener({
                try {
                    continuation.resume(get())
                } catch (failure: ExecutionException) {
                    // The future wraps the real cause. Hand the cause on so the caller sees what CameraX said.
                    continuation.resumeWithException(failure.cause ?: failure)
                }
            }, executor)
            continuation.invokeOnCancellation { cancel(true) }
        }
}
