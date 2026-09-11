package com.example.neuromorphicpaths.pipeline

import android.content.Context
import android.graphics.Bitmap
import com.example.neuromorphicpaths.device.CameraStreamManager
import com.example.neuromorphicpaths.geometry.GroundProjector
import com.example.neuromorphicpaths.geometry.HomographyEstimator
import com.example.neuromorphicpaths.steering.DirectionDebouncer
import com.example.neuromorphicpaths.steering.HeadingSmoother
import com.example.neuromorphicpaths.steering.SteeringController
import com.example.neuromorphicpaths.vision.edges.EdgeExtractor
import com.example.neuromorphicpaths.vision.segmentation.SidewalkSegmenter
import com.meta.wearable.dat.camera.types.VideoFrame
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.android.Utils
import org.opencv.core.Mat

class FramePipeline(
    private val scope: CoroutineScope,
    private val cameraStreamManager: CameraStreamManager,
    context: Context
) {
    private val segmenter = SidewalkSegmenter(context)
    private val edgeExtractor = EdgeExtractor()
    private val homographyEstimator = HomographyEstimator()
    private val steeringController = SteeringController()
    private val headingSmoother = HeadingSmoother()
    private val directionDebouncer = DirectionDebouncer()
    
    private var groundProjector: GroundProjector? = null
    private var currentHomography: Mat? = null

    private val frameChannel = Channel<VideoFrame>(capacity = Channel.CONFLATED)
    
    private val _processedFrames = MutableSharedFlow<PipelineFrame>()
    val processedFrames: SharedFlow<PipelineFrame> = _processedFrames.asSharedFlow()

    init {
        scope.launch {
            cameraStreamManager.frames.collect { frame ->
                frameChannel.trySend(frame)
            }
        }

        scope.launch {
            for (frame in frameChannel) {
                val pipelineFrame = processFrame(frame)
                _processedFrames.emit(pipelineFrame)
            }
        }
    }

    private suspend fun processFrame(frame: VideoFrame): PipelineFrame = withContext(Dispatchers.Default) {
        // Initialize homography if not yet done
        if (currentHomography == null) {
            currentHomography = homographyEstimator.computeHomography(frame.width, frame.height)
            groundProjector = GroundProjector(currentHomography!!)
        }

        // TODO: Implement actual conversion from VideoFrame (ByteBuffer) to Bitmap
        // and downscaling for optimized processing.
        
        // Simple passthrough placeholder:
        val bitmap = Bitmap.createBitmap(frame.width, frame.height, Bitmap.Config.ARGB_8888)
        
        val mask = segmenter.segment(bitmap)
        val maskBitmap = Bitmap.createBitmap(mask.cols(), mask.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(mask, maskBitmap)
        
        val (left, right) = edgeExtractor.extractEdges(mask)
        mask.release()

        val sidewalkPosition = groundProjector?.computePosition(left, right, frame.height)
        val steeringResult = sidewalkPosition?.let { steeringController.computeSteering(it) }

        val timestampMs = frame.presentationTimeUs / 1000
        val smoothedHeading = steeringResult?.let { headingSmoother.smooth(it.recommendedHeadingDeg) }
        val stabilizedCommand = smoothedHeading?.let {
            val rawCmd = steeringController.mapToCommand(it)
            directionDebouncer.debounce(rawCmd, timestampMs)
        }

        PipelineFrame(
            bitmap = bitmap,
            segmentationMask = maskBitmap,
            timestampMs = timestampMs,
            rawWidth = frame.width,
            rawHeight = frame.height,
            leftEdge = left,
            rightEdge = right,
            sidewalkPosition = sidewalkPosition,
            steeringResult = steeringResult,
            smoothedHeadingDeg = smoothedHeading,
            stabilizedCommand = stabilizedCommand
        )
    }
}
