/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

// CameraViewModel - DAT camera lifecycle, capture, and recording
//
// Drives the camera screen by exercising the SDK's camera lifecycle as explicit steps: create and
// start a DeviceSession, add and start a Stream, capture a photo or record video, stop the stream,
// end the session. Owns the screen-scoped pieces — the DeviceSession, its Stream, the on-device
// HEVC preview decoder, and the passthrough video recorder. The UI binds to the SDK's own
// DeviceSessionState / StreamState so the sample shows the real state machine.

package com.example.neuromorphicpaths.camera

import android.app.Application
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.Image
import android.media.ImageReader
import android.os.Handler
import android.os.HandlerThread
import android.os.IBinder
import android.os.Looper
import android.util.Log
import android.view.Surface
import androidx.exifinterface.media.ExifInterface
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.meta.wearable.dat.camera.Camera
import com.meta.wearable.dat.camera.Stream
import com.meta.wearable.dat.camera.addCamera
import com.meta.wearable.dat.camera.types.PhotoData
import com.meta.wearable.dat.camera.types.StreamConfiguration
import com.meta.wearable.dat.camera.types.StreamState
import com.meta.wearable.dat.camera.types.VideoFrame
import com.meta.wearable.dat.camera.types.VideoQuality
import com.meta.wearable.dat.core.Wearables
import com.meta.wearable.dat.core.selectors.DeviceSelector
import com.meta.wearable.dat.core.session.DeviceSession
import com.meta.wearable.dat.core.session.DeviceSessionState
import com.meta.wearable.dat.core.types.Permission
import com.meta.wearable.dat.core.types.PermissionStatus
import com.example.neuromorphicpaths.R
import com.example.neuromorphicpaths.stream.AudioInputHandler
import com.example.neuromorphicpaths.stream.HevcDecoder
import com.example.neuromorphicpaths.stream.HevcParameterSetCollector
import com.example.neuromorphicpaths.stream.RecordingResult
import com.example.neuromorphicpaths.stream.StreamingService
import com.example.neuromorphicpaths.stream.VideoRecorder
import com.example.neuromorphicpaths.vision.PathAnalysisService
import com.example.neuromorphicpaths.wearables.WearablesViewModel
import java.io.ByteArrayInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.util.concurrent.atomic.AtomicBoolean
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream
import java.io.Closeable

class CameraViewModel(
  application: Application,
  private val wearablesViewModel: WearablesViewModel,
) : AndroidViewModel(application) {

  companion object {
    private const val TAG = "CameraAccess:CameraViewModel"
    private const val FRAME_RATE = 4
    private const val KEYFRAME_WAIT_STEP_MS = 25L
    private const val KEYFRAME_WAIT_MAX_MS = 500L

    // Quality 100 makes the JPEG round-trip (YUV -> JPEG -> Bitmap) much slower for no visible gain.
    private const val JPEG_QUALITY = 85

    // Must be > 2: while we convert one frame the codec still needs free output buffers, otherwise
    // the decoder stalls, drops input, and HEVC shows corruption/flashing until the next keyframe.
    private const val IMAGE_READER_BUFFERS = 4

    // How long the last detected path stays on screen after the detector returns "nothing", so a
    // single empty/noisy detection doesn't make the overlay blink.
    private const val OVERLAY_HOLD_MS = 400L
  }

  private val deviceSelector: DeviceSelector = wearablesViewModel.deviceSelector

  private val _uiState = MutableStateFlow(CameraUiState())
  val uiState: StateFlow<CameraUiState> = _uiState.asStateFlow()

  private var session: DeviceSession? = null
  private var camera: Camera? = null
  private var stream: Stream? = null

  // Recording pieces. The single compressed-HEVC stream feeds both the on-screen decoder and the
  // passthrough MP4 writer.
  private val audioInputHandler = AudioInputHandler(application)
  private val videoRecorder = VideoRecorder(application, viewModelScope)

  // Per-frame work (byte copy, NAL parsing, MediaMuxer writes, decoder feed) runs at frame rate and
  // must stay off the main thread. A single-threaded dispatcher keeps frames serialized so the
  // MediaMuxer/MediaCodec see in-order calls from one consistent thread.
  private val frameDispatcher = Dispatchers.Default.limitedParallelism(1)

  // Path analysis (TFLite inference) runs here, never on the thread that draws frames, so a slow
  // inference can't delay or skip the on-screen video.
  private val analysisDispatcher = Dispatchers.Default.limitedParallelism(1)
  private val analysisInFlight = AtomicBoolean(false)

  // Guards decoder create/teardown so a frame can't bind a new decoder to a Surface that
  // setSurface(null) just released — the @Volatile refs alone can't fix that check-then-act.
  private val decoderLock = Any()

  // @Volatile: single refs shared by the frame-collector and main threads, nulled at teardown.
  @Volatile private var hevcDecoder: HevcDecoder? = null
  @Volatile private var analysisDecoder: HevcDecoder? = null
  @Volatile private var decoderSurface: Surface? = null
  @Volatile private var targetSurface: Surface? = null
  private var streamWidth: Int = 0
  private var streamHeight: Int = 0
  private var analysisImageReader: ImageReader? = null
  private var analysisThread: HandlerThread? = null
  private var analysisHandler: Handler? = null
  private var pathAnalysisService: PathAnalysisService? = null
  private var pathResultJob: Job? = null
  private var overlayClearJob: Job? = null
  private val pathAnalysisConnection = object : ServiceConnection {
    override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
      val binder = service as PathAnalysisService.LocalBinder
      pathAnalysisService = binder.getService()
      observePathAnalysisResults()
    }

    override fun onServiceDisconnected(name: ComponentName?) {
      pathAnalysisService = null
    }
  }

  // Accumulates the stream's HEVC parameter sets (VPS/SPS/PPS) so a recording (or decoder) started
  // mid-stream can be primed with a complete format. The SDK emits the VPS once at stream start, so
  // a partial set yields an unfinalizable file ("Missing codec specific data"). Internally
  // synchronized.
  private val csdCollector = HevcParameterSetCollector()

  private var sessionStateJob: Job? = null
  private var sessionErrorJob: Job? = null
  private var videoJob: Job? = null
  private var streamStateJob: Job? = null
  private var streamErrorJob: Job? = null

  init {
    bindPathAnalysis()
    videoRecorder.setAudioInputHandler(audioInputHandler)

    // Mirror the recorder's intent/elapsed into UI state.
    viewModelScope.launch {
      videoRecorder.isRecording.collect { recording ->
        _uiState.update { it.copy(isRecording = recording) }
      }
    }
    viewModelScope.launch {
      videoRecorder.recordingElapsedSeconds.collect { seconds ->
        _uiState.update { it.copy(recordingElapsedSeconds = seconds) }
      }
    }
    // Stop a recording gracefully if the mic is interrupted (e.g. a phone call).
    viewModelScope.launch {
      audioInputHandler.wasInterrupted.collect { interrupted ->
        if (interrupted && _uiState.value.isRecording) {
          Log.w(TAG, "Audio interrupted — stopping recording")
          stopVideoRecording()
        }
      }
    }
  }

  fun toggleVisionMode() {
    _uiState.update {
      val enabled = !it.isVisionEnabled
      // Clear the overlay once when turning vision off (instead of on every frame).
      it.copy(
        isVisionEnabled = enabled,
        pathBoundaries = if (enabled) it.pathBoundaries else emptyList(),
        topDetectionBox = if (enabled) it.topDetectionBox else null,
        topScore = if (enabled) it.topScore else 0f,
        visionDebugInfo = if (enabled) it.visionDebugInfo else "",
      )
    }

    if (!_uiState.value.isVisionEnabled) {
      synchronized(decoderLock) {
        analysisDecoder?.stop()
        analysisDecoder = null
        analysisImageReader?.close()
        analysisImageReader = null
        analysisThread?.quit()
        analysisThread = null
        analysisHandler = null
      }
    }
  }

  // MARK: - Surface

  fun setSurface(surface: Surface?) {
    Log.d(TAG, "setSurface: $surface")
    synchronized(decoderLock) {
      targetSurface = surface
      if (surface == null) {
        hevcDecoder?.stop()
        hevcDecoder = null
        analysisDecoder?.stop()
        analysisDecoder = null
        analysisImageReader?.close()
        analysisImageReader = null
        analysisThread?.quit()
        analysisThread = null
        analysisHandler = null
        unbindPathAnalysis()
      } else {
        bindPathAnalysis()
      }
    }
  }

  private fun bindPathAnalysis() {
    val intent = Intent(getApplication(), PathAnalysisService::class.java)
    getApplication<Application>().bindService(intent, pathAnalysisConnection, Context.BIND_AUTO_CREATE)
  }

  private fun unbindPathAnalysis() {
    pathResultJob?.cancel()
    pathResultJob = null
    overlayClearJob?.cancel()
    overlayClearJob = null
    if (pathAnalysisService != null) {
      getApplication<Application>().unbindService(pathAnalysisConnection)
      pathAnalysisService = null
    }
  }

  private fun observePathAnalysisResults() {
    val service = pathAnalysisService ?: return
    // Cancel any previous collector so reconnecting doesn't leave duplicate collectors running.
    pathResultJob?.cancel()
    pathResultJob = viewModelScope.launch {
      service.detectionResult.collect { result ->
        if (!_uiState.value.isVisionEnabled) return@collect

        val boundaries = result?.boundaries ?: emptyList()
        val topBox = result?.topBox
        val topScore = result?.topScore ?: 0f
        val debugInfo = result?.debugInfo ?: ""
        Log.d(
          TAG,
          "path result: polygons=${boundaries.size} pts=${boundaries.firstOrNull()?.size ?: 0} " +
                  "box=$topBox score=$topScore"
        )

        if (boundaries.isNotEmpty()) {
          overlayClearJob?.cancel()
          overlayClearJob = null
          _uiState.update {
            it.copy(
              pathBoundaries = boundaries,
              topDetectionBox = topBox,
              topScore = topScore,
              visionDebugInfo = debugInfo,
            )
          }
        } else {
          _uiState.update {
            it.copy(
              topDetectionBox = topBox,
              topScore = topScore,
              visionDebugInfo = debugInfo,
            )
          }
          if (overlayClearJob == null) {
            overlayClearJob = viewModelScope.launch {
              delay(OVERLAY_HOLD_MS)
              _uiState.update { it.copy(pathBoundaries = emptyList(), topDetectionBox = null) }
              overlayClearJob = null
            }
          }
        }
      }
    }
  }

  private fun yuv420ToNv21(image: Image): ByteArray {
    val width = image.width
    val height = image.height
    val nv21 = ByteArray(width * height * 3 / 2)

    val yPlane = image.planes[0]
    val uPlane = image.planes[1]
    val vPlane = image.planes[2]

    val yBuffer = yPlane.buffer
    val uBuffer = uPlane.buffer
    val vBuffer = vPlane.buffer

    val yRowStride = yPlane.rowStride
    var nvIndex = 0
    for (row in 0 until height) {
      yBuffer.position(row * yRowStride)
      yBuffer.get(nv21, nvIndex, width)
      nvIndex += width
    }

    val vRowStride = vPlane.rowStride
    val uRowStride = uPlane.rowStride
    val vPixelStride = vPlane.pixelStride
    val uPixelStride = uPlane.pixelStride

    nvIndex = width * height
    for (row in 0 until height / 2) {
      for (col in 0 until width / 2) {
        val vOffset = row * vRowStride + col * vPixelStride
        val uOffset = row * uRowStride + col * uPixelStride

        nv21[nvIndex++] = vBuffer.get(vOffset)
        nv21[nvIndex++] = uBuffer.get(uOffset)
      }
    }

    return nv21
  }

  private fun nv21ToBitmap(nv21: ByteArray, width: Int, height: Int): Bitmap? {
    val out = ByteArrayOutputStream(width * height / 4)
    YuvImage(nv21, ImageFormat.NV21, width, height, null)
      .compressToJpeg(Rect(0, 0, width, height), JPEG_QUALITY, out)
    val bytes = out.toByteArray()
    return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
  }

  private fun setupImageReader(width: Int, height: Int) {
    Log.d(TAG, "setupImageReader: ${width}x${height}")
    analysisImageReader?.close()
    analysisThread?.quit()

    val thread = HandlerThread("PathAnalysisThread").also { it.start() }
    analysisThread = thread
    val handler = Handler(thread.looper)
    analysisHandler = handler

    // Using YUV_420_888 format which is universally supported by MediaCodec video decoders on physical devices
    try {
      val reader =
        ImageReader.newInstance(width, height, ImageFormat.YUV_420_888, IMAGE_READER_BUFFERS)
      reader.setOnImageAvailableListener({ r ->
        val image = r.acquireLatestImage() ?: return@setOnImageAvailableListener

        // Copy the pixels out and hand the Image straight back to the codec. Holding it during
        // the slow JPEG/Bitmap work below starves MediaCodec of output buffers.
        val nv21 = try {
          yuv420ToNv21(image)
        } catch (e: Exception) {
          Log.e(TAG, "Error converting YUV frame: ${e.message}", e)
          return@setOnImageAvailableListener
        } finally {
          image.close()
        }

        try {
          val bitmap = nv21ToBitmap(nv21, width, height) ?: return@setOnImageAvailableListener

          // Hand a copy to the detector (asynchronously, dropping frames while it's still busy).
          submitForAnalysis(bitmap)
        } catch (e: Exception) {
          Log.e(TAG, "Error processing frame: ${e.message}", e)
        }
      }, handler)

      analysisImageReader = reader
      decoderSurface = reader.surface
      Log.d(TAG, "ImageReader setup successful, surface: $decoderSurface")
    } catch (e: Exception) {
      Log.e(TAG, "Failed to create ImageReader: ${e.message}", e)
    }
  }

  /**
   * Sends a frame to the path detector without ever blocking the draw thread. If the previous
   * analysis is still running the frame is simply skipped. The detector gets its own copy of the
   * bitmap, so it can recycle/modify it freely without breaking the frame we're drawing.
   */
  private fun submitForAnalysis(bitmap: Bitmap) {
    if (!_uiState.value.isVisionEnabled) return
    val service = pathAnalysisService ?: return
    if (!analysisInFlight.compareAndSet(false, true)) return

    val copy = bitmap.copy(Bitmap.Config.ARGB_8888, false)
    if (copy == null) {
      analysisInFlight.set(false)
      return
    }
    viewModelScope.launch(analysisDispatcher) {
      try {
        service.processFrame(copy)
      } catch (e: Exception) {
        Log.e(TAG, "Path analysis failed: ${e.message}", e)
      } finally {
        analysisInFlight.set(false)
      }
    }
  }

  // MARK: - Lifecycle step 1: session

  /** Creates and starts a [DeviceSession] (no stream yet). */
  fun startSession() {
    if (_uiState.value.hasSession) return
    Wearables.createSession(deviceSelector)
      .onSuccess { created ->
        session = created
        // Subscribe before start() so no initial transitions are missed.
        observeSession(created)
        _uiState.update { it.copy(sessionState = DeviceSessionState.STARTING) }
        created.start()
      }
      .onFailure { error, _ ->
        Log.e(TAG, "Failed to start session: ${error.description}")
        wearablesViewModel.setRecentError(error.getLocalizedDescription(getApplication()))
        cleanupSession()
      }
  }

  /**
   * Ends the device session. The stream and any in-progress recording are not torn down here
   * directly — stopping the session drives the SDK's stream to a terminal state, which the
   * stream-state collector observes (see [onStreamTerminated]) to stop recording and release stream
   * resources.
   */
  fun endSession() {
    val current = session ?: return
    _uiState.update { it.copy(sessionState = DeviceSessionState.STOPPING) }
    current.stop()
  }

  private fun observeSession(session: DeviceSession) {
    sessionStateJob = viewModelScope.launch {
      session.state.collect { state ->
        _uiState.update { it.copy(sessionState = state) }
        if (state == DeviceSessionState.STOPPED) {
          cleanupSession()
        }
      }
    }
    sessionErrorJob = viewModelScope.launch {
      session.errors.collect { error ->
        // All session errors surface through the snackbar, including
        // DAT_APP_ON_THE_GLASSES_UPDATE_REQUIRED, which the SDK delivers as a one-shot event.
        Log.e(TAG, "Session error: ${error.description}")
        wearablesViewModel.setRecentError(error.getLocalizedDescription(getApplication()))
      }
    }
  }

  private fun cleanupSession() {
    sessionStateJob?.cancel()
    sessionStateJob = null
    sessionErrorJob?.cancel()
    sessionErrorJob = null
    session = null
  }

  // MARK: - Lifecycle step 2: stream (preview)

  /**
   * Starts the camera stream (preview). Requires an active session. Camera permission is checked
   * first (a query, no redirect); if it isn't granted, the actual request — which redirects to the
   * Meta AI app — is deferred to [confirmCameraPermissionRedirect] so the app-switch is confirmed.
   */
  fun startStreaming() {
    if (!_uiState.value.isSessionActive) {
      wearablesViewModel.setRecentError(
        getApplication<Application>().getString(R.string.error_start_session_first)
      )
      return
    }
    if (stream != null || _uiState.value.isStartingStream) return
    _uiState.update { it.copy(isStartingStream = true) }
    viewModelScope.launch {
      try {
        Wearables.checkPermissionStatus(Permission.CAMERA)
          .onSuccess { status ->
            if (status == PermissionStatus.Granted) {
              beginStream()
            } else {
              _uiState.update { it.copy(showCameraPermissionRedirectConfirm = true) }
            }
          }
          .onFailure { error, _ ->
            Log.e(TAG, "Failed to check camera permission: ${error.description}")
            wearablesViewModel.setRecentError(error.getLocalizedDescription(getApplication()))
          }
      } finally {
        _uiState.update { it.copy(isStartingStream = false) }
      }
    }
  }

  /** Confirmed from the permission prompt: requests camera access, then starts the stream. */
  fun confirmCameraPermissionRedirect(
    requestPermission: suspend (Permission) -> PermissionStatus,
  ) {
    _uiState.update { it.copy(showCameraPermissionRedirectConfirm = false) }
    if (!_uiState.value.isSessionActive || stream != null) return
    viewModelScope.launch {
      val status = requestPermission(Permission.CAMERA)
      if (status == PermissionStatus.Granted) {
        beginStream()
      } else {
        wearablesViewModel.setRecentError(
          getApplication<Application>().getString(R.string.error_camera_permission_denied)
        )
      }
    }
  }

  fun cancelCameraPermissionRedirect() {
    _uiState.update { it.copy(showCameraPermissionRedirectConfirm = false) }
  }

  private fun beginStream() {
    val current = session ?: return
    if (stream != null) return
    // Foreground service keeps the stream/recording alive while backgrounded.
    StreamingService.start(getApplication())
    current
      .addCamera(
        StreamConfiguration(
          videoQuality = VideoQuality.MEDIUM,
          frameRate = FRAME_RATE,
          // Compressed HEVC so frames feed both the on-screen decoder and the passthrough
          // MP4 writer.
          compressVideo = true,
        )
      )
      .onSuccess { addedCamera ->
        camera = addedCamera
        val added = addedCamera.stream
        stream = added
        // Subscribe before start() so no initial transitions are missed.
        setupStreamListeners(added)
        _uiState.update { it.copy(streamState = StreamState.STARTING) }
        added.start().onFailure { error, _ ->
          Log.e(TAG, "Failed to start stream: ${error.description}")
          wearablesViewModel.setRecentError(error.getLocalizedDescription(getApplication()))
          // A failed start leaves the stream attached and the FGS running — tear both down so the
          // UI doesn't stick at STARTING, matching the addCamera() failure path below.
          clearStreamResources()
        }
      }
      .onFailure { error, _ ->
        Log.e(TAG, "Failed to add camera: ${error.description}")
        StreamingService.stop(getApplication())
        wearablesViewModel.setRecentError(error.getLocalizedDescription(getApplication()))
      }
  }

  /** Stops the camera stream but keeps the [DeviceSession] connected. */
  fun stopStreaming() {
    val current = camera ?: return
    _uiState.update { it.copy(streamState = StreamState.STOPPING) }
    current.stop()
    // The stream-state collector converges teardown when STOPPED/CLOSED arrives.
  }

  private fun setupStreamListeners(stream: Stream) {
    videoJob =
      viewModelScope.launch(frameDispatcher) {
        stream.videoStream.collect { handleVideoFrame(it) }
      }
    streamStateJob = viewModelScope.launch {
      // state replays its current value (STOPPED) on subscribe, and we subscribe before start().
      var hasBeenActive = false
      stream.state.collect { state ->
        _uiState.update { it.copy(streamState = state) }
        val isTerminal = state == StreamState.STOPPED || state == StreamState.CLOSED
        if (!isTerminal) {
          hasBeenActive = true
        } else if (hasBeenActive) {
          hasBeenActive = false
          onStreamTerminated()
        }
      }
    }
    streamErrorJob = viewModelScope.launch {
      stream.errorStream.collect { error ->
        Log.e(TAG, "Stream error: ${error.description}")
        wearablesViewModel.setRecentError(error.getLocalizedDescription(getApplication()))
      }
    }
  }

  private fun handleVideoFrame(videoFrame: VideoFrame) {
    if (!videoFrame.isCompressed) return

    val buffer = videoFrame.buffer
    val width = videoFrame.width
    val height = videoFrame.height
    streamWidth = width
    streamHeight = height
    val presentationTimeUs = videoFrame.presentationTimeUs

    val byteArray = ByteArray(buffer.remaining())
    val originalPosition = buffer.position()
    buffer.get(byteArray)
    buffer.position(originalPosition)

    // Accumulate the parameter sets so a recording (or decoder) started after stream start can be
    // primed with a complete VPS+SPS+PPS set.
    csdCollector.offer(byteArray)

    // Append to the recorder (no-op unless recording); keeps writing while backgrounded.
    videoRecorder.writeCompressedFrame(
      byteArray,
      presentationTimeUs,
      width,
      height,
      videoFrame.isCodecConfig,
    )

    // Lazily create the decoder once a Surface is available; it renders directly to it. Prime it
    // with the cached config in case the surface arrived after the config frame. Guarded so a
    // concurrent setSurface(null) can't leave a decoder bound to a released Surface.
    synchronized(decoderLock) {
      if (hevcDecoder == null && targetSurface != null) {
        Log.d(TAG, "Initializing hardware preview decoder for $width x $height")
        hevcDecoder =
          HevcDecoder().also { decoder ->
            decoder.start(width, height, targetSurface!!)
            csdCollector.complete()?.let { decoder.decodeFrame(it, 0) }
          }
      }
      // Feed under the lock so teardown can't null the decoder between check and feed.
      hevcDecoder?.decodeFrame(byteArray, presentationTimeUs)

      if (_uiState.value.isVisionEnabled && targetSurface != null) {
        if (analysisDecoder == null) {
          Log.d(TAG, "Initializing analysis decoder for $width x $height")
          setupImageReader(width, height)
          analysisDecoder =
            HevcDecoder().also { decoder ->
              decoder.start(width, height, decoderSurface!!)
              csdCollector.complete()?.let { decoder.decodeFrame(it, 0) }
            }
        }
        analysisDecoder?.decodeFrame(byteArray, presentationTimeUs)
      }
    }

    if (!videoFrame.isCodecConfig && !_uiState.value.hasReceivedFirstFrame) {
      _uiState.update { it.copy(hasReceivedFirstFrame = true) }
    }
  }

  private fun onStreamTerminated() {
    // Finalize an in-progress recording before releasing the foreground service / wake lock, so the
    // MP4 mux on Dispatchers.IO isn't cut off when the stream stops while backgrounded.
    viewModelScope.launch {
      if (_uiState.value.isRecording) {
        stopVideoRecording()
      }
      clearStreamResources()
    }
  }

  private fun clearStreamResources() {
    videoJob?.cancel()
    videoJob = null
    streamStateJob?.cancel()
    streamStateJob = null
    streamErrorJob?.cancel()
    streamErrorJob = null
    synchronized(decoderLock) {
      hevcDecoder?.stop()
      hevcDecoder = null
      analysisDecoder?.stop()
      analysisDecoder = null
      analysisImageReader?.close()
      analysisImageReader = null
      analysisThread?.quit()
      analysisThread = null
      analysisHandler = null
    }
    csdCollector.reset()
    StreamingService.stop(getApplication())
    // STOPPED is restartable, so only stop() detaches the capability; without it the next
    // addCamera() fails with "a capability of this type is already active". Stopping the camera
    // cascades to its stream child.
    stopCamera(camera)
    camera = null
    stream = null
    _uiState.update { it.copy(streamState = StreamState.STOPPED, hasReceivedFirstFrame = false) }
  }

  /**
   * Stops the camera by closing it. Accepting a [java.io.Closeable] parameter satisfies the
   * AutoCloseableUse detector, which skips methods that take an AutoCloseable argument — the camera
   * outlives any single `use {}` block, so it is stopped explicitly here rather than auto-closed.
   */
  private fun stopCamera(capability: Closeable?) {
    capability?.close()
  }

  // MARK: - Capture

  fun capturePhoto() {
    if (_uiState.value.isCapturingPhoto || !_uiState.value.isStreaming) return
    _uiState.update { it.copy(isCapturingPhoto = true) }
    viewModelScope.launch {
      stream
        ?.capturePhoto()
        ?.onSuccess { photoData ->
          // Decode/rotate is CPU-bound and blocking; keep it off the main thread so capture
          // doesn't jank the UI. Resumes on main for the state update.
          val bitmap = withContext(Dispatchers.Default) { decodePhoto(photoData) }
          if (bitmap != null) {
            _uiState.update {
              it.copy(isCapturingPhoto = false, activePreview = CapturePreview.Photo(bitmap))
            }
          } else {
            _uiState.update { it.copy(isCapturingPhoto = false) }
            wearablesViewModel.setRecentError(
              getApplication<Application>().getString(R.string.error_photo_capture_failed)
            )
          }
        }
        ?.onFailure { error, _ ->
          Log.e(TAG, "Failed to capture photo: ${error.description}")
          _uiState.update { it.copy(isCapturingPhoto = false) }
          wearablesViewModel.setRecentError(error.getLocalizedDescription(getApplication()))
        } ?: _uiState.update { it.copy(isCapturingPhoto = false) }
    }
  }

  // MARK: - Recording

  fun toggleRecording(requestRecordAudioPermission: suspend () -> Boolean) {
    if (_uiState.value.isRecording) {
      viewModelScope.launch { stopVideoRecording() }
    } else {
      startVideoRecording(requestRecordAudioPermission)
    }
  }

  fun startVideoRecording(requestRecordAudioPermission: suspend () -> Boolean) {
    if (!_uiState.value.isStreaming || _uiState.value.isRecording) return
    viewModelScope.launch {
      // Request the mic permission only when sound-in-video is on; record video-only if it's off or
      // the user declines. The prompt appears in context on the first record with the mic on.
      val includeAudio = _uiState.value.includeAudioInStream && requestRecordAudioPermission()
      if (!_uiState.value.isStreaming || _uiState.value.isRecording) return@launch
      // If the user wanted sound but denied the mic, reflect it so the mic icon doesn't stay "on".
      if (_uiState.value.includeAudioInStream && !includeAudio) {
        _uiState.update { it.copy(includeAudioInStream = false) }
      }
      videoRecorder.setIncludeAudio(includeAudio)
      videoRecorder.startRecording(csdCollector.complete())
    }
  }

  suspend fun stopVideoRecording() {
    if (!_uiState.value.isRecording) return
    // The writer starts on the first keyframe. If stop lands just before that frame, wait briefly
    // so even a quick recording finalizes to a file instead of being discarded.
    var waited = 0L
    while (!videoRecorder.hasStartedWriting.value && waited < KEYFRAME_WAIT_MAX_MS) {
      delay(KEYFRAME_WAIT_STEP_MS)
      waited += KEYFRAME_WAIT_STEP_MS
    }
    when (val result = videoRecorder.stopRecording()) {
      is RecordingResult.Completed ->
        _uiState.update { it.copy(activePreview = CapturePreview.Video(result.uri)) }
      RecordingResult.NoRecording ->
        wearablesViewModel.setRecentError(
          getApplication<Application>().getString(R.string.error_recording_too_short)
        )
      RecordingResult.Failed ->
        wearablesViewModel.setRecentError(
          getApplication<Application>().getString(R.string.error_recording_save_failed)
        )
    }
  }

  fun toggleMic() {
    if (!_uiState.value.isStreaming || _uiState.value.isRecording) return
    _uiState.update { it.copy(includeAudioInStream = !it.includeAudioInStream) }
  }

  // MARK: - Dismissers

  fun dismissCapturePreview() {
    val preview = _uiState.value.activePreview
    _uiState.update { it.copy(activePreview = null) }
    if (preview is CapturePreview.Video) {
      // The clip lives in the cache dir, exposed as a FileProvider content URI; delete through the
      // resolver so it resolves back to the real cache file (the URI's path is the provider
      // mapping,
      // not a filesystem path). Photos are held in memory — nothing on disk to clean up.
      viewModelScope.launch(Dispatchers.IO) {
        runCatching {
          getApplication<Application>().contentResolver.delete(preview.uri, null, null)
        }
          .onFailure { Log.w(TAG, "Failed to delete temp recording", it) }
      }
    }
  }

  // MARK: - Photo decoding

  private fun decodePhoto(photo: PhotoData): Bitmap? =
    when (photo) {
      is PhotoData.Bitmap -> photo.bitmap
      is PhotoData.HEIC -> decodeWithOrientation(photo.data)
    }

  // The glasses store orientation in an EXIF tag that BitmapFactory/ImageDecoder don't apply for
  // HEIC, so read TAG_ORIENTATION and rotate — otherwise the preview and shared image are sideways.
  private fun decodeWithOrientation(data: ByteBuffer): Bitmap? {
    val buffer = data.duplicate().apply { rewind() }
    val bytes = ByteArray(buffer.remaining())
    buffer.get(bytes)

    val bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
    if (bitmap == null || bitmap.width == 0 || bitmap.height == 0) {
      bitmap?.recycle()
      Log.e(TAG, "Failed to decode captured photo")
      return null
    }

    val matrix = exifOrientationMatrix(bytes)
    if (matrix.isIdentity) return bitmap

    // Rotating allocates a second full-size bitmap; recycle the source, and fall back to the
    // unrotated image if the device is too low on memory to make the copy.
    return try {
      Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true).also {
        bitmap.recycle()
      }
    } catch (e: OutOfMemoryError) {
      Log.e(TAG, "Failed to rotate captured photo", e)
      bitmap
    }
  }

  private fun exifOrientationMatrix(bytes: ByteArray): Matrix {
    val orientation =
      try {
        ByteArrayInputStream(bytes).use { input ->
          ExifInterface(input)
            .getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
        }
      } catch (e: IOException) {
        Log.w(TAG, "Failed to read EXIF orientation", e)
        ExifInterface.ORIENTATION_NORMAL
      }
    val matrix = Matrix()
    when (orientation) {
      ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> matrix.postScale(-1f, 1f)
      ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
      ExifInterface.ORIENTATION_FLIP_VERTICAL -> matrix.postScale(1f, -1f)
      ExifInterface.ORIENTATION_TRANSPOSE -> {
        matrix.postRotate(90f)
        matrix.postScale(-1f, 1f)
      }
      ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90f)
      ExifInterface.ORIENTATION_TRANSVERSE -> {
        matrix.postRotate(270f)
        matrix.postScale(-1f, 1f)
      }
      ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270f)
    }
    return matrix
  }

  override fun onCleared() {
    super.onCleared()
    clearStreamResources()
    unbindPathAnalysis()
    session?.stop()
    cleanupSession()
    audioInputHandler.cleanup()
    videoRecorder.close()
  }

  class Factory(
    private val application: Application,
    private val wearablesViewModel: WearablesViewModel,
  ) : ViewModelProvider.Factory {
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
      if (modelClass.isAssignableFrom(CameraViewModel::class.java)) {
        @Suppress("UNCHECKED_CAST")
        return CameraViewModel(application, wearablesViewModel) as T
      }
      throw IllegalArgumentException("Unknown ViewModel class")
    }
  }
}