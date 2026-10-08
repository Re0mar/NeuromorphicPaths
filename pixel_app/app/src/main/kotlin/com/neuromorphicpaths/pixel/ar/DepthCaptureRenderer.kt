package com.neuromorphicpaths.pixel.ar

import android.opengl.GLES11Ext
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.util.Log
import com.google.ar.core.Plane
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import com.google.ar.core.exceptions.CameraNotAvailableException
import com.neuromorphicpaths.pixel.wire.DepthMessage
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

/**
 * Drives the ARCore session from a GL surface, draws the camera, and hands each new frame's depth
 * to a callback.
 *
 * Two rates on purpose. The picture is drawn on every draw, at the display rate, because ARCore's
 * texture is valid on every draw and the display expects a picture every refresh. A depth message
 * is sent once per ARCore frame, which [NewFrameGate] decides, because the first phone run sent
 * sixty copies a second of frames that arrived at thirty. The gate guards the message, not the
 * picture.
 *
 * @param onFrameHandled called once per new ARCore frame, past the gate and before conversion, with
 * the frame's own timestamp in nanoseconds. A frame with no depth is still handled, which is how the
 * timing log tells "never sent" from "never seen". Runs on the GL thread, so it must not block
 */
class DepthCaptureRenderer(
    private val sessionProvider: () -> Session?,
    private val onFrame: (DepthMessage) -> Unit,
    private val onState: (CaptureState) -> Unit,
    private val onFrameHandled: (Long) -> Unit = {},
) : GLSurfaceView.Renderer {
    private var cameraTexture = 0
    private var frames = 0
    private var framesWithDepth = 0
    private val newFrames = NewFrameGate()
    private var background: CameraBackground? = null

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        val textures = IntArray(1)
        GLES20.glGenTextures(1, textures, 0)
        cameraTexture = textures[0]
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, cameraTexture)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
        GLES20.glClearColor(0f, 0f, 0f, 1f)
        // A new surface means a new GL context, so the program is built again with the texture.
        background = CameraBackground().also { it.createOnGlThread(cameraTexture) }
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        GLES20.glViewport(0, 0, width, height)
        sessionProvider()?.setDisplayGeometry(0, width, height)
    }

    override fun onDrawFrame(gl: GL10?) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)
        val session = sessionProvider() ?: return
        session.setCameraTextureName(cameraTexture)

        val frame = try {
            session.update()
        } catch (unavailable: CameraNotAvailableException) {
            // Another app has the camera, or it is being switched. Expected while the app is
            // coming to the foreground, and the next draw tries again.
            onState(CaptureState.CameraUnavailable)
            return
        }
        // The picture first, on every draw. Then the gate: a draw that got the frame the
        // previous draw already handled sends nothing.
        background?.draw(frame)
        if (!newFrames.isNew(frame.timestamp)) return
        onFrameHandled(frame.timestamp)
        frames += 1
        val camera = frame.camera
        val tracking = camera.trackingState == TrackingState.TRACKING

        val floor = chooseFloor(session)
        val message = ArCoreToWire.convert(frame, camera, floor)
        if (message != null) {
            framesWithDepth += 1
            onFrame(message)
        }
        onState(
            CaptureState.Running(
                tracking = tracking,
                frames = frames,
                framesWithDepth = framesWithDepth,
                hasFloor = floor != null,
                depthSize = message?.let { "${it.columns}x${it.rows}" },
            ),
        )
    }

    /**
     * The largest upward-facing plane ARCore is tracking, which is the floor when there is one.
     *
     * The choice itself is [FloorChoice.chooseFloorIndex], a pure function with its own tests. This
     * only builds its input and maps the winning index back to the plane. The laptop fits its own
     * floor when this is null, and gates whatever is sent, so neither side has to be right alone.
     */
    private fun chooseFloor(session: Session): Plane? {
        val planes = session.getAllTrackables(Plane::class.java)
            .filter { it.trackingState == TrackingState.TRACKING && it.type == Plane.Type.HORIZONTAL_UPWARD_FACING }
        val candidates = planes.map { FloorCandidate(it.extentX, it.extentZ, it.centerPose.ty()) }
        val index = FloorChoice.chooseFloorIndex(candidates) ?: return null
        return planes[index]
    }

    private companion object {
        const val TAG = "DepthCaptureRenderer"
    }
}

/** What the screen shows about capture. */
sealed interface CaptureState {
    data object CameraUnavailable : CaptureState
    data class Running(
        val tracking: Boolean,
        val frames: Int,
        val framesWithDepth: Int,
        val hasFloor: Boolean,
        val depthSize: String?,
    ) : CaptureState
}
