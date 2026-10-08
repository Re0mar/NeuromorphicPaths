package com.neuromorphicpaths.pixel.ar

import android.opengl.GLES11Ext
import android.opengl.GLES20
import com.google.ar.core.Coordinates2d
import com.google.ar.core.Frame
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * Draws the camera image ARCore renders into the external texture, as a full-screen quad.
 *
 * The quad's corners are fixed in normalized device coordinates. Its texture coordinates come
 * from [Frame.transformCoordinates2d], which is the one place display rotation is handled: ARCore
 * knows the sensor's orientation and the display's, and hands back the texture coordinates that
 * put the picture upright. Rotating the quad by hand from the display's rotation disagrees with
 * it on some devices. Everything here is allocated once, so a draw allocates nothing.
 */
class CameraBackground {
    private val quadCoordinates: FloatBuffer = directFloatBuffer(QUAD_NDC_CORNERS)
    private val textureCoordinates: FloatBuffer = directFloatBuffer(FloatArray(QUAD_NDC_CORNERS.size))
    private var program = 0
    private var positionAttribute = 0
    private var textureCoordinateAttribute = 0
    private var textureUniform = 0
    private var textureId = 0
    private var textureCoordinatesReady = false

    /** Compile and link the shaders. Must run on the GL thread, after the texture exists. */
    fun createOnGlThread(cameraTextureId: Int) {
        textureId = cameraTextureId
        val vertexShader = compile(GLES20.GL_VERTEX_SHADER, VERTEX_SHADER)
        val fragmentShader = compile(GLES20.GL_FRAGMENT_SHADER, FRAGMENT_SHADER)
        program = GLES20.glCreateProgram()
        GLES20.glAttachShader(program, vertexShader)
        GLES20.glAttachShader(program, fragmentShader)
        GLES20.glLinkProgram(program)
        val linked = IntArray(1)
        GLES20.glGetProgramiv(program, GLES20.GL_LINK_STATUS, linked, 0)
        if (linked[0] == 0) {
            // A silent failure here is a black screen that looks like the old app.
            val log = GLES20.glGetProgramInfoLog(program)
            GLES20.glDeleteProgram(program)
            throw IllegalStateException("camera background program failed to link: $log")
        }
        GLES20.glDeleteShader(vertexShader)
        GLES20.glDeleteShader(fragmentShader)
        positionAttribute = GLES20.glGetAttribLocation(program, "a_Position")
        textureCoordinateAttribute = GLES20.glGetAttribLocation(program, "a_TextureCoordinate")
        textureUniform = GLES20.glGetUniformLocation(program, "u_Texture")
    }

    /** Draw this frame's camera image. Called on every draw, whether or not the frame is new. */
    fun draw(frame: Frame) {
        if (frame.hasDisplayGeometryChanged() || !textureCoordinatesReady) {
            quadCoordinates.position(0)
            textureCoordinates.position(0)
            frame.transformCoordinates2d(
                Coordinates2d.OPENGL_NORMALIZED_DEVICE_COORDINATES,
                quadCoordinates,
                Coordinates2d.TEXTURE_NORMALIZED,
                textureCoordinates,
            )
            textureCoordinatesReady = true
        }

        // The camera is the backdrop. It must never be hidden by a depth test against nothing.
        GLES20.glDisable(GLES20.GL_DEPTH_TEST)
        GLES20.glDepthMask(false)

        GLES20.glUseProgram(program)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureId)
        GLES20.glUniform1i(textureUniform, 0)

        quadCoordinates.position(0)
        textureCoordinates.position(0)
        GLES20.glVertexAttribPointer(positionAttribute, COORDINATES_PER_VERTEX, GLES20.GL_FLOAT, false, 0, quadCoordinates)
        GLES20.glVertexAttribPointer(textureCoordinateAttribute, COORDINATES_PER_VERTEX, GLES20.GL_FLOAT, false, 0, textureCoordinates)
        GLES20.glEnableVertexAttribArray(positionAttribute)
        GLES20.glEnableVertexAttribArray(textureCoordinateAttribute)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, VERTEX_COUNT)
        GLES20.glDisableVertexAttribArray(positionAttribute)
        GLES20.glDisableVertexAttribArray(textureCoordinateAttribute)

        GLES20.glDepthMask(true)
        GLES20.glEnable(GLES20.GL_DEPTH_TEST)
    }

    private fun compile(type: Int, source: String): Int {
        val shader = GLES20.glCreateShader(type)
        GLES20.glShaderSource(shader, source)
        GLES20.glCompileShader(shader)
        val compiled = IntArray(1)
        GLES20.glGetShaderiv(shader, GLES20.GL_COMPILE_STATUS, compiled, 0)
        if (compiled[0] == 0) {
            val log = GLES20.glGetShaderInfoLog(shader)
            GLES20.glDeleteShader(shader)
            throw IllegalStateException("camera background shader failed to compile: $log")
        }
        return shader
    }

    private fun directFloatBuffer(values: FloatArray): FloatBuffer =
        ByteBuffer.allocateDirect(values.size * BYTES_PER_FLOAT).order(ByteOrder.nativeOrder()).asFloatBuffer().also {
            it.put(values)
            it.position(0)
        }

    private companion object {
        const val BYTES_PER_FLOAT = 4
        const val COORDINATES_PER_VERTEX = 2
        const val VERTEX_COUNT = 4

        // A triangle strip covering the whole viewport: bottom left, top left, bottom right, top right.
        val QUAD_NDC_CORNERS = floatArrayOf(-1f, -1f, -1f, 1f, 1f, -1f, 1f, 1f)

        const val VERTEX_SHADER = """
            attribute vec4 a_Position;
            attribute vec2 a_TextureCoordinate;
            varying vec2 v_TextureCoordinate;
            void main() {
                gl_Position = a_Position;
                v_TextureCoordinate = a_TextureCoordinate;
            }
        """

        const val FRAGMENT_SHADER = """
            #extension GL_OES_EGL_image_external : require
            precision mediump float;
            varying vec2 v_TextureCoordinate;
            uniform samplerExternalOES u_Texture;
            void main() {
                gl_FragColor = texture2D(u_Texture, v_TextureCoordinate);
            }
        """
    }
}
