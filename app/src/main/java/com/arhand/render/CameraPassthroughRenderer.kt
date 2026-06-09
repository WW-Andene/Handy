package com.arhand.render

import android.graphics.Bitmap
import android.graphics.SurfaceTexture
import android.opengl.GLES11Ext
import android.opengl.GLES30
import android.opengl.GLUtils
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * Draws a full-screen camera preview behind the hand mesh.
 *
 * Uses a standard GL_TEXTURE_2D (not SurfaceTexture/OES) because frames
 * arrive as Bitmaps from CameraFrameProvider. Each frame, the Bitmap is
 * uploaded to the texture via GLUtils.texImage2D and drawn on a fullscreen
 * NDC quad before the hand mesh renders. Depth writes are disabled so the
 * background never occludes hand geometry.
 *
 * Coordinate note: camera Bitmaps from YUV conversion are in sensor
 * orientation. Front camera images need a horizontal flip (mirrorX).
 * This is handled by choosing the correct UV set in the vertex buffer.
 */
class CameraPassthroughRenderer {

    private var programId = 0
    private var vboId = 0
    private var textureId = 0

    private var uTextureLoc = 0
    private var uMirrorXLoc = 0

    @Volatile var latestBitmap: Bitmap? = null
    @Volatile var mirrorX: Boolean = true

    // ─── Fullscreen quad: NDC positions + UVs ────────────────────────────
    // Two triangles covering [-1,1]² in NDC.
    // UV (0,0) = top-left, (1,1) = bottom-right — matches Android Bitmap layout.
    private val quadVerts = floatArrayOf(
        // x      y     u     v
        -1f,  -1f,  0f,  1f,   // bottom-left
         1f,  -1f,  1f,  1f,   // bottom-right
        -1f,   1f,  0f,  0f,   // top-left
         1f,  -1f,  1f,  1f,   // bottom-right
         1f,   1f,  1f,  0f,   // top-right
        -1f,   1f,  0f,  0f    // top-left
    )

    private lateinit var quadBuffer: FloatBuffer

    fun init() {
        quadBuffer = ByteBuffer
            .allocateDirect(quadVerts.size * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
            .apply { put(quadVerts); position(0) }

        programId = buildProgram(ShaderPrograms.CAMERA_VERT, ShaderPrograms.CAMERA_FRAG)

        uTextureLoc = GLES30.glGetUniformLocation(programId, "uTexture")
        uMirrorXLoc = GLES30.glGetUniformLocation(programId, "uMirrorX")

        // VBO
        val buf = IntArray(1)
        GLES30.glGenBuffers(1, buf, 0)
        vboId = buf[0]
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, vboId)
        GLES30.glBufferData(
            GLES30.GL_ARRAY_BUFFER,
            quadVerts.size * 4,
            quadBuffer,
            GLES30.GL_STATIC_DRAW
        )
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, 0)

        // Texture
        GLES30.glGenTextures(1, buf, 0)
        textureId = buf[0]
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, textureId)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, 0)
    }

    /**
     * Call once per frame from onDrawFrame, BEFORE drawing hand geometry.
     * Uploads the latest Bitmap if available, then draws the fullscreen quad.
     */
    fun draw() {
        val bmp = latestBitmap ?: return

        // Upload Bitmap → GL texture (only if we have a frame)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, textureId)
        GLUtils.texImage2D(GLES30.GL_TEXTURE_2D, 0, bmp, 0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, 0)

        // Draw — no depth write, no depth test so quad sits behind everything
        GLES30.glDepthMask(false)
        GLES30.glDisable(GLES30.GL_DEPTH_TEST)

        GLES30.glUseProgram(programId)

        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, textureId)
        GLES30.glUniform1i(uTextureLoc, 0)
        GLES30.glUniform1i(uMirrorXLoc, if (mirrorX) 1 else 0)

        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, vboId)

        val stride = 4 * 4 // 4 floats × 4 bytes
        val posLoc = GLES30.glGetAttribLocation(programId, "aPosition")
        val uvLoc  = GLES30.glGetAttribLocation(programId, "aUV")

        GLES30.glEnableVertexAttribArray(posLoc)
        GLES30.glVertexAttribPointer(posLoc, 2, GLES30.GL_FLOAT, false, stride, 0)

        GLES30.glEnableVertexAttribArray(uvLoc)
        GLES30.glVertexAttribPointer(uvLoc, 2, GLES30.GL_FLOAT, false, stride, 2 * 4)

        GLES30.glDrawArrays(GLES30.GL_TRIANGLES, 0, 6)

        GLES30.glDisableVertexAttribArray(posLoc)
        GLES30.glDisableVertexAttribArray(uvLoc)
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, 0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, 0)

        // Restore depth state for subsequent hand mesh drawing
        GLES30.glDepthMask(true)
        GLES30.glEnable(GLES30.GL_DEPTH_TEST)
    }

    fun release() {
        if (vboId != 0) GLES30.glDeleteBuffers(1, intArrayOf(vboId), 0)
        if (textureId != 0) GLES30.glDeleteTextures(1, intArrayOf(textureId), 0)
        if (programId != 0) GLES30.glDeleteProgram(programId)
    }

    // ─── Shader compiler helper ───────────────────────────────────────────

    private fun buildProgram(vertSrc: String, fragSrc: String): Int {
        val vert = compileShader(GLES30.GL_VERTEX_SHADER, vertSrc)
        val frag = compileShader(GLES30.GL_FRAGMENT_SHADER, fragSrc)
        return GLES30.glCreateProgram().also { prog ->
            GLES30.glAttachShader(prog, vert)
            GLES30.glAttachShader(prog, frag)
            GLES30.glLinkProgram(prog)
            GLES30.glDeleteShader(vert)
            GLES30.glDeleteShader(frag)
        }
    }

    private fun compileShader(type: Int, src: String): Int {
        return GLES30.glCreateShader(type).also { shader ->
            GLES30.glShaderSource(shader, src)
            GLES30.glCompileShader(shader)
        }
    }
}
