package com.arhand.render

import android.opengl.GLES30
import android.opengl.Matrix
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * G1 — GL renderer for the live-deforming scanned hand mesh.
 *
 * Receives a pre-deformed position array from [LiveMeshDeformer] every tracking frame
 * (written from the tracking coroutine via [updateMesh]) and draws it on the GL thread
 * via [draw].
 *
 * Normals are re-computed CPU-side from the deformed positions each frame. At 4 000–8 000
 * vertices this costs < 0.5ms and avoids the complexity of transforming a separate normal
 * buffer through the skinning matrix palette.
 *
 * Uses the existing [ShaderPrograms.PHONG_VERT] / [ShaderPrograms.PHONG_FRAG] shaders with
 * a distinctive teal skin colour (`#1FC8A8`) so the live-deforming mesh is visually distinct
 * from the static snapshot mesh and the procedural `HandRenderer`.
 */
class LiveMeshRenderer {

    // ─── GL handles ───────────────────────────────────────────────────────────
    private var program     = 0
    private val vao         = IntArray(1)
    private val posVbo      = IntArray(1)
    private val normVbo     = IntArray(1)

    // ─── Mesh state (written from tracking thread, read on GL thread) ─────────
    @Volatile private var pendingPositions: FloatArray? = null
    @Volatile private var positionsDirty               = false
    private var vertexCount  = 0

    // ─── Colour ───────────────────────────────────────────────────────────────
    // Teal-ish "mirror hand" colour — distinct from the procedural mesh skin tone.
    private val liveColor = floatArrayOf(0.122f, 0.784f, 0.659f)   // #1FC8A8

    // ─── Public API ───────────────────────────────────────────────────────────

    fun init() {
        program = buildProg(ShaderPrograms.PHONG_VERT, ShaderPrograms.PHONG_FRAG)
        GLES30.glGenVertexArrays(1, vao,     0)
        GLES30.glGenBuffers(1,    posVbo,    0)
        GLES30.glGenBuffers(1,    normVbo,   0)
    }

    /**
     * Called from the tracking coroutine with CPU-deformed positions.
     * The actual VBO upload happens lazily on the GL thread.
     */
    fun updateMesh(positions: FloatArray) {
        pendingPositions = positions
        positionsDirty   = true
    }

    /** Clear mesh — called when live mode is toggled off or scan is reset. */
    fun clearMesh() {
        pendingPositions = null
        positionsDirty   = false
        vertexCount      = 0
    }

    fun draw(view: FloatArray, proj: FloatArray, torchOn: Boolean, timeSec: Float) {
        if (positionsDirty) {
            pendingPositions?.let { uploadPositions(it) }
            positionsDirty = false
        }

        if (vertexCount == 0) return

        val model = FloatArray(16).also { Matrix.setIdentityM(it, 0) }

        GLES30.glUseProgram(program)

        GLES30.glUniformMatrix4fv(loc("uModel"),      1, false, model,   0)
        GLES30.glUniformMatrix4fv(loc("uView"),       1, false, view,    0)
        GLES30.glUniformMatrix4fv(loc("uProjection"), 1, false, proj,    0)

        // Normal matrix = mat3 from identity model
        val nm = floatArrayOf(
            1f, 0f, 0f,
            0f, 1f, 0f,
            0f, 0f, 1f
        )
        GLES30.glUniformMatrix3fv(loc("uNormalMatrix"), 1, false, nm, 0)

        setLightingUniforms(torchOn, timeSec)

        GLES30.glUniform3fv(loc("uBaseColor"), 1, liveColor, 0)
        GLES30.glUniform1f(loc("uAlpha"), 0.92f)
        GLES30.glUniform1f(loc("uTime"), timeSec)
        GLES30.glUniform1i(loc("uInferred"), 0)
        GLES30.glUniform1i(loc("uTorchOn"), if (torchOn) 1 else 0)

        GLES30.glBindVertexArray(vao[0])
        GLES30.glDrawArrays(GLES30.GL_TRIANGLES, 0, vertexCount)
        GLES30.glBindVertexArray(0)
    }

    fun release() {
        if (posVbo[0]  != 0) { GLES30.glDeleteBuffers(1, posVbo,  0); posVbo[0]  = 0 }
        if (normVbo[0] != 0) { GLES30.glDeleteBuffers(1, normVbo, 0); normVbo[0] = 0 }
        if (vao[0]     != 0) { GLES30.glDeleteVertexArrays(1, vao, 0); vao[0]    = 0 }
        if (program    != 0) { GLES30.glDeleteProgram(program);        program    = 0 }
        vertexCount = 0
    }

    // ─── GL helpers ───────────────────────────────────────────────────────────

    private fun uploadPositions(positions: FloatArray) {
        val vc = positions.size / 3
        vertexCount = vc
        if (vc == 0) return

        val normals = computeFlatNormals(positions)

        GLES30.glBindVertexArray(vao[0])
        uploadVbo(posVbo[0],  positions, 3, "aPosition")
        uploadVbo(normVbo[0], normals,   3, "aNormal")
        GLES30.glBindVertexArray(0)
    }

    private fun uploadVbo(vbo: Int, data: FloatArray, components: Int, attrib: String) {
        val aLoc = GLES30.glGetAttribLocation(program, attrib)
        if (aLoc < 0) return
        val buf = toFloatBuffer(data)
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, vbo)
        GLES30.glBufferData(GLES30.GL_ARRAY_BUFFER, data.size * 4, buf, GLES30.GL_DYNAMIC_DRAW)
        GLES30.glEnableVertexAttribArray(aLoc)
        GLES30.glVertexAttribPointer(aLoc, components, GLES30.GL_FLOAT, false, 0, 0)
    }

    private fun loc(name: String) = GLES30.glGetUniformLocation(program, name)

    private fun setLightingUniforms(torchOn: Boolean, timeSec: Float) {
        val L = LightingModel
        fun v3(name: String, v: FloatArray) = GLES30.glUniform3fv(loc(name), 1, v, 0)
        fun f1(name: String, v: Float)      = GLES30.glUniform1f(loc(name), v)

        v3("uAmbientColor",   L.ambientColor);  f1("uAmbientIntensity", L.ambientIntensity)
        v3("uKeyDir",  L.keyDir);  v3("uKeyColor",  L.keyColor);  f1("uKeyIntensity",  L.keyIntensity)
        v3("uRimDir",  L.rimDir);  v3("uRimColor",  L.rimColor);  f1("uRimIntensity",  L.rimIntensity)
        v3("uSSSDir",  L.sssDir);  v3("uSSSColor",  L.sssColor);  f1("uSSSIntensity",  L.sssIntensity)
        v3("uFillDir", L.fillDir); v3("uFillColor", L.fillColor); f1("uFillIntensity", L.fillIntensity)
        v3("uTorchPos",   floatArrayOf(0f, 0.2f, 0.5f))
        v3("uTorchColor", L.torchColor); f1("uTorchIntensity", L.torchIntensity)
        GLES30.glUniform3fv(loc("uTorchAttenuation"), 1, L.torchAttenuation, 0)
        v3("uCameraPos", floatArrayOf(0f, 0f, 5f))
        f1("uTime", timeSec)
    }

    private fun computeFlatNormals(positions: FloatArray): FloatArray {
        val normals = FloatArray(positions.size)
        var i = 0
        while (i + 9 <= positions.size) {
            val ax = positions[i];   val ay = positions[i+1]; val az = positions[i+2]
            val bx = positions[i+3]; val by = positions[i+4]; val bz = positions[i+5]
            val cx = positions[i+6]; val cy = positions[i+7]; val cz = positions[i+8]
            val ux = bx - ax; val uy = by - ay; val uz = bz - az
            val vx = cx - ax; val vy = cy - ay; val vz = cz - az
            var nx = uy * vz - uz * vy
            var ny = uz * vx - ux * vz
            var nz = ux * vy - uy * vx
            val len = kotlin.math.sqrt((nx * nx + ny * ny + nz * nz).toDouble())
                .toFloat().coerceAtLeast(1e-6f)
            nx /= len; ny /= len; nz /= len
            for (j in 0 until 3) {
                normals[i + j * 3]     = nx
                normals[i + j * 3 + 1] = ny
                normals[i + j * 3 + 2] = nz
            }
            i += 9
        }
        return normals
    }

    private fun toFloatBuffer(data: FloatArray): FloatBuffer =
        ByteBuffer.allocateDirect(data.size * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
            .apply { put(data); position(0) }

    private fun buildProg(vert: String, frag: String): Int {
        val vs = GLES30.glCreateShader(GLES30.GL_VERTEX_SHADER)
            .also { GLES30.glShaderSource(it, vert); GLES30.glCompileShader(it) }
        val fs = GLES30.glCreateShader(GLES30.GL_FRAGMENT_SHADER)
            .also { GLES30.glShaderSource(it, frag); GLES30.glCompileShader(it) }
        return GLES30.glCreateProgram()
            .also {
                GLES30.glAttachShader(it, vs)
                GLES30.glAttachShader(it, fs)
                GLES30.glLinkProgram(it)
            }
    }
}
