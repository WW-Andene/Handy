package com.arhand.render

import android.opengl.GLES30
import android.opengl.Matrix
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Renders the marching cubes depth mesh.
 * Supports solid (Phong semi-transparent) and wireframe modes.
 */
class DepthMeshRenderer {
    private var phongProgram = 0
    private var lineProgram  = 0
    private var meshPositions: FloatArray = FloatArray(0)
    private var meshNormals:   FloatArray = FloatArray(0)

    fun init() {
        phongProgram = buildProg(ShaderPrograms.PHONG_VERT, ShaderPrograms.PHONG_FRAG)
        lineProgram  = buildProg(ShaderPrograms.LINE_VERT,  ShaderPrograms.LINE_FRAG)
    }

    fun update(positions: FloatArray) {
        meshPositions = positions
        // Compute flat normals from triangle vertices
        meshNormals = computeFlatNormals(positions)
    }

    fun draw(view: FloatArray, proj: FloatArray, mode: RenderMode) {
        if (meshPositions.isEmpty()) return

        val model = FloatArray(16).also { Matrix.setIdentityM(it, 0) }

        if (mode == RenderMode.WIREFRAME || mode == RenderMode.SKELETON) {
            GLES30.glUseProgram(lineProgram)
            GLES30.glUniformMatrix4fv(GLES30.glGetUniformLocation(lineProgram, "uModel"), 1, false, model, 0)
            GLES30.glUniformMatrix4fv(GLES30.glGetUniformLocation(lineProgram, "uView"),  1, false, view,  0)
            GLES30.glUniformMatrix4fv(GLES30.glGetUniformLocation(lineProgram, "uProjection"), 1, false, proj, 0)
            GLES30.glUniform4fv(GLES30.glGetUniformLocation(lineProgram, "uColor"), 1,
                floatArrayOf(0f, 1f, 0.898f, 0.4f), 0)
            drawRaw(lineProgram, meshPositions, GLES30.GL_TRIANGLES)
        } else {
            GLES30.glUseProgram(phongProgram)
            GLES30.glUniformMatrix4fv(GLES30.glGetUniformLocation(phongProgram, "uModel"), 1, false, model, 0)
            GLES30.glUniformMatrix4fv(GLES30.glGetUniformLocation(phongProgram, "uView"),  1, false, view,  0)
            GLES30.glUniformMatrix4fv(GLES30.glGetUniformLocation(phongProgram, "uProjection"), 1, false, proj, 0)
            GLES30.glUniform3fv(GLES30.glGetUniformLocation(phongProgram, "uBaseColor"), 1, floatArrayOf(0.5f, 0.8f, 1.0f), 0)
            GLES30.glUniform1f(GLES30.glGetUniformLocation(phongProgram, "uAlpha"), 0.4f)
            GLES30.glUniform1i(GLES30.glGetUniformLocation(phongProgram, "uTorchOn"), 0)
            GLES30.glUniform1i(GLES30.glGetUniformLocation(phongProgram, "uInferred"), 0)
            GLES30.glDepthMask(false)
            drawRaw(phongProgram, meshPositions, GLES30.GL_TRIANGLES)
            GLES30.glDepthMask(true)
        }
    }

    private fun drawRaw(prog: Int, verts: FloatArray, glMode: Int) {
        val buf = ByteBuffer.allocateDirect(verts.size * 4).order(ByteOrder.nativeOrder()).asFloatBuffer().apply { put(verts); position(0) }
        val vbo = IntArray(1); GLES30.glGenBuffers(1, vbo, 0)
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, vbo[0])
        GLES30.glBufferData(GLES30.GL_ARRAY_BUFFER, verts.size * 4, buf, GLES30.GL_DYNAMIC_DRAW)
        val loc = GLES30.glGetAttribLocation(prog, "aPosition")
        GLES30.glEnableVertexAttribArray(loc)
        GLES30.glVertexAttribPointer(loc, 3, GLES30.GL_FLOAT, false, 0, 0)
        GLES30.glDrawArrays(glMode, 0, verts.size / 3)
        GLES30.glDeleteBuffers(1, vbo, 0)
    }

    private fun computeFlatNormals(positions: FloatArray): FloatArray {
        val normals = FloatArray(positions.size)
        var i = 0
        while (i + 8 < positions.size) {
            val ax = positions[i];   val ay = positions[i+1]; val az = positions[i+2]
            val bx = positions[i+3]; val by = positions[i+4]; val bz = positions[i+5]
            val cx = positions[i+6]; val cy = positions[i+7]; val cz = positions[i+8]
            val ux = bx-ax; val uy = by-ay; val uz = bz-az
            val vx = cx-ax; val vy = cy-ay; val vz = cz-az
            val nx = uy*vz - uz*vy
            val ny = uz*vx - ux*vz
            val nz = ux*vy - uy*vx
            val len = Math.sqrt((nx*nx + ny*ny + nz*nz).toDouble()).toFloat().coerceAtLeast(1e-6f)
            for (j in 0 until 3) { normals[i+j*3] = nx/len; normals[i+j*3+1] = ny/len; normals[i+j*3+2] = nz/len }
            i += 9
        }
        return normals
    }

    private fun buildProg(v: String, f: String): Int {
        val vs = GLES30.glCreateShader(GLES30.GL_VERTEX_SHADER).also { GLES30.glShaderSource(it, v); GLES30.glCompileShader(it) }
        val fs = GLES30.glCreateShader(GLES30.GL_FRAGMENT_SHADER).also { GLES30.glShaderSource(it, f); GLES30.glCompileShader(it) }
        return GLES30.glCreateProgram().also { GLES30.glAttachShader(it, vs); GLES30.glAttachShader(it, fs); GLES30.glLinkProgram(it) }
    }
}
