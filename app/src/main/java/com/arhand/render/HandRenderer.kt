package com.arhand.render

import android.opengl.GLES30
import android.opengl.Matrix
import com.arhand.tracking.HandLandmarks
import com.arhand.tracking.LM
import com.arhand.util.Vec3
import com.arhand.util.landmarkToWorld
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.IntBuffer

/**
 * Renders a single hand in mesh / wireframe / skeleton modes.
 * Maintains VAOs/VBOs for all geometry parts.
 * Port of the HandRenderer class from the HTML prototype.
 */
class HandRenderer {
    private var phongProgram  = 0
    private var basicProgram  = 0
    private var lineProgram   = 0

    private val vaos = mutableListOf<Int>()
    private val vbos = mutableListOf<Int>()

    // Skin color — rgba(210,170,120,1) in float
    private val skinColor = floatArrayOf(0.824f, 0.667f, 0.471f)

    // Persistent VAO + VBOs for mesh drawing — allocated once in init(), reused every frame.
    // Using a single VAO for all mesh parts; geometry is re-uploaded per-part via glBufferData.
    private val meshVao = IntArray(1)
    private val meshVbo = IntArray(2)  // [0]=positions, [1]=normals
    private val meshIbo = IntArray(1)

    fun init() {
        phongProgram = compileProgram(ShaderPrograms.PHONG_VERT, ShaderPrograms.PHONG_FRAG)
        basicProgram = compileProgram(ShaderPrograms.BASIC_VERT, ShaderPrograms.BASIC_FRAG)
        lineProgram  = compileProgram(ShaderPrograms.LINE_VERT,  ShaderPrograms.LINE_FRAG)

        GLES30.glGenVertexArrays(1, meshVao, 0)
        GLES30.glGenBuffers(2, meshVbo, 0)
        GLES30.glGenBuffers(1, meshIbo, 0)
    }

    private var currentLms: HandLandmarks? = null
    private var currentAspect = 1f
    private var currentMirrorX = true

    fun update(lms: HandLandmarks, aspect: Float, mirrorX: Boolean) {
        currentLms = lms
        currentAspect = aspect
        currentMirrorX = mirrorX
        // Rebuild VBOs — deferred to draw() on GL thread
    }

    fun clear() { currentLms = null }

    fun draw(view: FloatArray, proj: FloatArray, mode: RenderMode, torchOn: Boolean, timeSec: Float) {
        val lms = currentLms ?: return
        if (lms.size < 21) return

        // Convert landmarks to world-space Vec3
        val pts = lms.map { lm ->
            val (x, y, z) = landmarkToWorld(lm, currentAspect, mirrorX = currentMirrorX)
            Vec3(x, y, z)
        }

        when (mode) {
            RenderMode.MESH,
            RenderMode.LIVE_MESH -> drawMesh(pts, lms, view, proj, torchOn, timeSec)
            RenderMode.WIREFRAME -> drawWireframe(pts, view, proj)
            RenderMode.SKELETON,
            RenderMode.ASSET_3D,
            RenderMode.ASSET_2D  -> drawSkeleton(pts, lms, view, proj, timeSec)
        }
    }

    private fun drawMesh(pts: List<Vec3>, lms: HandLandmarks, view: FloatArray, proj: FloatArray, torchOn: Boolean, timeSec: Float) {
        GLES30.glUseProgram(phongProgram)

        val model = FloatArray(16).also { Matrix.setIdentityM(it, 0) }
        val normalMatrix = FloatArray(9) // mat3 from model

        // Compute 3x3 normal matrix from 4x4 model matrix (identity here = same)
        normalMatrix[0] = model[0]; normalMatrix[1] = model[1]; normalMatrix[2] = model[2]
        normalMatrix[3] = model[4]; normalMatrix[4] = model[5]; normalMatrix[5] = model[6]
        normalMatrix[6] = model[8]; normalMatrix[7] = model[9]; normalMatrix[8] = model[10]

        setMatrixUniforms(phongProgram, model, view, proj)
        GLES30.glUniformMatrix3fv(
            GLES30.glGetUniformLocation(phongProgram, "uNormalMatrix"), 1, false, normalMatrix, 0
        )
        setLightingUniforms(phongProgram, torchOn, timeSec, pts)

        GLES30.glUniform3fv(GLES30.glGetUniformLocation(phongProgram, "uBaseColor"), 1, skinColor, 0)
        GLES30.glUniform1f(GLES30.glGetUniformLocation(phongProgram, "uAlpha"), 0.97f)
        GLES30.glUniform1f(GLES30.glGetUniformLocation(phongProgram, "uTime"), timeSec)

        // Draw each finger tube
        val unitR = computeUnitR(pts)
        for (fi in HandMeshBuilder.FINGER_CHAINS.indices) {
            val chain = HandMeshBuilder.FINGER_CHAINS[fi]
            val prof  = HandMeshBuilder.FINGER_PROFILES[fi]
            val fingerPts = chain.map { pts[it] }
            val radii = floatArrayOf(prof.base * unitR, prof.mid * unitR, prof.mid * unitR * 0.85f, prof.tip * unitR)
            val bulged = HandMeshBuilder.addKnuckleBulge(radii, prof.knuckleMult)

            val mesh = HandMeshBuilder.buildTube(fingerPts, bulged.toList())
            val hasInferred = chain.any { lms[it].inferred }
            GLES30.glUniform1i(GLES30.glGetUniformLocation(phongProgram, "uInferred"), if (hasInferred) 1 else 0)
            drawMeshData(phongProgram, mesh)
        }

        // Palm
        val palmMesh = HandMeshBuilder.buildPalmGeo(pts, unitR)
        GLES30.glUniform1i(GLES30.glGetUniformLocation(phongProgram, "uInferred"), 0)
        drawMeshData(phongProgram, palmMesh)

        // Webbings
        for (wi in 0 until 3) {
            try {
                val webMesh = HandMeshBuilder.buildWebbingGeo(pts, wi)
                drawMeshData(phongProgram, webMesh)
            } catch (_: Exception) {}
        }

        // Nails
        GLES30.glUseProgram(basicProgram)
        val nailColor = floatArrayOf(0.9f, 0.82f, 0.78f, 0.85f)
        GLES30.glUniform4fv(GLES30.glGetUniformLocation(basicProgram, "uColor"), 1, nailColor, 0)
        setMatrixUniforms(basicProgram, model, view, proj)
        for (fi in HandMeshBuilder.FINGER_CHAINS.indices) {
            val chain = HandMeshBuilder.FINGER_CHAINS[fi]
            val tipIdx = chain.last()
            val nailMesh = HandMeshBuilder.buildNailGeo(computeUnitR(pts) * HandMeshBuilder.FINGER_PROFILES[fi].tip)
            // Translate nail to tip position
            val tipPos = pts[tipIdx]
            val tipModel = FloatArray(16).also { Matrix.setIdentityM(it, 0) }
            Matrix.translateM(tipModel, 0, tipPos.x, tipPos.y, tipPos.z)
            setMatrixUniforms(basicProgram, tipModel, view, proj)
            drawMeshData(basicProgram, nailMesh)
        }
    }

    private fun drawWireframe(pts: List<Vec3>, view: FloatArray, proj: FloatArray) {
        GLES30.glUseProgram(lineProgram)
        val model = FloatArray(16).also { Matrix.setIdentityM(it, 0) }
        setMatrixUniforms(lineProgram, model, view, proj)
        val wireColor = floatArrayOf(0f, 1f, 0.898f, 0.7f) // plasma
        GLES30.glUniform4fv(GLES30.glGetUniformLocation(lineProgram, "uColor"), 1, wireColor, 0)

        for (fi in HandMeshBuilder.FINGER_CHAINS.indices) {
            val chain = HandMeshBuilder.FINGER_CHAINS[fi]
            val fingerPts = chain.map { pts[it] }
            drawLineStrip(fingerPts)
        }
    }

    private fun drawSkeleton(pts: List<Vec3>, lms: HandLandmarks, view: FloatArray, proj: FloatArray, timeSec: Float) {
        GLES30.glUseProgram(lineProgram)
        val model = FloatArray(16).also { Matrix.setIdentityM(it, 0) }
        setMatrixUniforms(lineProgram, model, view, proj)
        val skelColor = floatArrayOf(0f, 1f, 0.898f, 0.9f)
        GLES30.glUniform4fv(GLES30.glGetUniformLocation(lineProgram, "uColor"), 1, skelColor, 0)

        val lineVerts = mutableListOf<Float>()
        val connections = listOf(
            0 to 1, 1 to 2, 2 to 3, 3 to 4,
            0 to 5, 5 to 6, 6 to 7, 7 to 8,
            0 to 9, 9 to 10, 10 to 11, 11 to 12,
            0 to 13, 13 to 14, 14 to 15, 15 to 16,
            0 to 17, 17 to 18, 18 to 19, 19 to 20,
            5 to 9, 9 to 13, 13 to 17
        )
        for ((a, b) in connections) {
            if (a < pts.size && b < pts.size) {
                lineVerts.add(pts[a].x); lineVerts.add(pts[a].y); lineVerts.add(pts[a].z)
                lineVerts.add(pts[b].x); lineVerts.add(pts[b].y); lineVerts.add(pts[b].z)
            }
        }
        drawLines(lineVerts.toFloatArray())
    }

    // ─── GL helpers ───────────────────────────────────────────────────────

    private fun computeUnitR(pts: List<Vec3>): Float {
        if (pts.size < 10) return 0.03f
        val w = pts[0]; val mk = pts[9]
        val d = (w - mk).length()
        return d * 0.18f
    }

    private fun setMatrixUniforms(prog: Int, model: FloatArray, view: FloatArray, proj: FloatArray) {
        GLES30.glUniformMatrix4fv(GLES30.glGetUniformLocation(prog, "uModel"),      1, false, model, 0)
        GLES30.glUniformMatrix4fv(GLES30.glGetUniformLocation(prog, "uView"),       1, false, view,  0)
        GLES30.glUniformMatrix4fv(GLES30.glGetUniformLocation(prog, "uProjection"), 1, false, proj,  0)
    }

    private fun setLightingUniforms(prog: Int, torchOn: Boolean, timeSec: Float, pts: List<Vec3>) {
        val L = LightingModel
        fun setV3(name: String, v: FloatArray) =
            GLES30.glUniform3fv(GLES30.glGetUniformLocation(prog, name), 1, v, 0)
        fun setF(name: String, v: Float) =
            GLES30.glUniform1f(GLES30.glGetUniformLocation(prog, name), v)
        fun setI(name: String, v: Int) =
            GLES30.glUniform1i(GLES30.glGetUniformLocation(prog, name), v)

        setV3("uAmbientColor",    L.ambientColor);   setF("uAmbientIntensity", L.ambientIntensity)
        setV3("uKeyDir",          L.keyDir);          setV3("uKeyColor", L.keyColor);   setF("uKeyIntensity",  L.keyIntensity)
        setV3("uRimDir",          L.rimDir);          setV3("uRimColor", L.rimColor);   setF("uRimIntensity",  L.rimIntensity)
        setV3("uSSSDir",          L.sssDir);          setV3("uSSSColor", L.sssColor);   setF("uSSSIntensity",  L.sssIntensity)
        setV3("uFillDir",         L.fillDir);         setV3("uFillColor", L.fillColor); setF("uFillIntensity", L.fillIntensity)

        // Torch position — index fingertip (landmark 8)
        if (pts.size > 8) {
            setV3("uTorchPos", floatArrayOf(pts[8].x, pts[8].y, pts[8].z))
        }
        setV3("uTorchColor", L.torchColor)
        setF("uTorchIntensity", L.torchIntensity)
        GLES30.glUniform3fv(GLES30.glGetUniformLocation(prog, "uTorchAttenuation"), 1, L.torchAttenuation, 0)
        setI("uTorchOn", if (torchOn) 1 else 0)

        setV3("uCameraPos", floatArrayOf(0f, 0f, 5f))
        setF("uTime", timeSec)
    }

    private fun drawMeshData(prog: Int, mesh: HandMeshBuilder.MeshData) {
        if (mesh.positions.isEmpty()) return

        GLES30.glBindVertexArray(meshVao[0])

        // Positions
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, meshVbo[0])
        val posBuf = toFloatBuffer(mesh.positions)
        GLES30.glBufferData(GLES30.GL_ARRAY_BUFFER, mesh.positions.size * 4, posBuf, GLES30.GL_DYNAMIC_DRAW)
        val posLoc = GLES30.glGetAttribLocation(prog, "aPosition")
        GLES30.glEnableVertexAttribArray(posLoc)
        GLES30.glVertexAttribPointer(posLoc, 3, GLES30.GL_FLOAT, false, 0, 0)

        // Normals
        if (mesh.normals.isNotEmpty()) {
            GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, meshVbo[1])
            val normBuf = toFloatBuffer(mesh.normals)
            GLES30.glBufferData(GLES30.GL_ARRAY_BUFFER, mesh.normals.size * 4, normBuf, GLES30.GL_DYNAMIC_DRAW)
            val normLoc = GLES30.glGetAttribLocation(prog, "aNormal")
            if (normLoc >= 0) {
                GLES30.glEnableVertexAttribArray(normLoc)
                GLES30.glVertexAttribPointer(normLoc, 3, GLES30.GL_FLOAT, false, 0, 0)
            }
        }

        // Indices
        GLES30.glBindBuffer(GLES30.GL_ELEMENT_ARRAY_BUFFER, meshIbo[0])
        val idxBuf = toIntBuffer(mesh.indices)
        GLES30.glBufferData(GLES30.GL_ELEMENT_ARRAY_BUFFER, mesh.indices.size * 4, idxBuf, GLES30.GL_DYNAMIC_DRAW)

        GLES30.glDrawElements(GLES30.GL_TRIANGLES, mesh.indices.size, GLES30.GL_UNSIGNED_INT, 0)

        GLES30.glBindVertexArray(0)
        GLES30.glBindBuffer(GLES30.GL_ELEMENT_ARRAY_BUFFER, 0)
    }

    private fun drawLineStrip(pts: List<Vec3>) {
        val verts = FloatArray(pts.size * 3)
        pts.forEachIndexed { i, p -> verts[i*3] = p.x; verts[i*3+1] = p.y; verts[i*3+2] = p.z }
        drawLines(verts, GLES30.GL_LINE_STRIP)
    }

    private fun drawLines(verts: FloatArray, mode: Int = GLES30.GL_LINES) {
        val buf = toFloatBuffer(verts)
        val vbo = IntArray(1); GLES30.glGenBuffers(1, vbo, 0)
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, vbo[0])
        GLES30.glBufferData(GLES30.GL_ARRAY_BUFFER, verts.size * 4, buf, GLES30.GL_DYNAMIC_DRAW)
        val loc = GLES30.glGetAttribLocation(lineProgram, "aPosition")
        GLES30.glEnableVertexAttribArray(loc)
        GLES30.glVertexAttribPointer(loc, 3, GLES30.GL_FLOAT, false, 0, 0)
        GLES30.glDrawArrays(mode, 0, verts.size / 3)
        GLES30.glDeleteBuffers(1, vbo, 0)
    }

    private fun toFloatBuffer(data: FloatArray): FloatBuffer =
        ByteBuffer.allocateDirect(data.size * 4).order(ByteOrder.nativeOrder()).asFloatBuffer().apply { put(data); position(0) }

    private fun toIntBuffer(data: IntArray): IntBuffer =
        ByteBuffer.allocateDirect(data.size * 4).order(ByteOrder.nativeOrder()).asIntBuffer().apply { put(data); position(0) }

    private fun compileProgram(vertSrc: String, fragSrc: String): Int {
        val vert = compileShader(GLES30.GL_VERTEX_SHADER, vertSrc)
        val frag = compileShader(GLES30.GL_FRAGMENT_SHADER, fragSrc)
        return GLES30.glCreateProgram().also {
            GLES30.glAttachShader(it, vert)
            GLES30.glAttachShader(it, frag)
            GLES30.glLinkProgram(it)
        }
    }

    private fun compileShader(type: Int, src: String): Int =
        GLES30.glCreateShader(type).also {
            GLES30.glShaderSource(it, src)
            GLES30.glCompileShader(it)
        }
}
