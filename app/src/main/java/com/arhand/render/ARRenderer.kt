package com.arhand.render

import android.graphics.Bitmap
import android.opengl.GLES30
import android.opengl.GLSurfaceView
import android.opengl.Matrix
import com.arhand.tracking.HandLandmarks
import com.arhand.util.PerfMonitor
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

/**
 * Main OpenGL ES 3.0 renderer.
 * GLSurfaceView.Renderer — runs on the GL thread, separate from tracking and UI threads.
 *
 * Pipeline: onDrawFrame → drawCameraPassthrough → updateHandMeshes → drawScene
 * Camera: Orthographic (matches THREE.OrthographicCamera from HTML prototype)
 * MVP: calculated manually via Matrix.orthoM + aspect ratio
 */
class ARRenderer(private val perfMonitor: PerfMonitor) : GLSurfaceView.Renderer {

    // Shared state updated from tracking thread (volatile for visibility)
    @Volatile var handsData: List<HandLandmarks> = emptyList()
    @Volatile var renderMode: RenderMode = RenderMode.MESH
    @Volatile var torchOn: Boolean = true
    @Volatile var showCloud: Boolean = false
    @Volatile var mirrorX: Boolean = true
    @Volatile var scanCloudPoints: FloatArray? = null
    @Volatile var depthMeshPositions: FloatArray? = null

    private val handRenderers = arrayOf(HandRenderer(), HandRenderer())
    private val pointCloudRenderer = PointCloudRenderer()
    private val depthMeshRenderer = DepthMeshRenderer()
    private val cameraPassthrough = CameraPassthroughRenderer()

    private val projMatrix = FloatArray(16)
    private val viewMatrix = FloatArray(16)
    private var aspect = 1f

    private var timeMs = 0L

    /**
     * Called from the tracking coroutine whenever a new camera Bitmap is available.
     * Thread-safe: just sets a volatile reference; the GL thread reads it in onDrawFrame.
     */
    fun submitCameraFrame(bitmap: Bitmap) {
        cameraPassthrough.latestBitmap = bitmap
    }

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        // Transparent clear — passthrough quad covers the background
        GLES30.glClearColor(0f, 0f, 0f, 0f)
        GLES30.glEnable(GLES30.GL_DEPTH_TEST)
        GLES30.glEnable(GLES30.GL_BLEND)
        GLES30.glBlendFunc(GLES30.GL_SRC_ALPHA, GLES30.GL_ONE_MINUS_SRC_ALPHA)

        cameraPassthrough.init()
        handRenderers.forEach { it.init() }
        pointCloudRenderer.init()
        depthMeshRenderer.init()

        // Identity view matrix — camera at origin looking down -Z
        Matrix.setLookAtM(viewMatrix, 0,
            0f, 0f, 5f,   // eye
            0f, 0f, 0f,   // center
            0f, 1f, 0f    // up
        )
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        GLES30.glViewport(0, 0, width, height)
        aspect = width.toFloat() / height.toFloat()
        // Orthographic projection — matches THREE.OrthographicCamera(-aspect, aspect, 1, -1)
        Matrix.orthoM(projMatrix, 0, -aspect, aspect, -1f, 1f, -10f, 10f)
    }

    override fun onDrawFrame(gl: GL10?) {
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT or GLES30.GL_DEPTH_BUFFER_BIT)

        // 1. Camera passthrough — draws behind everything, no depth write
        cameraPassthrough.mirrorX = mirrorX
        cameraPassthrough.draw()

        // 2. Hand mesh / wireframe / skeleton on top
        timeMs = System.currentTimeMillis()
        val timeSec = (timeMs % 100000L) / 1000f

        val hands = handsData
        for (i in hands.indices) {
            val lms = hands[i]
            handRenderers[i].update(lms, aspect, mirrorX)
            handRenderers[i].draw(viewMatrix, projMatrix, renderMode, torchOn, timeSec)
        }
        // Clear unused slots
        for (i in hands.size until handRenderers.size) {
            handRenderers[i].clear()
        }

        if (showCloud) {
            scanCloudPoints?.let { pointCloudRenderer.update(it) }
            pointCloudRenderer.draw(viewMatrix, projMatrix)
        }

        depthMeshPositions?.let {
            depthMeshRenderer.update(it)
            depthMeshRenderer.draw(viewMatrix, projMatrix, renderMode)
        }

        perfMonitor.onFrame()
    }
}

enum class RenderMode { MESH, WIREFRAME, SKELETON }
