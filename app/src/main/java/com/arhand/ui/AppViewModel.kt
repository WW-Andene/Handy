package com.arhand.ui

import android.app.Application
import android.graphics.Bitmap
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.arhand.camera.CameraController
import com.arhand.camera.CameraFrameProvider
import com.arhand.depth.DepthCarver
import com.arhand.export.GLBExporter
import com.arhand.export.PointCloudExporter
import com.arhand.render.ARRenderer
import com.arhand.render.RenderMode
import com.arhand.scanner.CLAHEAnalyzer
import com.arhand.scanner.PersonalModelStore
import com.arhand.scanner.Scanner
import com.arhand.tracking.HandPipeline
import com.arhand.tracking.HandTracker
import com.arhand.util.FrameThrottler
import com.arhand.util.PerfMonitor
import com.arhand.util.Vec3
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch
import java.io.File

data class AppUiState(
    val renderMode: RenderMode = RenderMode.MESH,
    val torchOn: Boolean = true,
    val showCloud: Boolean = false,
    val isFrontCamera: Boolean = true,
    val scanActive: Boolean = false,
    val exportedGlbPath: String? = null,
    val hasStoredModel: Boolean = false,
    val showSplash: Boolean = true,
    val depthMode: Boolean = false
)

class AppViewModel(app: Application) : AndroidViewModel(app) {

    val uiState = MutableStateFlow(AppUiState())

    val perfMonitor    = PerfMonitor()
    val handPipeline   = HandPipeline()
    val scanner        = Scanner()
    val clahe          = CLAHEAnalyzer()
    val frameThrottler = FrameThrottler()
    val modelStore     = PersonalModelStore(app)

    val renderer = ARRenderer(perfMonitor)

    private val frameProvider = CameraFrameProvider()
    lateinit var cameraController: CameraController
    private var handTracker: HandTracker? = null

    // Captured world-space landmark frames for depth carving
    private val capturedFrames = mutableListOf<List<Vec3>>()

    init {
        // Wire tracking results → pipeline
        viewModelScope.launch {
            handPipeline.processed.collect { hands ->
                renderer.handsData = hands.map { it.landmarks }
                renderer.mirrorX = uiState.value.isFrontCamera

                // Feed scanner if active
                if (uiState.value.scanActive) {
                    val primary = hands.firstOrNull()
                    val aspect = 9f / 16f  // portrait
                    scanner.update(primary?.landmarks, clahe.lastContrastScore, System.currentTimeMillis(), aspect)

                    // Capture frames for depth carving
                    primary?.landmarks?.let { lms ->
                        if (lms.size == 21) {
                            capturedFrames.add(DepthCarver.landmarksToWorld(lms, aspect))
                        }
                    }
                }
            }
        }

        // Watch scanner state → trigger processing when done
        viewModelScope.launch {
            scanner.status.collect { status ->
                when (status.state) {
                    Scanner.ScanState.PROCESSING -> processScan()
                    else -> {}
                }
            }
        }

        viewModelScope.launch {
            uiState.value = uiState.value.copy(hasStoredModel = modelStore.hasModel())
        }
    }

    fun initCamera() {
        cameraController = CameraController(getApplication(), frameProvider)
        handTracker = HandTracker(getApplication()) { result, timestamp ->
            val rawHands = handTracker!!.parseResult(result)
            val conf = if (result.handedness().isNotEmpty())
                result.handedness()[0].firstOrNull()?.score() ?: 0.5f
            else 0f
            handPipeline.update(rawHands, conf, timestamp)
            perfMonitor.onFrame()
        }
        handTracker!!.init()

        // Wire frame provider → renderer passthrough + CLAHE + tracker
        viewModelScope.launch(Dispatchers.Default) {
            frameProvider.frames.collect { bitmap ->
                // Every frame feeds the camera passthrough regardless of inference throttle
                renderer.submitCameraFrame(bitmap)

                if (frameThrottler.shouldInfer()) {
                    val t0 = System.currentTimeMillis()
                    clahe.process(bitmap)
                    handTracker!!.detect(bitmap, System.currentTimeMillis())
                    perfMonitor.onInference((System.currentTimeMillis() - t0).toFloat())
                    frameThrottler.reportInferenceMs((System.currentTimeMillis() - t0).toFloat())
                }
            }
        }

        cameraController.start()
        uiState.value = uiState.value.copy(showSplash = false)
    }

    fun toggleRenderMode() {
        val next = when (renderer.renderMode) {
            RenderMode.MESH      -> RenderMode.WIREFRAME
            RenderMode.WIREFRAME -> RenderMode.SKELETON
            RenderMode.SKELETON  -> RenderMode.MESH
        }
        renderer.renderMode = next
        uiState.update { it.copy(renderMode = next) }
    }

    fun toggleTorch() {
        val next = !uiState.value.torchOn
        renderer.torchOn = next
        uiState.update { it.copy(torchOn = next) }
    }

    fun toggleCloud() {
        uiState.update { it.copy(showCloud = !it.showCloud) }
        renderer.showCloud = uiState.value.showCloud
    }

    fun switchCamera() {
        cameraController.switchCamera()
        uiState.update { it.copy(isFrontCamera = cameraController.isFrontFacing()) }
    }

    fun startScan() {
        capturedFrames.clear()
        scanner.start()
        uiState.update { it.copy(scanActive = true) }
    }

    fun cancelScan() {
        scanner.reset()
        capturedFrames.clear()
        uiState.update { it.copy(scanActive = false) }
    }

    private fun processScan() {
        viewModelScope.launch(Dispatchers.Default) {
            try {
                // Run marching cubes
                val meshPositions = DepthCarver.carveAndExtract(capturedFrames)
                renderer.depthMeshPositions = meshPositions

                // Export GLB
                val glbFile = GLBExporter.export(
                    getApplication(), meshPositions, scanner.cloudPoints
                )

                // Persist model metadata
                val scoresJson = scanner.status.value.poseScores
                    .joinToString(",", prefix="[", postfix="]") { "%.3f".format(it) }
                modelStore.saveModel(glbFile, scanner.cloudPoints.size, scoresJson)

                scanner.markDone()
                uiState.update {
                    it.copy(
                        scanActive = false,
                        exportedGlbPath = glbFile.absolutePath,
                        hasStoredModel = true
                    )
                }
            } catch (e: Exception) {
                scanner.markFailed()
                uiState.update { it.copy(scanActive = false) }
            }
        }
    }

    fun shareGlb() {
        val path = uiState.value.exportedGlbPath ?: return
        // Share intent launched from UI layer
    }

    override fun onCleared() {
        handTracker?.close()
        cameraController.stop()
        super.onCleared()
    }
}
