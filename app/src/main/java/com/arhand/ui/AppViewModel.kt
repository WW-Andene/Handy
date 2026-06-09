package com.arhand.ui

import android.app.Application
import android.graphics.Bitmap
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.arhand.camera.CameraController
import com.arhand.camera.CameraFrameProvider
import com.arhand.depth.DepthCarver
import com.arhand.export.GLBExporter
import com.arhand.export.GltfAnimationExporter
import com.arhand.export.PointCloudExporter
import com.arhand.render.ARRenderer
import com.arhand.render.RenderMode
import com.arhand.scanner.CLAHEAnalyzer
import com.arhand.mocap.AssetLoader
import com.arhand.mocap.BoneRetargeter
import com.arhand.mocap.LoadedAsset
import com.arhand.mocap.MotionRecorder
import com.arhand.mocap.RetargetResult
import com.arhand.tracking.HandPipeline
import com.arhand.tracking.HandTracker
import com.arhand.tracking.HandLandmarks
import com.arhand.util.FrameThrottler
import com.arhand.util.PerfMonitor
import com.arhand.util.Vec3
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch
import java.io.File

import com.arhand.scanner.HandBiometrics

data class AppUiState(
    val renderMode: RenderMode = RenderMode.MESH,
    val torchOn: Boolean = true,
    val showCloud: Boolean = false,
    val isFrontCamera: Boolean = true,
    val scanActive: Boolean = false,
    val exportedGlbPath: String? = null,
    val hasStoredModel: Boolean = false,
    val showSplash: Boolean = true,
    val depthMode: Boolean = false,
    /** Biometrics computed after the most recent scan. Null until first scan completes. */
    val handBiometrics: HandBiometrics? = null,
    /** True while an asset GLB is being loaded from storage. */
    val assetLoading: Boolean = false,
    /** Display name of the currently loaded asset, or null for the default puppet. */
    val loadedAssetName: String? = null,
    /** True while [MotionRecorder] is actively recording. */
    val isRecording: Boolean = false,
    /** Absolute path of the last exported BVH file, or null. */
    val exportedBvhPath: String? = null,
    /** Absolute path of the last exported animated GLB file, or null. */
    val exportedGltfPath: String? = null
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

    /**
     * Real surface aspect ratio (width / height) published from the GL thread via
     * [ARRenderer.onSurfaceChanged]. Consumed by the scanner coroutine so we never
     * pass a hardcoded portrait constant.
     *
     * Default: 9/16 portrait — overwritten as soon as the surface is sized.
     */
    val currentAspect: MutableStateFlow<Float> = MutableStateFlow(9f / 16f)

    private val frameProvider = CameraFrameProvider()
    lateinit var cameraController: CameraController
    private var handTracker: HandTracker? = null

    // Captured world-space landmark frames for depth carving
    private val capturedFrames = mutableListOf<List<Vec3>>()

    // Landmark frames paired with quality scores — used for biometrics computation
    private val biometricFrames = mutableListOf<Pair<HandLandmarks, Float>>()

    // Latest mesh produced by processScan() — exposed for the 3D viewer
    val depthMeshPositions: MutableStateFlow<FloatArray> = MutableStateFlow(FloatArray(0))

    // ── Mocap state ────────────────────────────────────────────────────────
    /** Currently loaded asset. Starts as the default puppet (no mesh, symmetric bind pose). */
    private var loadedAsset: LoadedAsset = AssetLoader.DEFAULT_PUPPET

    /** Retargeter instance — rebuilt whenever [loadedAsset] changes. */
    private var boneRetargeter: BoneRetargeter = BoneRetargeter(AssetLoader.DEFAULT_PUPPET.bindPose)

    /** Most recent retarget result — updated every tracking frame. Null until first frame. */
    val latestRetargetResult: MutableStateFlow<RetargetResult?> = MutableStateFlow(null)

    /** Motion recorder — buffers joint rotations for BVH export. */
    val motionRecorder = MotionRecorder()

    init {
        // Publish aspect changes from the GL thread into currentAspect
        renderer.onAspectChanged = { aspect ->
            currentAspect.value = aspect
        }

        // Wire tracking results → pipeline
        viewModelScope.launch {
            handPipeline.processed.collect { hands ->
                renderer.handsData = hands.map { it.landmarks }
                renderer.mirrorX = uiState.value.isFrontCamera

                // Retarget the primary hand onto the loaded asset each frame
                val primary = hands.firstOrNull()
                primary?.let { hand ->
                    val result = boneRetargeter.retarget(
                        lms     = hand.landmarks,
                        aspect  = currentAspect.value,
                        mirrorX = uiState.value.isFrontCamera
                    )
                    if (result != null) {
                        latestRetargetResult.value    = result
                        renderer.latestRetargetResult = result  // forward to GL thread
                        motionRecorder.pushFrame(result)        // record if active
                    }
                }

                // Feed scanner if active — use real device aspect
                if (uiState.value.scanActive) {
                    val aspect  = currentAspect.value
                    scanner.update(primary?.landmarks, clahe.lastContrastScore, System.currentTimeMillis(), aspect)

                    // Capture frames for depth carving
                    primary?.landmarks?.let { lms ->
                        if (lms.size == 21) {
                            capturedFrames.add(DepthCarver.landmarksToWorld(lms, aspect))
                            // Accumulate for biometrics — sample every 3rd frame to reduce
                            // redundancy while still getting coverage across the pose hold
                            if (capturedFrames.size % 3 == 0) {
                                val quality = scanner.status.value.quality
                                biometricFrames.add(Pair(lms, quality))
                            }
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
                renderer.submitCameraFrame(bitmap)

                if (frameThrottler.shouldInfer()) {
                    val t0 = System.currentTimeMillis()
                    clahe.process(bitmap)   // updates clahe.lastContrastScore; return value unused
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
        biometricFrames.clear()
        scanner.start()
        uiState.update { it.copy(scanActive = true) }
    }

    fun cancelScan() {
        scanner.reset()
        capturedFrames.clear()
        biometricFrames.clear()
        uiState.update { it.copy(scanActive = false) }
    }

    private fun processScan() {
        viewModelScope.launch(Dispatchers.Default) {
            try {
                val meshPos = DepthCarver.carveAndExtract(capturedFrames)
                renderer.depthMeshPositions = meshPos
                depthMeshPositions.value    = meshPos   // expose for 3D viewer

                val glbFile = GLBExporter.export(
                    getApplication(), meshPos, scanner.cloudPoints
                )

                val scoresJson = scanner.status.value.poseScores
                    .joinToString(",", prefix="[", postfix="]") { "%.3f".format(it) }

                // Compute hand biometrics from the accumulated landmark frames
                val biometrics = HandBiometrics.compute(
                    frames = biometricFrames.toList(),
                    aspect = currentAspect.value
                )

                modelStore.saveModel(glbFile, scanner.cloudPoints.size, scoresJson, biometrics)

                scanner.markDone()
                uiState.update {
                    it.copy(
                        scanActive      = false,
                        exportedGlbPath = glbFile.absolutePath,
                        hasStoredModel  = true,
                        handBiometrics  = biometrics
                    )
                }
            } catch (e: Exception) {
                scanner.markFailed()
                uiState.update { it.copy(scanActive = false) }
            }
        }
    }

    /**
     * Load a user GLB asset from [uri] on the IO dispatcher, then rebuild [boneRetargeter].
     * Pass null to revert to the default puppet.
     *
     * Called from the UI layer after the file picker returns a result:
     * ```kotlin
     * val launcher = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri ->
     *     viewModel.loadAsset(uri, uri?.lastPathSegment)
     * }
     * launcher.launch("model/gltf-binary")
     * ```
     */
    fun loadAsset(uri: Uri?, displayName: String?) {
        viewModelScope.launch(Dispatchers.IO) {
            uiState.update { it.copy(assetLoading = true) }
            try {
                val asset = AssetLoader.load(getApplication(), uri)
                loadedAsset   = asset
                boneRetargeter = BoneRetargeter(asset.bindPose)
                renderer.loadedAsset = asset   // forward to GL thread
                uiState.update {
                    it.copy(
                        assetLoading    = false,
                        loadedAssetName = if (uri == null) null else (displayName ?: "asset.glb")
                    )
                }
            } catch (e: Exception) {
                loadedAsset   = AssetLoader.DEFAULT_PUPPET
                boneRetargeter = BoneRetargeter(AssetLoader.DEFAULT_PUPPET.bindPose)
                renderer.loadedAsset = null
                uiState.update { it.copy(assetLoading = false, loadedAssetName = null) }
            }
        }
    }

    fun shareGlb() {
        val path = uiState.value.exportedGlbPath ?: return
        // Share intent launched from UI layer
    }

    // ─── Motion recording ─────────────────────────────────────────────────────

    /**
     * Start a new BVH recording session. Clears any previously buffered frames.
     * The REC button in [ControlPanel] calls this.
     */
    fun startRecording() {
        motionRecorder.start()
        uiState.update { it.copy(isRecording = true, exportedBvhPath = null) }
    }

    /**
     * Stop recording and export the buffered frames as both a BVH file and an
     * animated GLB file (glTF animation track embedded in the asset).
     *
     * Both exports run on [Dispatchers.IO]. [AppUiState.exportedBvhPath] and
     * [AppUiState.exportedGltfPath] are updated when each file is written.
     * Files are written to the app's external "mocap" directory.
     */
    fun stopRecordingAndExport() {
        motionRecorder.stop()
        uiState.update { it.copy(isRecording = false) }

        viewModelScope.launch(Dispatchers.IO) {
            val outputDir = getApplication<android.app.Application>()
                .getExternalFilesDir("mocap") ?: return@launch
            outputDir.mkdirs()

            val snapshot = motionRecorder.frameSnapshot()
            val label    = "take_${System.currentTimeMillis() / 1000}"

            // BVH export
            val bvhFile = motionRecorder.exportBvh(outputDir, label)
            if (bvhFile != null) {
                uiState.update { it.copy(exportedBvhPath = bvhFile.absolutePath) }
            }

            // glTF animated GLB export
            val gltfFile = GltfAnimationExporter.export(
                frames    = snapshot,
                asset     = loadedAsset,
                outputDir = outputDir,
                label     = "${label}_anim"
            )
            if (gltfFile != null) {
                uiState.update { it.copy(exportedGltfPath = gltfFile.absolutePath) }
            }
        }
    }

    override fun onCleared() {
        handTracker?.close()
        cameraController.stop()
        super.onCleared()
    }
}
