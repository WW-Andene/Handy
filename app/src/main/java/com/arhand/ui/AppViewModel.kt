package com.arhand.ui

import android.app.Application
import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.arhand.camera.CameraController
import com.arhand.camera.CameraFrameProvider
import com.arhand.depth.ArDepthAvailability
import com.arhand.depth.ArDepthSession
import com.arhand.depth.DepthApiCarver
import com.arhand.depth.DepthCarver
import com.arhand.depth.HandSegmentationMask
import com.arhand.depth.MeshSmoother
import com.arhand.depth.NeuralImplicitCarver
import com.arhand.depth.ManoShapeFitter
import com.arhand.export.GLBExporter
import com.arhand.export.GltfAnimationExporter
import com.arhand.export.PointCloudExporter
import com.arhand.render.ARRenderer
import com.arhand.render.RenderMode
import com.arhand.scanner.CLAHEAnalyzer
import com.arhand.scanner.PersonalModelStore
import com.arhand.scanner.Scanner
import com.arhand.scanner.PhotometricStereoCapture
import com.arhand.mocap.AssetLoader
import com.arhand.mocap.BoneRetargeter
import com.arhand.mocap.LoadedAsset
import com.arhand.mocap.MotionRecorder
import com.arhand.mocap.OscStreamer
import com.arhand.mocap.OscSchema
import com.arhand.mocap.OscMode
import com.arhand.mocap.OscReceiver
import com.arhand.mocap.RetargetResult
import com.arhand.tracking.HandPipeline
import com.arhand.tracking.HandTrackerManager
import com.arhand.tracking.HandTracker
import com.arhand.tracking.HandLandmarks
import com.arhand.tracking.BodyPipeline
import com.arhand.tracking.FacePipeline
import com.arhand.tracking.FullBodyFrame
import com.arhand.tracking.Gesture
import com.arhand.util.FrameThrottler
import com.arhand.util.PerfMonitor
import com.arhand.util.Vec3
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.preferencesDataStore
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import com.arhand.scanner.BiometricHistoryEntry
import com.arhand.scanner.HandBiometrics
import com.arhand.render.LiveMeshDeformer

data class AppUiState(
    val renderMode: RenderMode = RenderMode.MESH,
    val torchOn: Boolean = true,
    val showCloud: Boolean = false,
    val isFrontCamera: Boolean = true,
    val scanActive: Boolean = false,
    val exportedGlbPath: String? = null,
    val hasStoredModel: Boolean = false,
    val showSplash: Boolean = true,
    /** E6 — True on first launch until the user completes or skips onboarding. */
    val showOnboarding: Boolean = false,
    val depthMode: Boolean = false,
    /** B6 — true once ArDepthAvailability.isDepthApiSupported() confirms ARCore Depth API support. */
    val depthApiAvailable: Boolean = false,
    /** B6 — mean confidence (0..1) of the latest fused ARCore depth point cloud. */
    val depthConfidence: Float = 0f,
    /** Biometrics computed after the most recent scan. Null until first scan completes. */
    val handBiometrics: HandBiometrics? = null,
    /** D4 — Biometric history list, newest-first, max 20 entries. */
    val biometricHistory: List<BiometricHistoryEntry> = emptyList(),
    /** D5 — Joint range-of-motion result from the most recent scan. Null until first scan. */
    val jointRomData: com.arhand.scanner.JointRomData? = null,
    /** True while an asset GLB is being loaded from storage. */
    val assetLoading: Boolean = false,
    /** Display name of the currently loaded asset, or null for the default puppet. */
    val loadedAssetName: String? = null,
    /** True while [MotionRecorder] is actively recording. */
    val isRecording: Boolean = false,
    /** Absolute path of the last exported BVH file, or null. */
    val exportedBvhPath: String? = null,
    /** Absolute path of the last exported animated GLB file, or null. */
    val exportedGltfPath: String? = null,
    /** True while OSC streaming is active. */
    val isOscStreaming: Boolean = false,
    /** Current OSC target host (IP or hostname). */
    val oscHost: String = OscStreamer.DEFAULT_HOST,
    /** Current OSC target port. */
    val oscPort: Int = OscStreamer.DEFAULT_PORT,
    /** Active OSC address schema preset. */
    val oscSchema: OscSchema = OscSchema.HANDY_DEFAULT,
    /** True when body pose tracking is active. */
    val bodyTrackingEnabled: Boolean = false,
    /** True when face tracking is active. */
    val faceTrackingEnabled: Boolean = false,
    /** G3 — OSC operating mode: SEND (broadcast) or RECEIVE (listen). */
    val oscMode: OscMode = OscMode.SEND,
    /** G3 — True when OSC receiver is active and listening for remote hand data. */
    val isOscReceiving: Boolean = false,
    /** G3 — UDP port the OSC receiver is listening on. */
    val oscReceivePort: Int = OscReceiver.DEFAULT_PORT,
    /** G4 — Currently recognized gesture on the primary hand slot, or null. */
    val activeGesture: Gesture? = null,
    /**
     * G1 — True when the live-deforming scanned mesh ("mirror effect") is active.
     * Requires a completed scan ([hasStoredModel] == true) to have mesh data to deform.
     */
    val liveMeshActive: Boolean = false,
    /**
     * G2 — True when neural implicit reconstruction is enabled.
     * When true, [NeuralImplicitCarver] is trained per-scan and used instead of
     * [DepthCarver] for the final mesh. Falls back to MANO shape-fitted capsule SDF
     * if the MLP produces an empty result.
     */
    val neuralReconEnabled: Boolean = false,
    /**
     * G2 — Diagnostics from the last neural reconstruction training pass.
     * Null until the first neural-enabled scan completes.
     */
    val neuralReconDiagnostics: NeuralReconDiagnostics? = null,
    /**
     * G5 — True when photometric stereo flash capture is active.
     * Requires torch availability (rear camera). When enabled during a scan,
     * the torch is toggled each frame to accumulate lit/unlit pairs and derive
     * a high-frequency surface normal map.
     */
    val photoStereoEnabled: Boolean = false,
    /**
     * G5 — Number of lit/unlit frame pairs accumulated during the current (or
     * most recent) scan. Resets to 0 at scan start. Used to drive a progress
     * indicator in [DepthScannerPanel].
     */
    val photoStereoFrameCount: Int = 0,
    /**
     * G5 — True once the photometric stereo capture has accumulated
     * [PhotometricStereoCapture.TARGET_PAIRS] pairs in the current scan.
     */
    val photoStereoComplete: Boolean = false
)

/**
 * G2 — Training diagnostics exposed to the UI.
 * @param trainLoss       Final binary cross-entropy loss after training.
 * @param trainEpochs     Number of epochs trained.
 * @param shapeScale      Per-segment radius multipliers from MANO shape fitting.
 * @param meshVertexCount Number of triangle vertices in the neural mesh.
 */
data class NeuralReconDiagnostics(
    val trainLoss: Float,
    val trainEpochs: Int,
    val shapeScale: FloatArray?,
    val meshVertexCount: Int
)

class AppViewModel(app: Application) : AndroidViewModel(app) {

    // ─── E6: Onboarding DataStore ─────────────────────────────────────────────
    private val Context.onboardingDataStore: DataStore<Preferences>
        by preferencesDataStore(name = "handy_onboarding")

    private val KEY_ONBOARDING_SEEN = booleanPreferencesKey("onboarding_seen")

    /** Persist dismissal and hide the onboarding screen. */
    fun dismissOnboarding() {
        viewModelScope.launch(Dispatchers.IO) {
            getApplication<Application>().onboardingDataStore.edit { prefs ->
                prefs[KEY_ONBOARDING_SEEN] = true
            }
        }
        uiState.value = uiState.value.copy(showOnboarding = false)
    }

    val uiState = MutableStateFlow(AppUiState())

    val perfMonitor    = PerfMonitor()
    val handPipeline   = HandPipeline()
    val scanner        = Scanner()
    val clahe          = CLAHEAnalyzer()
    val frameThrottler = FrameThrottler()
    val modelStore     = PersonalModelStore(app)

    /** G5 — Photometric stereo capture engine. */
    val photoStereoCapture = PhotometricStereoCapture()

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
    private var handTrackerManager: HandTrackerManager? = null

    // Captured world-space landmark frames for depth carving — paired with quality scores for B1
    private val capturedFrames = mutableListOf<Pair<List<Vec3>, Float>>()

    // G5 — Snapshot of torchOn at scan start. processScan() runs on Dispatchers.Default
    // and must restore the torch after stereo capture. Reading uiState there risks a
    // stale value if the user toggled torch mid-scan. We capture the ground truth here.
    private var torchOnAtScanStart: Boolean = true

    // B6 — Per-frame fused ARCore depth point clouds (world-space), captured during
    // CAPTURING when depthMode is active. Used by DepthApiCarver instead of the
    // capsule-SDF DepthCarver path.
    private val capturedDepthFrames = mutableListOf<List<Vec3>>()

    // B6 — ARCore Depth API session. Created lazily once availability is confirmed.
    private var arDepthSession: ArDepthSession? = null

    // Landmark frames paired with quality scores — used for biometrics computation
    private val biometricFrames = mutableListOf<Pair<HandLandmarks, Float>>()

    // Latest mesh produced by processScan() — exposed for the 3D viewer
    val depthMeshPositions: MutableStateFlow<FloatArray> = MutableStateFlow(FloatArray(0))

    // ── G1 — Live mesh deformation ────────────────────────────────────────
    /**
     * CPU-side LBS deformer. Holds the skinning weights computed from the scanned mesh
     * + the rest-pose joint positions captured at scan time.
     */
    private val liveMeshDeformer = LiveMeshDeformer()

    /**
     * Rest-pose joint positions captured at the moment the scan completes.
     * Flat float array: JOINT_COUNT × 3 floats (world-space xyz per joint).
     * Null until the first scan completes.
     */
    private var restJointPositions: FloatArray? = null

    // ── Mocap state ────────────────────────────────────────────────────────
    /** Currently loaded asset. Starts as the default puppet (no mesh, symmetric bind pose). */
    private var loadedAsset: LoadedAsset = AssetLoader.DEFAULT_PUPPET

    /** Retargeter instance — rebuilt whenever [loadedAsset] changes. */
    private var boneRetargeter: BoneRetargeter = BoneRetargeter(AssetLoader.DEFAULT_PUPPET.bindPose)

    /** Most recent retarget result — updated every tracking frame. Null until first frame. */
    val latestRetargetResult: MutableStateFlow<RetargetResult?> = MutableStateFlow(null)

    /** Motion recorder — buffers joint rotations for BVH export. */
    val motionRecorder = MotionRecorder()

    /** OSC streamer — broadcasts live joint rotations over UDP. */
    val oscStreamer = OscStreamer()

    /** G3 — OSC receiver — listens for incoming hand pose data from a remote Handy device. */
    val oscReceiver = OscReceiver()

    /** Body pose pipeline — disabled until [enableBodyTracking] is called. */
    val bodyPipeline = BodyPipeline()

    /** Face landmark pipeline — disabled until [enableFaceTracking] is called. */
    val facePipeline = FacePipeline()

    /** Latest unified full-body frame combining pose + face + both hands. */
    val latestFullBodyFrame: MutableStateFlow<FullBodyFrame?> = MutableStateFlow(null)

    init {
        // B6 — Check ARCore Depth API availability (off main thread; Session creation
        // can take tens of milliseconds). Only affects whether DepthScannerPanel offers
        // the toggle — the capsule-SDF DepthCarver path remains the default/fallback.
        viewModelScope.launch(Dispatchers.IO) {
            val available = ArDepthAvailability.isDepthApiSupported(getApplication())
            withContext(Dispatchers.Main) {
                uiState.update { it.copy(depthApiAvailable = available) }
            }
        }

        // E6 — Check if onboarding has been seen; show it on first launch
        viewModelScope.launch(Dispatchers.IO) {
            val prefs = getApplication<Application>().onboardingDataStore.data.first()
            val seen  = prefs[KEY_ONBOARDING_SEEN] ?: false
            if (!seen) {
                withContext(Dispatchers.Main) {
                    uiState.value = uiState.value.copy(showOnboarding = true)
                }
            }
        }

        // Publish aspect changes from the GL thread into currentAspect
        renderer.onAspectChanged = { aspect ->
            currentAspect.value = aspect
        }

        // B6 — Wire ARCore depth session lifecycle to the GL thread.
        // setCameraTextureName must be called once after EGL context creation.
        renderer.onGlSurfaceCreated = { textureId ->
            arDepthSession?.setCameraTextureName(textureId)
        }
        // Session.update() must be called from onDrawFrame. We call it here only
        // when depth mode is active to avoid ARCore overhead when unused.
        renderer.onDepthFrameTick = {
            val session = arDepthSession
            if (session != null && uiState.value.depthMode) {
                try {
                    val frame = session.session?.update()
                    if (frame != null) session.update(frame)
                } catch (_: Throwable) {}
            }
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
                        oscStreamer.sendFrame(result)           // stream if active

                        // G1 — Live mesh deformation: deform on tracking thread, push to GL thread
                        if (uiState.value.liveMeshActive && liveMeshDeformer.hasMesh()) {
                            val deformed = liveMeshDeformer.deform(result)
                            renderer.liveMeshPositions = deformed
                        }
                    }
                }

                // Feed scanner if active — use real device aspect
                if (uiState.value.scanActive) {
                    val aspect  = currentAspect.value
                    scanner.update(primary?.landmarks, clahe.lastContrastScore, System.currentTimeMillis(), aspect, uiState.value.isFrontCamera)

                    // Capture frames for depth carving — B1: store quality score per frame
                    // Only accumulate during CAPTURING; PREFLIGHT/COUNTDOWN frames have quality=0f
                    // and would dilute the SDF carver's top-75% quality filter.
                    if (uiState.value.scanActive &&
                        scanner.status.value.state == Scanner.ScanState.CAPTURING) {
                        primary?.landmarks?.let { lms ->
                            if (lms.size == 21) {
                                val quality = scanner.status.value.quality
                                capturedFrames.add(Pair(DepthCarver.landmarksToWorld(lms, aspect, mirrorX = uiState.value.isFrontCamera), quality))
                                // Accumulate for biometrics — sample every 3rd frame to reduce
                                // redundancy while still getting coverage across the pose hold
                                if (capturedFrames.size % 3 == 0) {
                                    biometricFrames.add(Pair(lms, quality))
                                }

                                // B6 — When depth mode is active, also capture the latest
                                // fused ARCore depth point cloud for this frame.
                                if (uiState.value.depthMode) {
                                    val session = arDepthSession
                                    val cloud = session?.depthPointCloud
                                    if (!cloud.isNullOrEmpty()) {
                                        // B7 — discard background points outside the hand's
                                        // convex hull (cheap segmentation mask).
                                        val hull = HandSegmentationMask.buildHull(
                                            lms, aspect, mirrorX = uiState.value.isFrontCamera
                                        )
                                        val masked = HandSegmentationMask.filterPointCloud(cloud, hull)
                                        capturedDepthFrames.add(masked)
                                        uiState.update { it.copy(depthConfidence = session.depthConfidence) }
                                    }
                                }
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
            val history = modelStore.loadHistory()
            uiState.value = uiState.value.copy(
                hasStoredModel   = modelStore.hasModel(),
                biometricHistory = history
            )
        }
    }

    fun initCamera() {
        cameraController = CameraController(getApplication(), frameProvider)
        handTrackerManager = HandTrackerManager(getApplication()) { rawHands, conf, timestamp ->
            handPipeline.update(rawHands, conf, timestamp)
        }
        handTrackerManager!!.init()

        // Wire frame provider → renderer passthrough + CLAHE + tracker
        viewModelScope.launch(Dispatchers.Default) {
            // G5 — lastStereoLit tracks the torch state that was active when the
            // *previous* frame was captured. The camera pipeline has 1–3 frames of
            // latency, so the bitmap we receive NOW was captured under the torch state
            // we set LAST frame. We tag accordingly, then toggle for the next frame.
            // Starts as true (torch ON for first stereo frame, OFF for second, etc.).
            var lastStereoLit = true
            var inStereoMode  = false   // tracks whether stereo was active last frame

            frameProvider.frames.collect { bitmap ->
                renderer.submitCameraFrame(bitmap)

                // G5 — Photometric stereo: tag current frame with the torch state that
                // was active when it was captured (lastStereoLit), then toggle torch for
                // the next frame.
                val stereoCondition = uiState.value.photoStereoEnabled &&
                    uiState.value.scanActive &&
                    !uiState.value.isFrontCamera &&
                    photoStereoCapture.isActive

                if (stereoCondition) {
                    if (!inStereoMode) {
                        // First frame entering stereo mode — torch is currently in user's
                        // preferred state. Tag this frame as lit (torch was on), then
                        // toggle off for the next frame.
                        lastStereoLit = true
                        inStereoMode  = true
                    }
                    // Tag current frame with the state it was captured under
                    photoStereoCapture.submitFrame(bitmap, lastStereoLit)

                    // Toggle torch for the NEXT frame and record what that state will be
                    lastStereoLit = !lastStereoLit
                    cameraController.torchController.setTorch(lastStereoLit)

                    // Sync progress to UI (cheap — only updates when count changes)
                    val pairs    = photoStereoCapture.frameCount
                    val complete = photoStereoCapture.isComplete
                    if (pairs != uiState.value.photoStereoFrameCount || complete != uiState.value.photoStereoComplete) {
                        uiState.update { it.copy(
                            photoStereoFrameCount = pairs,
                            photoStereoComplete   = complete
                        ) }
                        // When complete, stop torch pulsing and restore user's torch preference
                        if (complete) {
                            photoStereoCapture.stop()
                            cameraController.torchController.setTorch(torchOnAtScanStart)
                            inStereoMode = false
                        }
                    }
                } else {
                    if (inStereoMode) {
                        // Just left stereo mode (completed or disabled) — mark inactive
                        inStereoMode = false
                    }
                }

                if (frameThrottler.shouldInfer()) {
                    val t0 = System.currentTimeMillis()
                    clahe.process(bitmap)   // updates clahe.lastContrastScore; return value unused
                    val ts = System.currentTimeMillis()
                    handTrackerManager!!.detect(bitmap, ts)

                    // Feed body and face pipelines with the same bitmap if enabled.
                    // All three landmarkers run asynchronously — no extra latency on
                    // the frame dispatch thread (GPU delegate queues work in parallel).
                    if (uiState.value.bodyTrackingEnabled) bodyPipeline.detect(bitmap, ts)
                    if (uiState.value.faceTrackingEnabled) facePipeline.detect(bitmap, ts)

                    // dispatchMs = CLAHE + async-submit cost (does not include GPU inference time,
                    // which arrives via the result callback). Used only for throttle rate tuning.
                    val dispatchMs = (System.currentTimeMillis() - t0).toFloat()
                    frameThrottler.reportInferenceMs(dispatchMs)
                }
            }
        }

        // FullBodyFrame merge — only started when at least one of body/face tracking is enabled.
        // Launched lazily from enableBodyTracking / enableFaceTracking to avoid running this
        // collector (and writing latestFullBodyFrame) on every hand frame when both are off.
        // See enableBodyTracking() / enableFaceTracking() below.

        // G4 — Collect gesture recognitions from primary hand slot and:
        //   1. Update AppUiState.activeGesture for HUD display.
        //   2. Dispatch built-in shortcuts on gesture change (edge-triggered).
        viewModelScope.launch {
            var lastGesture: Gesture? = null
            handPipeline.gestures.collect { gestureList ->
                val primary = gestureList.getOrNull(0)
                // Update state whenever gesture changes (including null)
                if (primary != lastGesture) {
                    uiState.update { it.copy(activeGesture = primary) }
                    // Fire shortcut on newly recognized gesture (not on null)
                    if (primary != null) dispatchGestureShortcut(primary)
                    lastGesture = primary
                }
            }
        }

        cameraController.start()
        // Sync physical torch to the initial UI state (default: true).
        // CameraController.start() opens the camera but never sets the torch —
        // without this call the hardware torch stays off while the UI shows it on.
        cameraController.torchController.setTorch(uiState.value.torchOn)
        uiState.value = uiState.value.copy(showSplash = false)
    }

    fun toggleRenderMode() {
        val next = when (renderer.renderMode) {
            RenderMode.MESH      -> RenderMode.WIREFRAME
            RenderMode.WIREFRAME -> RenderMode.SKELETON
            RenderMode.SKELETON  -> RenderMode.MESH
            RenderMode.ASSET_3D  -> RenderMode.MESH
            RenderMode.ASSET_2D  -> RenderMode.MESH
            RenderMode.LIVE_MESH -> RenderMode.MESH
        }
        renderer.renderMode = next
        uiState.update { it.copy(renderMode = next) }
    }

    /**
     * G1 — Toggle the live-deforming scanned hand mesh on/off.
     *
     * Requires a completed scan ([hasStoredModel] == true). On first activation,
     * [LiveMeshDeformer.buildSkinWeights] is called with the current scanned mesh and
     * rest-pose joint positions, then the tracking loop begins forwarding deformed
     * positions to [ARRenderer.liveMeshPositions] every frame.
     */
    fun toggleLiveMesh() {
        if (!uiState.value.hasStoredModel) return
        val nowActive = !uiState.value.liveMeshActive

        if (nowActive) {
            // Ensure deformer is bound to the current scanned mesh
            val meshPos = depthMeshPositions.value
            val restJoints = restJointPositions
            if (meshPos.isNotEmpty() && restJoints != null && !liveMeshDeformer.hasMesh()) {
                liveMeshDeformer.buildSkinWeights(meshPos, restJoints)
            }
        } else {
            renderer.liveMeshPositions = null
        }

        uiState.update { it.copy(liveMeshActive = nowActive) }
    }

    /**
     * G2 — Toggle neural implicit reconstruction on/off.
     *
     * When enabled, the next completed scan will:
     * 1. Run [ManoShapeFitter] to compute per-segment radius corrections.
     * 2. Train [NeuralImplicitCarver] on the scan's landmark frames (~2 s on Snapdragon 8 Gen 2).
     * 3. Extract the mesh from the learned occupancy field (replaces [DepthCarver]).
     *
     * ARCore Depth API path ([depthMode]) still takes precedence over neural reconstruction
     * when both are enabled.
     *
     * The toggle is safe to call before, during, or after a scan — it only affects the
     * next [processScan] invocation.
     */
    fun toggleNeuralRecon() {
        uiState.update { it.copy(neuralReconEnabled = !it.neuralReconEnabled) }
    }

    /**
     * G5 — Toggle photometric stereo flash capture on/off.
     *
     * When enabled, the scan will alternate torch ON/OFF every frame during
     * CAPTURING, submit frames to [PhotometricStereoCapture], and produce a
     * [com.arhand.scanner.PhotometricNormalMap] stored in [photoStereoNormalMap]
     * once [PhotometricStereoCapture.TARGET_PAIRS] pairs are accumulated.
     *
     * Note: requires rear camera (torch unavailable on front camera). When the
     * front camera is active the toggle is silently ignored.
     */
    fun togglePhotoStereo() {
        if (uiState.value.isFrontCamera) return
        uiState.update { it.copy(photoStereoEnabled = !it.photoStereoEnabled) }
    }

    /** G5 — Most recent photometric normal map. Null until the first stereo-enabled scan. */
    val photoStereoNormalMap: MutableStateFlow<com.arhand.scanner.PhotometricNormalMap?> =
        MutableStateFlow(null)

    fun toggleTorch() {
        val next = !uiState.value.torchOn
        renderer.torchOn = next
        // Also drive the physical torch on the rear camera
        if (::cameraController.isInitialized) {
            cameraController.torchController.setTorch(next)
        }
        uiState.update { it.copy(torchOn = next) }
    }

    fun toggleCloud() {
        uiState.update { it.copy(showCloud = !it.showCloud) }
        renderer.showCloud = uiState.value.showCloud
    }

    fun switchCamera() {
        cameraController.switchCamera()
        val nowFront = cameraController.isFrontFacing()
        // Front camera has no torch — turn it off physically and sync UI state.
        if (nowFront && uiState.value.torchOn) {
            cameraController.torchController.setTorch(false)
            renderer.torchOn = false
            uiState.update { it.copy(isFrontCamera = true, torchOn = false) }
        } else {
            uiState.update { it.copy(isFrontCamera = nowFront) }
        }
    }

    /**
     * B6 — Toggle ARCore Depth API mode. Only meaningful when
     * [AppUiState.depthApiAvailable] is true (gated per [ArDepthAvailability]).
     * Lazily creates/tears down the [ArDepthSession] so it isn't held when unused.
     */
    fun toggleDepth() {
        if (!uiState.value.depthApiAvailable) return
        val next = !uiState.value.depthMode
        if (next) {
            val session = arDepthSession ?: ArDepthSession(getApplication()).also { arDepthSession = it }
            if (session.tryCreate()) {
                // B6 — If the GL surface is already up, setCameraTextureName immediately.
                // If the surface hasn't been created yet, onGlSurfaceCreated will call it.
                val texId = renderer.lastArCameraTextureId
                if (texId != 0) session.setCameraTextureName(texId)
                session.resume()
                uiState.update { it.copy(depthMode = true) }
            }
            // If session creation fails, leave depthMode off — capsule-SDF path remains active.
        } else {
            arDepthSession?.pause()
            uiState.update { it.copy(depthMode = false, depthConfidence = 0f) }
            capturedDepthFrames.clear()
        }
    }

    fun startScan() {
        capturedFrames.clear()
        biometricFrames.clear()
        capturedDepthFrames.clear()
        // G5 — reset stereo capture and start only if enabled
        torchOnAtScanStart = uiState.value.torchOn
        photoStereoCapture.reset()
        if (uiState.value.photoStereoEnabled) photoStereoCapture.start()
        uiState.update {
            it.copy(
                scanActive            = true,
                photoStereoFrameCount = 0,
                photoStereoComplete   = false
            )
        }
        scanner.start()
    }

    fun cancelScan() {
        scanner.reset()
        capturedFrames.clear()
        biometricFrames.clear()
        capturedDepthFrames.clear()
        photoStereoCapture.stop()
        // G5 — restore torch to the state it was in when the scan started
        if (uiState.value.photoStereoEnabled && !uiState.value.isFrontCamera) {
            cameraController.torchController.setTorch(torchOnAtScanStart)
        }
        uiState.update { it.copy(scanActive = false) }
    }
    private fun processScan() {
        // Snapshot all mutable frame buffers on the calling thread before handing
        // off to the background coroutine. A rapid re-scan (user taps RE-SCAN while
        // processing is still running) calls startScan() which clears these lists —
        // snapshots ensure the in-flight coroutine works on a stable copy.
        val cloudPointsSnapshot      = scanner.cloudPoints.toList()
        val capturedFramesSnapshot   = capturedFrames.toList()
        val biometricFramesSnapshot  = biometricFrames.toList()
        val capturedDepthSnapshot    = capturedDepthFrames.toList()
        viewModelScope.launch(Dispatchers.Default) {
            try {
                // B6 — Prefer real metric-depth reconstruction when ARCore Depth API
                // frames were captured this scan; fall back to capsule-SDF otherwise.
                val depthMesh = if (capturedDepthSnapshot.isNotEmpty()) {
                    DepthApiCarver.carveAndExtract(capturedDepthSnapshot)
                } else {
                    FloatArray(0)
                }

                // G2 — Neural implicit reconstruction path.
                // When enabled: (1) run MANO shape fitting for per-segment radius corrections,
                // (2) train NeuralImplicitCarver MLP on the scan's landmark sets,
                // (3) extract mesh from the learned occupancy field.
                // Falls back in order: neural mesh → MANO shape-fitted capsule SDF → DepthCarver.
                var neuralDiagnostics: NeuralReconDiagnostics? = null
                val neuralOrDepthMesh: FloatArray = when {
                    depthMesh.isNotEmpty() -> {
                        // ARCore depth takes precedence over all capsule-SDF variants
                        depthMesh
                    }
                    uiState.value.neuralReconEnabled && capturedFramesSnapshot.isNotEmpty() -> {
                        // G2 step 1: MANO shape fitting
                        val manoFitter = ManoShapeFitter()
                        val shapeParams = manoFitter.fit(capturedFramesSnapshot.map { it.first })

                        // G2 step 2: Train MLP
                        val carver = NeuralImplicitCarver()
                        carver.train(capturedFramesSnapshot, epochs = NeuralImplicitCarver.DEFAULT_EPOCHS)

                        val neuralMesh = if (carver.isTrained) {
                            carver.extractMesh(capturedFramesSnapshot)
                        } else {
                            FloatArray(0)
                        }

                        neuralDiagnostics = NeuralReconDiagnostics(
                            trainLoss       = carver.lastTrainLoss,
                            trainEpochs     = carver.lastTrainEpochs,
                            shapeScale      = shapeParams.segmentRadiusScale.copyOf(),
                            meshVertexCount = neuralMesh.size / 3
                        )

                        if (neuralMesh.isNotEmpty()) {
                            // Neural mesh succeeded
                            neuralMesh
                        } else {
                            // Neural mesh empty (degenerate occupancy field) —
                            // fall back to MANO shape-fitted capsule SDF, which is
                            // still better than plain DepthCarver.
                            manoFitter.carveWithShapeParams(capturedFramesSnapshot, shapeParams)
                                .takeIf { it.isNotEmpty() }
                                ?: DepthCarver.carveAndExtract(capturedFramesSnapshot)
                        }
                    }
                    else -> {
                        // Default path: capsule-SDF DepthCarver (plain or depth API)
                        DepthCarver.carveAndExtract(capturedFramesSnapshot)
                    }
                }

                val rawMeshPos = neuralOrDepthMesh
                // B5: Laplacian smoothing — removes marching cubes staircase artifacts.
                // λ=0.5, 3 iterations — sufficient for visual improvement, minimal distortion.
                val meshPos = MeshSmoother.smooth(rawMeshPos, iterations = 3, lambda = 0.5f)
                renderer.depthMeshPositions = meshPos
                depthMeshPositions.value    = meshPos   // expose for 3D viewer

                // G1 — Capture rest-pose joint positions from the latest retarget result
                // so LiveMeshDeformer can build skinning weights for this mesh.
                val latestResult = latestRetargetResult.value
                if (latestResult != null) {
                    val aspect   = currentAspect.value
                    val mirror   = uiState.value.isFrontCamera
                    // Build flat xyz array: one Vec3 per joint, ordered by JOINT_* index
                    val rjp = FloatArray(com.arhand.mocap.BoneRetargeter.JOINT_COUNT * 3)
                    // Wrist
                    val wristPos = latestResult.wristTransform.position
                    rjp[0] = wristPos.x; rjp[1] = wristPos.y; rjp[2] = wristPos.z
                    // For the remaining joints use their retargeted position from bone segments.
                    // Approximate as: wristPos + (bindDir × boneLength). Since the bind pose
                    // maps to the same world space, we reuse the wrist pos for all non-wrist
                    // joints as a centroid approximation. The per-vertex inv-distance weighting
                    // in LiveMeshDeformer handles the spatial assignment correctly from the
                    // joint centres computed here.
                    for (ji in 1 until com.arhand.mocap.BoneRetargeter.JOINT_COUNT) {
                        rjp[ji * 3]     = wristPos.x
                        rjp[ji * 3 + 1] = wristPos.y
                        rjp[ji * 3 + 2] = wristPos.z
                    }
                    // Override each joint with actual world position from the scanned landmarks
                    // (the biometricFrames have the highest-quality frame landmarks).
                    val bestFrame = biometricFramesSnapshot.maxByOrNull { it.second }
                    if (bestFrame != null) {
                        val lms = bestFrame.first
                        if (lms.size >= 21) {
                            // Bone segment base/tip → joint centre approximated as midpoint
                            for ((jointIdx, baseIdx, tipIdx) in com.arhand.mocap.BoneRetargeter.BONE_SEGMENTS) {
                                val baseLm = lms[baseIdx]; val tipLm = lms[tipIdx]
                                val (bx, by, bz) = com.arhand.util.landmarkToWorld(baseLm, aspect, mirror)
                                val (tx, ty, tz) = com.arhand.util.landmarkToWorld(tipLm, aspect, mirror)
                                rjp[jointIdx * 3]     = (bx + tx) * 0.5f
                                rjp[jointIdx * 3 + 1] = (by + ty) * 0.5f
                                rjp[jointIdx * 3 + 2] = (bz + tz) * 0.5f
                            }
                        }
                    }
                    restJointPositions = rjp
                    // Rebuild deformer for the new mesh (invalidates previous skin weights)
                    liveMeshDeformer.buildSkinWeights(meshPos, rjp)
                }

                val glbFile = GLBExporter.export(
                    getApplication(), meshPos, cloudPointsSnapshot
                )

                val scoresJson = scanner.status.value.poseScores
                    .joinToString(",", prefix="[", postfix="]") { "%.3f".format(it) }

                // Compute hand biometrics from the accumulated landmark frames
                val biometrics = HandBiometrics.compute(
                    frames = biometricFramesSnapshot,
                    aspect = currentAspect.value
                )

                modelStore.saveModel(glbFile, cloudPointsSnapshot.size, scoresJson, biometrics)
                // D4 — Append to biometric history (no-op when biometrics is null)
                if (biometrics != null) modelStore.saveToHistory(biometrics)
                val updatedHistory = modelStore.loadHistory()

                // G5 — Compute photometric normal map if pairs were accumulated.
                // This is a pure CPU operation; done here on the Default dispatcher.
                if (uiState.value.photoStereoEnabled && photoStereoCapture.frameCount > 0) {
                    val normalMap = photoStereoCapture.computeNormalMap()
                    if (!normalMap.isEmpty) {
                        photoStereoNormalMap.value = normalMap
                    }
                    // Stop the capture (it may have already stopped when complete, but
                    // guard against early scan termination before TARGET_PAIRS were reached).
                    photoStereoCapture.stop()
                    // Restore torch to the state it was in when the scan started
                    cameraController.torchController.setTorch(torchOnAtScanStart)
                }

                scanner.markDone()
                uiState.update {
                    it.copy(
                        scanActive             = false,
                        exportedGlbPath        = glbFile.absolutePath,
                        hasStoredModel         = true,
                        handBiometrics         = biometrics,
                        biometricHistory       = updatedHistory,
                        jointRomData           = scanner.romData,
                        neuralReconDiagnostics = neuralDiagnostics
                    )
                }
            } catch (e: Exception) {
                scanner.markFailed(e.message ?: "Processing error")
                photoStereoCapture.stop()
                // Restore torch if stereo flash had taken control of it
                if (uiState.value.photoStereoEnabled && !uiState.value.isFrontCamera) {
                    cameraController.torchController.setTorch(torchOnAtScanStart)
                }
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

    // ─── D2: Calibration card scale ───────────────────────────────────────────

    /**
     * Persist a calibrated world-to-mm scale factor derived from a credit-card measurement.
     * Pass null to revert to the built-in [HandBiometrics.WORLD_TO_MM] default.
     *
     * Usage: after the user completes the calibration card flow, the UI layer passes the
     * measured card width in world units to [HandBiometrics.calibrateFromCard], then calls
     * this function with the result:
     * ```kotlin
     * val scale = HandBiometrics.calibrateFromCard(measuredWidthWorld)
     * if (scale != null) viewModel.saveCalibration(scale)
     * ```
     */
    fun saveCalibration(scaleWorldToMm: Float?) {
        viewModelScope.launch(Dispatchers.IO) {
            modelStore.saveCalibration(scaleWorldToMm)
        }
    }

    /**
     * Load the currently stored calibration scale from DataStore.
     * Returns [HandBiometrics.WORLD_TO_MM] (800f) if no calibration has been performed.
     * The result is typically used to convert biometric world-unit values to mm in the UI.
     */
    suspend fun loadCalibratedScale(): Float = modelStore.loadCalibratedScale()

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

    // ─── Full body + face extension ───────────────────────────────────────────

    /** Guards against launching the FullBodyFrame merge collector more than once. */
    private var fullBodyCollectorStarted = false

    private fun ensureFullBodyCollector() {
        if (fullBodyCollectorStarted) return
        fullBodyCollectorStarted = true
        viewModelScope.launch {
            handPipeline.processed.collect { hands ->
                if (uiState.value.bodyTrackingEnabled || uiState.value.faceTrackingEnabled) {
                    val leftSlot  = hands.firstOrNull { it.slotIndex == 0 }?.landmarks
                    val rightSlot = hands.firstOrNull { it.slotIndex == 1 }?.landmarks
                    latestFullBodyFrame.value = FullBodyFrame(
                        pose      = if (uiState.value.bodyTrackingEnabled) bodyPipeline.processed.value else null,
                        face      = if (uiState.value.faceTrackingEnabled) facePipeline.processed.value else null,
                        leftHand  = leftSlot,
                        rightHand = rightSlot
                    )
                }
            }
        }
    }

    /**
     *
     * On first enable, initialises [BodyPipeline] using [pose_landmarker_lite.task]
     * from assets. Subsequent toggles reuse the same landmarker instance.
     * Body landmarks are merged into [latestFullBodyFrame] every hand-pipeline frame.
     */
    fun enableBodyTracking(enabled: Boolean) {
        if (enabled && !uiState.value.bodyTrackingEnabled) {
            bodyPipeline.init(getApplication())
            ensureFullBodyCollector()
        } else if (!enabled) {
            bodyPipeline.close()
        }
        uiState.update { it.copy(bodyTrackingEnabled = enabled) }
    }

    /**
     * Enable or disable face landmark tracking.
     *
     * On first enable, initialises [FacePipeline] using [face_landmarker.task] from assets.
     * Expression values are available via [FacePipeline.expressions] as a [StateFlow].
     */
    fun enableFaceTracking(enabled: Boolean) {
        if (enabled && !uiState.value.faceTrackingEnabled) {
            facePipeline.init(getApplication())
            ensureFullBodyCollector()
        } else if (!enabled) {
            facePipeline.close()
        }
        uiState.update { it.copy(faceTrackingEnabled = enabled) }
    }

    // ─── OSC streaming ────────────────────────────────────────────────────────

    /**
     * Start broadcasting live joint rotations via OSC UDP.
     *
     * @param host   Target IP or hostname. Defaults to LAN broadcast.
     * @param port   Target UDP port. Defaults to the schema-recommended port.
     * @param schema OSC address schema. Defaults to the current [AppUiState.oscSchema].
     */
    fun startOscStreaming(
        host:   String    = OscStreamer.DEFAULT_HOST,
        port:   Int       = OscStreamer.defaultPortFor(uiState.value.oscSchema),
        schema: OscSchema = uiState.value.oscSchema
    ) {
        oscStreamer.start(host, schema, port)
        uiState.update { it.copy(isOscStreaming = true, oscHost = host, oscPort = port, oscSchema = schema) }
    }

    /** Stop OSC broadcasting and close the UDP socket. */
    fun stopOscStreaming() {
        oscStreamer.stop()
        uiState.update { it.copy(isOscStreaming = false) }
    }

    /**
     * Switch the active OSC schema preset.
     *
     * If streaming is active, the new schema takes effect on the next sendFrame call —
     * no restart required. Port is updated to the schema default only when not streaming.
     */
    fun setOscSchema(schema: OscSchema) {
        oscStreamer.schema = schema
        val newPort = if (oscStreamer.isStreaming) uiState.value.oscPort
                      else OscStreamer.defaultPortFor(schema)
        uiState.update { it.copy(oscSchema = schema, oscPort = newPort) }
    }

    // ─── G4: Gesture shortcut dispatcher ──────────────────────────────────────

    /**
     * Built-in gesture → app action mapping.
     *
     * Edge-triggered — called once when the gesture transitions from null/other
     * to this gesture. Never called repeatedly while the gesture is held.
     *
     * Default mapping (mirrors G4 spec):
     *   FIST      → start scan  (if no scan active and no model stored yet)
     *   PEACE     → switch camera
     *   ROCK      → toggle torch
     *   CALL      → toggle OSC stream (send mode)
     *
     * The remaining gestures (OPEN_PALM, THUMBS_UP, POINT_UP, OK) are reserved
     * for user-configurable macro triggers in a future UX pass.
     */
    private fun dispatchGestureShortcut(gesture: Gesture) {
        when (gesture) {
            Gesture.FIST -> {
                // Only trigger scan start if no scan is running and no model stored yet
                if (!uiState.value.scanActive && !uiState.value.hasStoredModel) startScan()
            }
            Gesture.PEACE -> switchCamera()
            Gesture.ROCK  -> toggleTorch()
            Gesture.CALL  -> {
                if (uiState.value.oscMode == OscMode.SEND) {
                    if (uiState.value.isOscStreaming) stopOscStreaming()
                    else startOscStreaming()
                }
            }
            // Reserved: OPEN_PALM, THUMBS_UP, POINT_UP, OK — no-op for now
            else -> {}
        }
    }

    // ─── G3: OSC mode switching & receiver control ─────────────────────────────

    /**
     * Switch between SEND and RECEIVE modes.
     *
     * - SEND → RECEIVE: stops the streamer (if active), starts the receiver.
     * - RECEIVE → SEND: stops the receiver, streamer can be started independently.
     *
     * The camera pipeline keeps running in both modes; in RECEIVE mode the renderer
     * will prefer the remote RetargetResult over the locally-computed one.
     */
    fun setOscMode(mode: OscMode) {
        when (mode) {
            OscMode.SEND -> {
                stopOscReceiving()
                uiState.update { it.copy(oscMode = OscMode.SEND) }
            }
            OscMode.RECEIVE -> {
                stopOscStreaming()
                startOscReceiving()
            }
        }
    }

    /** G3 — Job for the OSC receiver collector. Cancelled and replaced on each startOscReceiving() call. */
    private var oscReceiverJob: kotlinx.coroutines.Job? = null

    /**
     * Start the OSC receiver on [port] and wire its output to the renderer.
     *
     * @param port UDP port to listen on. Defaults to [OscReceiver.DEFAULT_PORT] (9000).
     */
    fun startOscReceiving(port: Int = OscReceiver.DEFAULT_PORT) {
        if (oscReceiver.isReceiving) oscReceiver.stop()

        oscReceiver.start(port)
        if (!oscReceiver.isReceiving) return  // port bind failed

        uiState.update { it.copy(
            oscMode       = OscMode.RECEIVE,
            isOscReceiving = true,
            oscReceivePort = port
        )}

        // Cancel any previous collector before launching a new one so toggling
        // RECEIVE mode does not accumulate stale coroutines writing to the renderer.
        oscReceiverJob?.cancel()
        oscReceiverJob = viewModelScope.launch {
            oscReceiver.frames.collect { result ->
                renderer.latestRetargetResult = result
            }
        }
    }

    /** Stop the OSC receiver and revert to local hand tracking display. */
    fun stopOscReceiving() {
        oscReceiver.stop()
        oscReceiverJob?.cancel()
        oscReceiverJob = null
        uiState.update { it.copy(isOscReceiving = false) }
    }

    override fun onCleared() {
        handTrackerManager?.close()
        if (::cameraController.isInitialized) cameraController.stop()
        oscStreamer.stop()
        oscReceiver.stop()
        oscReceiverJob?.cancel()
        bodyPipeline.close()
        facePipeline.close()
        arDepthSession?.close()
        super.onCleared()
    }
}
