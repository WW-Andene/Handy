package com.arhand.ui

import android.content.Intent
import android.opengl.GLSurfaceView
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.foundation.layout.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.FileProvider
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.arhand.mocap.OscSchema
import com.arhand.mocap.OscMode
import com.arhand.render.RenderMode
import com.arhand.scanner.Scanner
import java.io.File

class MainActivity : ComponentActivity() {

    private val vm: AppViewModel by viewModels()

    private val cameraPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) vm.initCamera()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent { HandyApp(vm = vm, onShareGlb = ::shareGlb) }
        cameraPermission.launch(android.Manifest.permission.CAMERA)
    }

    private fun shareGlb() {
        val path = vm.uiState.value.exportedGlbPath ?: return
        val file = File(path)
        if (!file.exists()) return
        val uri = FileProvider.getUriForFile(this, "$packageName.fileprovider", file)
        val intent = Intent(Intent.ACTION_SEND).apply {
            type = "model/gltf-binary"
            putExtra(Intent.EXTRA_STREAM, uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        startActivity(Intent.createChooser(intent, "Share Hand Model"))
    }
}

@Composable
fun HandyApp(vm: AppViewModel, onShareGlb: () -> Unit) {
    val uiState            by vm.uiState.collectAsStateWithLifecycle()
    val perfState          by vm.perfMonitor.state.collectAsStateWithLifecycle()
    val scanStatus         by vm.scanner.status.collectAsStateWithLifecycle()
    val depthMeshPositions by vm.depthMeshPositions.collectAsStateWithLifecycle()
    val processedHands     by vm.handPipeline.processed.collectAsStateWithLifecycle()

    var showResult by remember { mutableStateOf(false) }

    LaunchedEffect(scanStatus.state) {
        if (scanStatus.state == Scanner.ScanState.DONE) showResult = true
    }

    Box(modifier = Modifier.fillMaxSize()) {
        // GL Surface — RGBA8888 EGL config required so glClearColor alpha=0 is
        // transparent rather than black. Without setEGLConfigChooser the default
        // config on many devices is RGB (no alpha) and the clear produces opaque black.
        if (!uiState.showSplash) {
            AndroidView(
                modifier = Modifier.fillMaxSize(),
                factory = { ctx ->
                    GLSurfaceView(ctx).apply {
                        setEGLContextClientVersion(3)
                        // Request RGBA8888 so the alpha channel of the clear colour works
                        setEGLConfigChooser(8, 8, 8, 8, 16, 0)
                        setRenderer(vm.renderer)
                        renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
                    }
                }
            )
        }

        if (uiState.showSplash) SplashScreen()

        // E6 — Onboarding overlay: shown after splash on first launch
        if (!uiState.showSplash && uiState.showOnboarding) {
            OnboardingScreen(onDismiss = { vm.dismissOnboarding() })
        }

        if (!uiState.showSplash && !uiState.showOnboarding) {
            HudOverlay(
                perf           = perfState,
                renderMode     = uiState.renderMode.name,
                torchOn        = uiState.torchOn,
                landmarkCount  = processedHands.sumOf { it.landmarks.size },
                activeSlots    = processedHands.map { it.slotIndex }.sorted()
                                     .joinToString(",").ifEmpty { "--" },
                activeGesture  = uiState.activeGesture
            )
        }

        if (uiState.scanActive) {
            ScanOverlay(status = scanStatus, onCancel = { vm.cancelScan() })
        }

        if (!uiState.showSplash && !uiState.showOnboarding && !uiState.scanActive) {
            Column(modifier = Modifier.align(Alignment.BottomCenter)) {
                // G2 — DepthScannerPanel is now always shown (not gated by depthApiAvailable)
                // because the neural recon toggle is independent of ARCore Depth API.
                DepthScannerPanel(
                    depthMode             = uiState.depthMode,
                    depthConfidence       = uiState.depthConfidence,
                    onToggleDepth         = if (uiState.depthApiAvailable) vm::toggleDepth else ({}),
                    depthApiAvailable     = uiState.depthApiAvailable,
                    neuralReconEnabled    = uiState.neuralReconEnabled,
                    neuralDiagnostics     = uiState.neuralReconDiagnostics,
                    onToggleNeuralRecon   = vm::toggleNeuralRecon,
                    photoStereoEnabled    = uiState.photoStereoEnabled,
                    photoStereoFrameCount = uiState.photoStereoFrameCount,
                    photoStereoComplete   = uiState.photoStereoComplete,
                    isFrontCamera         = uiState.isFrontCamera,
                    onTogglePhotoStereo   = vm::togglePhotoStereo
                )
                OscPanel(
                    isStreaming    = uiState.isOscStreaming,
                    isReceiving    = uiState.isOscReceiving,
                    activeSchema   = uiState.oscSchema,
                    oscMode        = uiState.oscMode,
                    receivePort    = uiState.oscReceivePort,
                    onModeSelect   = vm::setOscMode,
                    onSchemaSelect = vm::setOscSchema,
                    onToggle       = {
                        when (uiState.oscMode) {
                            com.arhand.mocap.OscMode.SEND ->
                                if (uiState.isOscStreaming) vm.stopOscStreaming()
                                else vm.startOscStreaming()
                            com.arhand.mocap.OscMode.RECEIVE ->
                                if (uiState.isOscReceiving) vm.stopOscReceiving()
                                else vm.startOscReceiving()
                        }
                    }
                )
                ControlPanel(
                    renderMode     = uiState.renderMode.name,
                    torchOn        = uiState.torchOn,
                    showCloud      = uiState.showCloud,
                    scanActive     = uiState.scanActive,
                    hasModel       = uiState.hasStoredModel,
                    isRecording    = uiState.isRecording,
                    liveMeshActive = uiState.liveMeshActive,
                    onMode         = vm::toggleRenderMode,
                    onTorch        = vm::toggleTorch,
                    onCloud        = vm::toggleCloud,
                    onCamera       = vm::switchCamera,
                    onScan         = {
                        if (uiState.hasStoredModel) showResult = true
                        else vm.startScan()
                    },
                    onViewModel    = { showResult = true },
                    onRecord       = {
                        if (uiState.isRecording) vm.stopRecordingAndExport()
                        else vm.startRecording()
                    },
                    onLiveMesh     = vm::toggleLiveMesh
                )
            }
        }

        if (showResult) {
            ScanResultModal(
                pointCount    = scanStatus.totalPoints,
                glbPath       = uiState.exportedGlbPath,
                poseScores    = scanStatus.poseScores,
                meshPositions = depthMeshPositions,
                biometrics         = uiState.handBiometrics,
                biometricHistory   = uiState.biometricHistory,
                jointRomData       = uiState.jointRomData,
                onShare       = { onShareGlb(); showResult = false },
                onRescan      = { showResult = false; vm.startScan() },
                onDismiss     = { showResult = false }
            )
        }
    }
}
