package com.arhand.ui

import android.content.Intent
import android.net.Uri
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
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.FileProvider
import androidx.lifecycle.compose.collectAsStateWithLifecycle
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

        setContent {
            HandyApp(vm = vm, onShareGlb = ::shareGlb)
        }

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
    val uiState     by vm.uiState.collectAsStateWithLifecycle()
    val perfState   by vm.perfMonitor.state.collectAsStateWithLifecycle()
    val scanStatus  by vm.scanner.status.collectAsStateWithLifecycle()

    var showResult by remember { mutableStateOf(false) }

    // Show result modal when scan completes
    LaunchedEffect(scanStatus.state) {
        if (scanStatus.state == Scanner.ScanState.DONE) showResult = true
    }

    Box(modifier = Modifier.fillMaxSize()) {
        // GL Surface — camera preview + hand mesh rendering
        if (!uiState.showSplash) {
            AndroidView(
                modifier = Modifier.fillMaxSize(),
                factory = { ctx ->
                    GLSurfaceView(ctx).apply {
                        setEGLContextClientVersion(3)
                        setRenderer(vm.renderer)
                        renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
                    }
                }
            )
        }

        // Splash
        if (uiState.showSplash) SplashScreen()

        // HUD
        if (!uiState.showSplash) {
            HudOverlay(
                perf = perfState,
                renderMode = uiState.renderMode.name,
                torchOn = uiState.torchOn
            )
        }

        // Scan overlay
        if (uiState.scanActive) {
            ScanOverlay(status = scanStatus, onCancel = { vm.cancelScan() })
        }

        // Control panel + depth strip at bottom
        if (!uiState.showSplash && !uiState.scanActive) {
            Column(
                modifier = Modifier.align(Alignment.BottomCenter)
            ) {
                ControlPanel(
                    renderMode   = uiState.renderMode.name,
                    torchOn      = uiState.torchOn,
                    showCloud    = uiState.showCloud,
                    scanActive   = uiState.scanActive,
                    hasModel     = uiState.hasStoredModel,
                    onMode       = vm::toggleRenderMode,
                    onTorch      = vm::toggleTorch,
                    onCloud      = vm::toggleCloud,
                    onCamera     = vm::switchCamera,
                    onScan       = {
                        if (uiState.hasStoredModel) showResult = true
                        else vm.startScan()
                    },
                    onViewModel  = { showResult = true }
                )
            }
        }

        // Result modal
        if (showResult) {
            ScanResultModal(
                pointCount  = scanStatus.totalPoints,
                glbPath     = uiState.exportedGlbPath,
                poseScores  = scanStatus.poseScores,
                onShare     = { onShareGlb(); showResult = false },
                onRescan    = { showResult = false; vm.startScan() },
                onDismiss   = { showResult = false }
            )
        }
    }
}
