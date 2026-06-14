package com.arhand.feature.record

import android.app.Application
import com.arhand.export.GltfAnimationExporter
import com.arhand.mocap.LoadedAsset
import com.arhand.mocap.MotionRecorder
import com.arhand.util.Vec3
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/**
 * Owns the motion recording lifecycle and take list.
 *
 * Single responsibility: start/stop recording, manage take labels,
 * and accumulate the per-session take list.
 *
 * Export (BVH/GLB) is triggered here but executed via [MotionRecorder]
 * on the IO dispatcher. [AppViewModel] passes the scan-derived joint
 * offsets and pose meshes at export time.
 */
class RecordingManager(
    private val app:            Application,
    private val motionRecorder: MotionRecorder,
    private val scope:          CoroutineScope
) {
    private val _state = MutableStateFlow(RecordingState())
    val state: StateFlow<RecordingState> = _state.asStateFlow()

    fun start() {
        motionRecorder.start()
        val autoLabel = _state.value.currentLabel.ifBlank {
            "Take ${_state.value.takes.size + 1}"
        }
        _state.value = _state.value.copy(isRecording = true, currentLabel = autoLabel)
    }

    fun stop(): Long {
        val startMs = _state.value.recordingStartMs
        motionRecorder.stop()
        _state.value = _state.value.copy(isRecording = false)
        return startMs
    }

    /**
     * Mark recording as stopped without calling [MotionRecorder.stop] — used by
     * callers that need to call stop and frameSnapshot themselves in sequence.
     */
    fun markStopped() {
        _state.value = _state.value.copy(isRecording = false)
    }

    /**
     * Stop recording, export BVH + animated GLB, and append a [TakeEntry] to the take list.
     *
     * Runs on the calling coroutine — caller is responsible for dispatching to IO.
     *
     * @param outputDir       Directory to write export files into.
     * @param scannedOffsets  Per-joint metric offsets from scan, for BVH hierarchy (HAND-2).
     * @param loadedAsset     Currently loaded GLB asset, for glTF animation channel mapping.
     */
    suspend fun exportAndAppend(
        outputDir:      java.io.File,
        scannedOffsets: Map<Int, com.arhand.util.Vec3>?,
        loadedAsset:    com.arhand.mocap.LoadedAsset
    ) {
        val recordingStart = _state.value.recordingStartMs
        motionRecorder.stop()
        _state.value = _state.value.copy(isRecording = false)

        outputDir.mkdirs()

        val snapshot = motionRecorder.frameSnapshot()
        val rawLabel = _state.value.currentLabel.ifBlank {
            "take_${System.currentTimeMillis() / 1000}"
        }
        val label = rawLabel.replace(' ', '_').replace(Regex("[^A-Za-z0-9_\\-]"), "")

        val bvhFile = motionRecorder.exportBvh(outputDir, label, scannedOffsets)

        val gltfFile = com.arhand.export.GltfAnimationExporter.export(
            frames    = snapshot,
            asset     = loadedAsset,
            outputDir = outputDir,
            label     = "${label}_anim"
        )

        val durationMs = System.currentTimeMillis() - recordingStart
        appendTake(TakeEntry(
            label      = rawLabel,
            bvhPath    = bvhFile?.absolutePath,
            gltfPath   = gltfFile?.absolutePath,
            durationMs = durationMs,
            frameCount = snapshot.size
        ))
    }

    fun setLabel(label: String) {
        if (!_state.value.isRecording) {
            _state.value = _state.value.copy(currentLabel = label)
        }
    }

    fun appendTake(take: TakeEntry) {
        _state.value = _state.value.copy(
            takes        = _state.value.takes + take,
            currentLabel = ""
        )
    }

    fun markRecordingStarted() {
        _state.value = _state.value.copy(recordingStartMs = System.currentTimeMillis())
    }
}

/**
 * Immutable recording state snapshot.
 */
data class RecordingState(
    val isRecording:      Boolean    = false,
    val currentLabel:     String     = "",
    val takes:            List<TakeEntry> = emptyList(),
    val exportedBvhPath:  String?    = null,
    val exportedGltfPath: String?    = null,
    val recordingStartMs: Long       = 0L
)
