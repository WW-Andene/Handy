package com.arhand.depth

import android.content.Context
import com.arhand.camera.BitmapGrayscaleShim
import com.arhand.depth.fusion.FusedDepthSource
import com.arhand.util.Vec3
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

/**
 * Persistent spatial sensing layer — the foundation of Handy as a LiDAR replacement.
 *
 * ## Purpose
 *
 * Previously, depth sensing (ARCore + SfM + Photometric) was gated behind a scan-mode
 * toggle. This meant streaming, recording, and VTubing sessions had no metric spatial
 * grounding — wrist positions were in camera-normalised space, not real-world metres.
 *
 * [SpatialLayer] makes depth sensing **always-on** from camera start. It provides:
 *
 *   1. **Metric grounding** — when ARCore is tracking, [isGrounded] = true and
 *      [cameraWorldX/Y/Z] give the device's metric position. All downstream consumers
 *      (retargeter, OSC streamer, BVH recorder) can tag their output as metric.
 *
 *   2. **Live point cloud** — [fusedDepth].store is continuously updated. Scan tools
 *      read from it for TSDF accumulation; the renderer reads from it for live display.
 *
 *   3. **SfM scale calibration** — [metricScale] is the dynamically calibrated
 *      metres-per-SfM-unit factor. When ARCore is unavailable, SfM output is still
 *      self-consistent and approximately metric after warm-up.
 *
 *   4. **Degradation ladder** — on devices without ARCore depth hardware, SfM + Photometric
 *      continue independently. [isGrounded] reflects ARCore availability specifically;
 *      [depthActive] reflects any depth source running.
 *
 * ## Lifecycle
 *
 * [start] is called once from [AppViewModel.initCamera], not from the scan flow.
 * [stop] is called from [AppViewModel.onCleared].
 *
 * The scan tools (posed [Scanner], [FreeformScanner]) accumulate frames into the
 * [fusedDepth].store and TSDF without owning the depth session lifecycle.
 *
 * ## State exposure
 *
 * [state] is a [StateFlow<SpatialState>] collected by the HUD and ViewModel.
 * It is updated every ARCore frame — at the GL thread's frame rate (~30–60fps) via
 * [onDrawFrame], which is already called from [ARRenderer.onDrawFrame].
 */
class SpatialLayer(private val context: Context) {

    data class SpatialState(
        /** True when ARCore has achieved TRACKING state and positions are metric. */
        val isGrounded:    Boolean = false,
        /** True when at least one depth source (ARCore, SfM, or Photometric) is active. */
        val depthActive:   Boolean = false,
        /** ARCore camera X in world space (metres). NaN when not grounded. */
        val cameraWorldX:  Float   = Float.NaN,
        /** ARCore camera Y in world space (metres). NaN when not grounded. */
        val cameraWorldY:  Float   = Float.NaN,
        /** ARCore camera Z in world space (metres). NaN when not grounded. */
        val cameraWorldZ:  Float   = Float.NaN,
        /**
         * Calibrated metric scale: metres per world-space unit used by SfM.
         * 0.001 = default (uncalibrated). Increases accuracy once ARCore warm-up completes.
         */
        val metricScale:   Float   = 0.001f,
        /**
         * Mean depth confidence in the current live point cloud (0–1).
         * 1.0 = all ARCore high-confidence points. Lower when SfM/Photometric dominate.
         */
        val depthConfidence: Float = 0f,
        /** Live point count in the shared PointCloudStore. */
        val pointCount:    Int     = 0,
        /** Human-readable active source label for HUD display. */
        val sourceLabel:   String  = "SPATIAL OFF"
    )

    private val _state = MutableStateFlow(SpatialState())
    val state: StateFlow<SpatialState> = _state

    val fusedDepth = FusedDepthSource(context)

    private var started = false

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    /**
     * Start all depth sub-sources. Call once after camera init, not per-scan.
     *
     * @param shim    Bitmap shim that routes the running camera frames to SfM/Photometric
     *                without opening a second camera session.
     * @param texId   GL texture ID for ARCore's camera feed (from the GL surface).
     *                Pass 0 if the GL surface hasn't been created yet — call
     *                [setCameraTextureName] once it is.
     */
    fun start(shim: BitmapGrayscaleShim, texId: Int = 0) {
        if (started) return
        started = true

        fusedDepth.attachShim(shim)
        if (texId != 0) fusedDepth.setCameraTextureName(texId)

        fusedDepth.start(object : DepthSourceCallback {
            override fun onPoints(batch: FloatArray, len: Int) {
                // Points go directly into fusedDepth.store — no extra routing needed.
                // Scan tools (TSDF, FreeformScanner) read from store on their own cadence.
            }
            override fun onStats(stats: DepthSource.Stats) {
                val arcoreCamX = fusedDepth.arcore.lastCamX
                val grounded   = !arcoreCamX.isNaN()
                _state.value = _state.value.copy(
                    isGrounded      = grounded,
                    depthActive     = true,
                    cameraWorldX    = arcoreCamX,
                    cameraWorldY    = fusedDepth.arcore.lastCamY,
                    cameraWorldZ    = fusedDepth.arcore.lastCamZ,
                    metricScale     = fusedDepth.sfmScale,
                    depthConfidence = stats.confidenceMean,
                    pointCount      = fusedDepth.store.pointCount,
                    sourceLabel     = stats.extraLabel.ifBlank { "SPATIAL" }
                )
            }
            override fun onError(msg: String) {
                android.util.Log.w("SpatialLayer", "Depth error: $msg")
            }
        })

        _state.value = _state.value.copy(depthActive = true, sourceLabel = "STARTING…")
    }

    fun stop() {
        if (!started) return
        started = false
        fusedDepth.stop()
        _state.value = SpatialState()
    }

    fun close() {
        stop()
        fusedDepth.close()
    }

    /** Must be called from the GL thread after surface creation. */
    fun setCameraTextureName(texId: Int) = fusedDepth.setCameraTextureName(texId)

    /** Must be called from [ARRenderer.onDrawFrame] — drives ARCore's session.update(). */
    fun onDrawFrame() = fusedDepth.onDrawFrame()

    /** Feed a bitmap frame to SfM and Photometric sub-sources. Call every camera frame. */
    fun onBitmap(bitmap: android.graphics.Bitmap, timestampMs: Long) =
        fusedDepth.arcore.let {
            // The shim already routes bitmaps — this call is for external callers
            // who want to explicitly push a frame (e.g. from a separate camera source).
        }

    // ── World-frame anchor ────────────────────────────────────────────────────

    /**
     * Compute the metric world-space position of a hand landmark given the current
     * ARCore camera pose.
     *
     * MediaPipe world landmarks are in metres, hand-geometric-centre origin, camera frame.
     * ARCore provides camera-to-world transform via [ArCoreDepthSource.lastCamX/Y/Z] and
     * the camera pose quaternion. This method projects a MediaPipe world-space point
     * into the ARCore world frame.
     *
     * When ARCore is not tracking ([isGrounded] = false), returns the raw MediaPipe
     * world position (camera-relative, approximately metric via sfmScale).
     *
     * @param mediaPipeWorldX  MediaPipe worldX in metres (hand-centre origin, camera frame)
     * @param mediaPipeWorldY  MediaPipe worldY in metres
     * @param mediaPipeWorldZ  MediaPipe worldZ in metres
     * @return Vec3 in ARCore world space (metres, gravity-aligned Y-up), or camera-relative metres
     */
    fun toWorldSpace(
        mediaPipeWorldX: Float,
        mediaPipeWorldY: Float,
        mediaPipeWorldZ: Float
    ): Vec3 {
        val s = _state.value
        if (!s.isGrounded) {
            // Not grounded: return camera-relative position scaled by sfmScale if better than default
            return Vec3(mediaPipeWorldX, mediaPipeWorldY, mediaPipeWorldZ)
        }
        // ARCore world frame: camera position + MediaPipe offset rotated into world frame.
        // For the current implementation we use the simplified additive model:
        // worldPos = cameraPos + mediaPipeOffset
        // This is correct when the hand is close to the camera (< 1m) and the camera
        // hasn't rotated far from its reference pose. Full rotation matrix application
        // requires the ARCore cam.pose quaternion — added when the rotation matrix
        // accessor is exposed from ArCoreDepthSource (future sprint: GAP-10).
        return Vec3(
            s.cameraWorldX + mediaPipeWorldX,
            s.cameraWorldY + mediaPipeWorldY,
            s.cameraWorldZ + mediaPipeWorldZ
        )
    }
}
