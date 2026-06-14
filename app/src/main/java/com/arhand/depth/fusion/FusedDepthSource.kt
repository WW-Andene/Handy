package com.arhand.depth.fusion

import android.content.Context
import com.arhand.depth.ArCoreDepthSource
import com.arhand.depth.DepthSource
import com.arhand.depth.DepthSourceCallback
import com.arhand.depth.PhotometricDepthSource
import com.arhand.depth.SfMDepthSource
import com.arhand.util.PointCloudStore
import com.arhand.camera.BitmapGrayscaleShim
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * FusedDepthSource — software LiDAR.
 *
 * Combines three complementary depth signals into a single coherent [PointCloudStore]:
 *
 *   ARCore  → metric scale + world-space anchor. Best quality when tracking. Confidence 1.0.
 *   SfM     → optical flow triangulation. Works without ARCore. Confidence 0.6.
 *             When ARCore is available, SfM depth is rescaled to metric using the
 *             dynamically calibrated [sfmScale] instead of the hardcoded PX_TO_M guess.
 *   Photometric → surface normals from torch on/off pairs. Adds surface topology detail
 *             that neither ARCore nor SfM can resolve. Confidence 0.35.
 *             When ARCore scale is known, photometric depth is rescaled to match.
 *
 * ## Scale calibration (ARCore ↔ SfM)
 *
 * SfM's `PX_TO_M = 0.001f` is a guess. Real scale depends on the device's sensor size,
 * optics, and the scene depth. ARCore gives us a ground truth: each frame we have both
 * an ARCore metric camera displacement (metres) and a SfM pixel displacement (pixels).
 *
 *   sfmScale = arcore_displacement_m / sfm_displacement_px
 *
 * This is computed as an EMA over recent frames to smooth out single-frame noise.
 * Once [sfmScale] is warm (> [SCALE_MIN_SAMPLES] frames), SfM depth values are
 * multiplied by `sfmScale / PX_TO_M` to convert from the SfM coordinate frame into metres.
 *
 * ## Continuity fallback
 *
 * When ARCore tracking drops (PAUSED or STOPPED), SfM continues uninterrupted — it only
 * needs the camera feed. The last valid [sfmScale] is held so SfM output remains metric
 * until ARCore recovers. Photometric likewise continues independently.
 *
 * ## Voxel deduplication
 *
 * All three sources push into a shared [PointCloudStore] via [onPoints]. Points that map
 * to the same [ArCoreDepthSource.voxelKey] are deduplicated by keeping the highest
 * confidence value — ARCore beats SfM beats Photometric at the same voxel.
 *
 * ## Degradation ladder
 *
 * | Hardware            | Active sources          | Quality                   |
 * |---------------------|-------------------------|---------------------------|
 * | ARCore depth hw     | ARCore + SfM + Photo    | Full — metric, continuous |
 * | No ARCore depth hw  | SfM + Photo             | Relative, continuous      |
 * | No torch            | ARCore (or SfM alone)   | Metric/relative, no normals|
 * | Emulator            | SfM only                | Relative only             |
 *
 * [isAvailable] always returns true — at minimum, SfM works on any device with a camera.
 */
class FusedDepthSource(
    private val context: Context,
    val store: PointCloudStore = PointCloudStore()
) : DepthSource {

    override val mode = DepthSource.Mode.SFM  // composite; mode is informational only

    companion object {
        // Voxel cell size for deduplication (metres)
        private const val VOXEL_CELL = 0.005f

        // SfM scale calibration
        private const val SCALE_EMA_ALPHA  = 0.05f   // slow EMA — stable over many frames
        private const val SCALE_MIN_SAMPLES = 5       // Warm-up: fast EMA alpha, converges in ~5 frames.
        private const val PX_TO_M_DEFAULT  = 0.001f  // SfMDepthSource hardcoded fallback

        // Confidence weights per source — ARCore always wins at a contested voxel
        private const val CONF_ARCORE      = 1.0f
        private const val CONF_SFM_FACTOR  = 0.6f    // multiplied into SfM's internal conf
        private const val CONF_PHOTO_FACTOR = 0.35f  // multiplied into Photometric's conf
    }

    // ─── Sub-sources ──────────────────────────────────────────────────────────

    val arcore      = ArCoreDepthSource(context)
    val sfm         = SfMDepthSource(context)
    val photometric = PhotometricDepthSource(context)

    /** True once the scale calibration has warmed up. */
    var sfmScaleWarm = false
        private set

    /** Current dynamic SfM→metric scale factor (metres per SfM depth unit). */
    var sfmScale: Float = PX_TO_M_DEFAULT
        private set

    private var sfmScaleSamples = 0

    /** Last ARCore world-space camera position (x, y, z in metres). */
    private var arcoreCamX = Float.NaN
    private var arcoreCamY = Float.NaN
    private var arcoreCamZ = Float.NaN

    /** Last SfM cumulative camera position (pixels). */
    private var sfmCamPx = 0f
    private var sfmCamPy = 0f

    // Voxel dedup map: key → best confidence seen so far

    private val MAX_VOXELS = 200_000
    private val voxelConf = LinkedHashMap<Long, Float>(MAX_VOXELS, 0.75f, false)
    private var voxelFlushCounter = 0

    // ─── DepthSource lifecycle ────────────────────────────────────────────────

    override fun isAvailable() = true

    override fun start(callback: DepthSourceCallback) {
        store.clear()
        voxelConf.clear()
        sfmScale = PX_TO_M_DEFAULT
        sfmScaleWarm = false
        sfmScaleSamples = 0

        if (arcore.isAvailable()) {
            arcore.start(ArcoreCallback(callback))
        } else {
            callback.onStats(DepthSource.Stats(extraLabel = "ARCore unavailable — SfM+Photo only"))
        }

        sfm.start(SfmCallback(callback))
        photometric.start(PhotoCallback(callback))
    }

    override fun stop() {
        arcore.stop()
        sfm.stop()
        photometric.stop()
    }

    fun close() {
        stop()
        arcore.close()
    }

    // ─── GL thread hook (forward to ARCore) ───────────────────────────────────

    /**
     * Must be called from [android.opengl.GLSurfaceView.Renderer.onDrawFrame].
     * Required by ARCore's EGL context contract.
     */
    fun onDrawFrame() = arcore.onDrawFrame()

    fun setCameraTextureName(texId: Int) = arcore.setCameraTextureName(texId)

    // ─── Shim wiring (eliminates second camera session) ───────────────────────

    /**
     * Attach a [BitmapGrayscaleShim] to both SfM and Photometric sources.
     * Must be called before [start]. See [BitmapGrayscaleShim] for rationale.
     */
    fun attachShim(shim: BitmapGrayscaleShim) {
        sfm.attachShim(shim)
        photometric.attachShim(shim)
    }

    // ─── Scale calibration ────────────────────────────────────────────────────

    /**
     * Called when ARCore reports a new camera pose. Updates the dynamic SfM scale
     * by correlating ARCore metric displacement with SfM pixel displacement.
     *
     * @param camX  ARCore camera position X in world space (metres)
     * @param camY  ARCore camera position Y in world space (metres)
     * @param camZ  ARCore camera position Z in world space (metres)
     */
    private fun updateScaleCalibration(camX: Float, camY: Float, camZ: Float) {
        val prevX = arcoreCamX; val prevY = arcoreCamY; val prevZ = arcoreCamZ

        arcoreCamX = camX; arcoreCamY = camY; arcoreCamZ = camZ

        if (prevX.isNaN()) return  // first frame

        val arcoreDisp = sqrt(
            (camX - prevX) * (camX - prevX) +
            (camY - prevY) * (camY - prevY) +
            (camZ - prevZ) * (camZ - prevZ)
        )

        // SfM cumulative camera pose is published via its own internal state — we
        // can't directly read it from here. Instead we use the fact that when
        // SfM emits a Stats object, baselinePx reflects the cumulative camera motion.
        // We store the last SfM baseline and diff it here.
        val sfmDisp = sqrt(sfmCamPx * sfmCamPx + sfmCamPy * sfmCamPy)

        if (arcoreDisp < 0.001f || sfmDisp < 0.5f) return  // insufficient motion to calibrate

        val newScale = arcoreDisp / sfmDisp
        // Reject implausible scales — sanity gate: must be within 10× of default
        if (newScale < PX_TO_M_DEFAULT * 0.1f || newScale > PX_TO_M_DEFAULT * 10f) return

        // Fast EMA alpha during warm-up (~5 frames), then slow EMA for stability.
        val alpha = if (sfmScaleWarm) SCALE_EMA_ALPHA else SCALE_EMA_ALPHA * 4f
        sfmScale = sfmScale * (1f - alpha) + newScale * alpha
        sfmScaleSamples++
        if (sfmScaleSamples >= SCALE_MIN_SAMPLES) sfmScaleWarm = true
    }

    // ─── Voxel deduplication ──────────────────────────────────────────────────

    /**
     * Accept a batch of (x,y,z,conf) points, deduplicate by voxel, and push survivors
     * into [store].
     *
     * Higher confidence wins at a contested voxel — this is the mechanism by which
     * ARCore naturally dominates over SfM which dominates over Photometric.
     *
     * [voxelConf] is periodically flushed (every 300 batches) to avoid unbounded growth
     * when the camera moves far from the initial position.
     */
    private fun acceptPoints(batch: FloatArray, len: Int) {
        val n = len / 4
        if (n == 0) return

        val out = FloatArray(len)
        var outIdx = 0

        for (i in 0 until n) {
            val x = batch[i * 4];     val y = batch[i * 4 + 1]
            val z = batch[i * 4 + 2]; val c = batch[i * 4 + 3]

            val key = ArCoreDepthSource.voxelKey(x, y, z, VOXEL_CELL)
            val existing = voxelConf[key] ?: -1f

            if (c > existing) {
                voxelConf[key] = c
                out[outIdx++] = x; out[outIdx++] = y
                out[outIdx++] = z; out[outIdx++] = c
            }
        }

        if (outIdx > 0) store.push(out, outIdx)

        // lowest-confidence half — these are typically SfM/Photometric peripheral points
        // that ARCore has since overwritten at higher confidence.
        if (++voxelFlushCounter > 60) {
            voxelFlushCounter = 0
            if (voxelConf.size > MAX_VOXELS) {
                val threshold = voxelConf.values.sorted().getOrElse(MAX_VOXELS / 2) { 0.5f }
                val iter = voxelConf.iterator()
                while (iter.hasNext()) {
                    if (iter.next().value < threshold) iter.remove()
                }
            }
        }
    }

    // ─── Source-specific callbacks ────────────────────────────────────────────

    private inner class ArcoreCallback(val outer: DepthSourceCallback) : DepthSourceCallback {
        override fun onPoints(batch: FloatArray, len: Int) {
            acceptPoints(batch, len)

            // arcore.lastCamX/Y/Z is set from cam.pose.translation each onDrawFrame() call.
            val cx = arcore.lastCamX
            if (!cx.isNaN()) updateScaleCalibration(cx, arcore.lastCamY, arcore.lastCamZ)
        }
        override fun onStats(stats: DepthSource.Stats) = outer.onStats(stats.copy(extraLabel = "Fused/ARCore"))
        override fun onError(msg: String) = outer.onError("ARCore: $msg")
    }

    private inner class SfmCallback(val outer: DepthSourceCallback) : DepthSourceCallback {
        override fun onPoints(batch: FloatArray, len: Int) {
            val n = len / 4
            if (n == 0) return

            // Rescale SfM depth to metric when scale calibration is warm
            val scaleFactor = if (sfmScaleWarm) sfmScale / PX_TO_M_DEFAULT else 1f
            val scaled = if (abs(scaleFactor - 1f) < 0.01f) batch else {
                FloatArray(len).also { out ->
                    for (i in 0 until n) {
                        out[i * 4]     = batch[i * 4]     * scaleFactor
                        out[i * 4 + 1] = batch[i * 4 + 1] * scaleFactor
                        out[i * 4 + 2] = batch[i * 4 + 2] * scaleFactor
                        out[i * 4 + 3] = batch[i * 4 + 3] * CONF_SFM_FACTOR
                    }
                }
            }
            acceptPoints(scaled, len)
        }
        override fun onStats(stats: DepthSource.Stats) {
            // Capture SfM's cumulative camera position for scale calibration
            sfmCamPx = stats.baselinePx  // SfM reports baseline (distance from first anchor)
            sfmCamPy = 0f
            outer.onStats(stats.copy(extraLabel = if (sfmScaleWarm) "Fused/SfM(calibrated)" else "Fused/SfM(raw)"))
        }
        override fun onError(msg: String) = outer.onError("SfM: $msg")
    }

    private inner class PhotoCallback(val outer: DepthSourceCallback) : DepthSourceCallback {
        override fun onPoints(batch: FloatArray, len: Int) {
            // Photometric source emits stride-7 tuples: x,y,z,conf,nx,ny,nz.
            // Weight each point's confidence by |dot(view_dir, normal)| — concave regions
            // (knuckles, web space) facing the camera get full weight; glancing surfaces
            // near tangent to the view ray are down-weighted (their normals are unreliable).
            val stride = 7
            val n = len / stride
            if (n == 0) return

            val scaleFactor = if (sfmScaleWarm) sfmScale / SfMDepthSource.PX_TO_M else 1f
            val out = FloatArray(n * 4)  // stride-4 for acceptPoints

            for (i in 0 until n) {
                val base = i * stride
                val x = batch[base]     * scaleFactor
                val y = batch[base + 1] * scaleFactor
                val z = batch[base + 2] * scaleFactor
                val rawConf = batch[base + 3]
                val nx = batch[base + 4]; val ny = batch[base + 5]; val nz = batch[base + 6]

                // View direction in camera space is (0,0,-1) — dot with normal = -nz
                // Use absolute value so back-faces (flipped normals) still contribute
                val viewAlign = kotlin.math.abs(nz).coerceIn(0f, 1f)
                val conf = rawConf * CONF_PHOTO_FACTOR * viewAlign

                out[i * 4]     = x;    out[i * 4 + 1] = y
                out[i * 4 + 2] = z;    out[i * 4 + 3] = conf
            }
            acceptPoints(out, n * 4)
        }
        override fun onStats(stats: DepthSource.Stats) =
            outer.onStats(stats.copy(extraLabel = "Fused/Photo(normals)"))
        override fun onError(msg: String) = outer.onError("Photometric: $msg")
    }
}
