package com.arhand.tracking

/**
 * Temporal Depth Fusion (TDF) — port of applyTemporalDepthFusion() from the HTML prototype.
 *
 * Smooths MediaPipe's noisy Z proxy using a speed-adaptive EMA.
 * Fast motion → high alpha (responsive).
 * Slow/static → low alpha (smooth).
 *
 * This does NOT produce real metric depth — MediaPipe Z is still a proxy.
 * TDF only removes the frame-to-frame jitter that makes the proxy unusable for rendering.
 */
class TemporalDepthFusion {

    companion object {
        const val ALPHA_MIN    = 0.12f   // ~8-frame decay at rest
        const val ALPHA_MAX    = 0.55f   // responsive for fast motion
        const val SPEED_THRESH = 0.30f   // z-velocity (norm. coords/sec) at which α saturates
    }

    data class LandmarkDepthState(
        var z: Float,
        var dz: Float = 0f,       // EMA z velocity
        var tMs: Long,
        var primed: Boolean = false
    )

    private var states: Array<LandmarkDepthState?>? = null

    fun reset() {
        states = null
    }

    /**
     * Apply TDF in-place on a mutable landmark list.
     * Returns a new list with fused Z values.
     */
    fun apply(lms: List<Landmark>, nowMs: Long): List<Landmark> {
        if (states == null || states!!.size != lms.size) {
            states = Array(lms.size) { null }
        }
        val st = states!!
        return lms.mapIndexed { i, lm ->
            val rawZ = lm.z
            if (st[i] == null) {
                st[i] = LandmarkDepthState(z = rawZ, tMs = nowMs, primed = false)
                lm
            } else {
                val s = st[i]!!
                if (!s.primed) {
                    s.z = rawZ
                    s.tMs = nowMs
                    s.primed = true
                    lm
                } else {
                    val dt = maxOf((nowMs - s.tMs) * 0.001f, 1e-4f)
                    val instDz = (rawZ - s.z) / dt
                    // EMA velocity
                    s.dz = s.dz * 0.7f + instDz * 0.3f
                    val speedRatio = minOf(1f, Math.abs(s.dz) / SPEED_THRESH)
                    val alpha = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * speedRatio
                    s.z = s.z + alpha * (rawZ - s.z)
                    s.tMs = nowMs
                    lm.copy(z = s.z)
                }
            }
        }
    }
}
