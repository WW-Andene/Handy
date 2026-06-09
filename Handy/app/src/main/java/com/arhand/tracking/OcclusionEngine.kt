package com.arhand.tracking

import kotlin.math.*

/**
 * Occlusion Engine — port of detectOcclusion() + inferOccludedLandmark() from the HTML prototype.
 *
 * Detects likely occluded landmarks using biomechanical constraints:
 *   1. Palm plane normal check
 *   2. Segment length deviation check
 *   3. Z-chain reversal check (distal landmark shouldn't be behind proximal)
 *
 * Infers occluded positions using 3-way blend:
 *   FK (50%) + velocity prediction (35%) + history average (15%)
 */
class OcclusionEngine {

    companion object {
        val FINGER_CHAINS = listOf(
            listOf(0, 1, 2, 3, 4),    // Thumb
            listOf(0, 5, 6, 7, 8),    // Index
            listOf(0, 9, 10, 11, 12), // Middle
            listOf(0, 13, 14, 15, 16),// Ring
            listOf(0, 17, 18, 19, 20) // Pinky
        )

        const val VEL_EMA_ALPHA     = 0.35f
        const val VEL_EXTRAP_MAX_SEC = 0.20f   // 200ms ≈ 6 frames at 30fps
        const val OCC_PROB_EMA_ALPHA = 0.4f
        const val OCC_PROB_THRESHOLD = 0.55f
    }

    // Per-landmark: EMA occlusion probability
    val occlusionProb = FloatArray(21) { 0f }

    // Velocity history per landmark
    data class VelState(var vx: Float = 0f, var vy: Float = 0f, var vz: Float = 0f, var tMs: Long = 0L)
    private val velState = Array(21) { VelState() }

    // Position history ring buffer (last 4 frames)
    private val histSize = 4
    private val history: Array<Array<FloatArray>> = Array(histSize) { Array(21) { FloatArray(3) } }
    private var histIdx = 0

    fun reset() {
        occlusionProb.fill(0f)
        velState.forEach { it.vx = 0f; it.vy = 0f; it.vz = 0f; it.tMs = 0L }
        for (h in history) for (lm in h) lm.fill(0f)
        histIdx = 0
    }

    /**
     * Detect which landmarks are likely occluded.
     * Returns Uint8Array equivalent: 1 = occluded.
     */
    fun detectOcclusion(lms: List<Landmark>): BooleanArray {
        val occluded = BooleanArray(21) { false }
        if (lms.size < 21) return occluded

        // Palm plane normal (wrist → index MCP, wrist → pinky MCP)
        val p0 = lms[0]; val p5 = lms[5]; val p17 = lms[17]
        val v1x = p5.x - p0.x; val v1y = p5.y - p0.y; val v1z = (p5.z) - (p0.z)
        val v2x = p17.x - p0.x; val v2y = p17.y - p0.y; val v2z = (p17.z) - (p0.z)
        val nx = v1y * v2z - v1z * v2y
        val ny = v1z * v2x - v1x * v2z
        val nz = v1x * v2y - v1y * v2x
        val nLen = sqrt(nx * nx + ny * ny + nz * nz).coerceAtLeast(1e-6f)

        val palmRef = sqrt(
            (p5.x - p0.x).pow(2) + (p5.y - p0.y).pow(2) + ((p5.z) - (p0.z)).pow(2)
        ).coerceAtLeast(0.001f)

        for (chain in FINGER_CHAINS) {
            for (ci in 1 until chain.size) {
                val idx = chain[ci]
                val prev = lms[chain[ci - 1]]
                val curr = lms[idx]

                // Check 1: segment length deviation (too short = folded/occluded)
                val segLen = sqrt(
                    (curr.x - prev.x).pow(2) + (curr.y - prev.y).pow(2) + ((curr.z) - (prev.z)).pow(2)
                )
                val expectedRatio = if (ci == 1) 0.6f else 0.45f
                if (segLen < expectedRatio * palmRef * 0.3f) {
                    occluded[idx] = true
                }

                // Check 2: z-chain reversal (distal shouldn't be much behind proximal)
                if (ci > 1 && curr.z > prev.z + 0.04f) {
                    occluded[idx] = true
                }

                // Check 3: palm normal projection (landmark behind palm plane)
                val dot = (curr.x - p0.x) * (nx / nLen) +
                          (curr.y - p0.y) * (ny / nLen) +
                          (curr.z - p0.z) * (nz / nLen)
                if (dot < -0.08f) {
                    occluded[idx] = true
                }
            }
        }

        // Update EMA occlusion probability
        for (i in 0 until 21) {
            val raw = if (occluded[i]) 1f else 0f
            occlusionProb[i] = occlusionProb[i] * (1f - OCC_PROB_EMA_ALPHA) + raw * OCC_PROB_EMA_ALPHA
        }

        return occluded
    }

    /**
     * Update velocity history from accepted (non-inferred) landmarks.
     */
    fun updateVelocity(lms: List<Landmark>, occluded: BooleanArray, nowMs: Long) {
        if (lms.size < 21) return
        for (i in 0 until 21) {
            if (occluded[i]) continue
            val v = velState[i]
            if (v.tMs > 0L) {
                val dt = maxOf((nowMs - v.tMs) * 0.001f, 1e-4f)
                val rawVx = (lms[i].x - history[histIdx][i][0]) / dt
                val rawVy = (lms[i].y - history[histIdx][i][1]) / dt
                val rawVz = (lms[i].z - history[histIdx][i][2]) / dt
                v.vx = v.vx + VEL_EMA_ALPHA * (rawVx - v.vx)
                v.vy = v.vy + VEL_EMA_ALPHA * (rawVy - v.vy)
                v.vz = v.vz + VEL_EMA_ALPHA * (rawVz - v.vz)
            }
            history[histIdx][i][0] = lms[i].x
            history[histIdx][i][1] = lms[i].y
            history[histIdx][i][2] = lms[i].z
            v.tMs = nowMs
        }
        histIdx = (histIdx + 1) % histSize
    }

    /**
     * Infer position of an occluded landmark using FK + velocity + history blend.
     * FK 50% + velocity 35% + history 15%
     */
    fun inferLandmark(lms: MutableList<Landmark>, lmIdx: Int, occluded: BooleanArray, nowMs: Long): Landmark {
        val chain = FINGER_CHAINS.find { it.contains(lmIdx) } ?: return lms[lmIdx]
        val ci = chain.indexOf(lmIdx)
        if (ci <= 0) return lms[lmIdx]

        val prevIdx = chain[ci - 1]
        val prev = lms[prevIdx]

        // FK: extend from previous using average direction of chain
        val dx: Float; val dy: Float; val dz: Float
        if (ci >= 2) {
            val pp = lms[chain[ci - 2]]
            dx = prev.x - pp.x; dy = prev.y - pp.y; dz = prev.z - pp.z
        } else {
            dx = 0f; dy = -0.05f; dz = 0f
        }
        val fkX = prev.x + dx * 0.8f
        val fkY = prev.y + dy * 0.8f
        val fkZ = prev.z + dz * 0.8f

        // Velocity prediction
        val v = velState[lmIdx]
        val dt = if (v.tMs > 0L) minOf((nowMs - v.tMs) * 0.001f, VEL_EXTRAP_MAX_SEC) else 0f
        val fade = 1f - dt / VEL_EXTRAP_MAX_SEC
        val histLast = history[(histIdx + histSize - 1) % histSize][lmIdx]
        val velX = histLast[0] + v.vx * dt * fade
        val velY = histLast[1] + v.vy * dt * fade
        val velZ = histLast[2] + v.vz * dt * fade

        // History average
        var hx = 0f; var hy = 0f; var hz = 0f
        for (h in history) { hx += h[lmIdx][0]; hy += h[lmIdx][1]; hz += h[lmIdx][2] }
        hx /= histSize; hy /= histSize; hz /= histSize

        // Blend: FK 50% + velocity 35% + history 15%
        val rx = fkX * 0.50f + velX * 0.35f + hx * 0.15f
        val ry = fkY * 0.50f + velY * 0.35f + hy * 0.15f
        val rz = fkZ * 0.50f + velZ * 0.35f + hz * 0.15f

        return Landmark(rx, ry, rz, inferred = true)
    }

    /**
     * Apply occlusion inference to a full landmark list.
     * Returns new list with inferred landmarks marked.
     */
    fun applyInference(lms: List<Landmark>, confidence: Float, nowMs: Long): List<Landmark> {
        if (lms.size < 21 || confidence < 0.3f) return lms
        val occluded = detectOcclusion(lms)
        updateVelocity(lms, occluded, nowMs)
        val result = lms.toMutableList()
        for (i in 0 until 21) {
            if (occlusionProb[i] > OCC_PROB_THRESHOLD) {
                result[i] = inferLandmark(result, i, occluded, nowMs)
            }
        }
        return result
    }
}
