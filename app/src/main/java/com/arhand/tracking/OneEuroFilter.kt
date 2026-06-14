package com.arhand.tracking

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.exp

/**
 * One Euro Filter — port of the Géry Casiez implementation from the HTML prototype.
 *
 * Applies adaptive frequency filtering per landmark per axis (x, y, z).
 * Low beta = smooth but laggy; high beta = responsive but noisy.
 * Adjust minCutoff to control static noise, beta to control lag.
 */
class OneEuroFilter(
    private var minCutoffXY: Float,
    private var betaXY: Float,
    private var minCutoffZ: Float,
    private var betaZ: Float,
    private val dCutoff: Float = 1.0f   // Hz — standard per paper
) {
    data class ScalarState(
        var value: Float,
        var dvalue: Float,
        var lastTimeMs: Long
    )

    data class LandmarkState(
        val x: ScalarState,
        val y: ScalarState,
        val z: ScalarState
    )

    private var states: Array<LandmarkState?>? = null

    private fun alpha(cutoff: Float, dt: Float): Float {
        val tau = 1.0f / (2.0f * PI.toFloat() * cutoff)
        return 1.0f / (1.0f + tau / dt)
    }

    private fun filterScalar(
        state: ScalarState,
        raw: Float,
        nowMs: Long,
        minCutoff: Float,
        beta: Float
    ): Float {
        val dt = maxOf((nowMs - state.lastTimeMs) * 0.001f, 1e-4f)
        val aD = alpha(dCutoff, dt)
        val dRaw = (raw - state.value) / dt
        state.dvalue = state.dvalue + aD * (dRaw - state.dvalue)
        val cutoff = minCutoff + beta * abs(state.dvalue)
        val a = alpha(cutoff, dt)
        state.value = state.value + a * (raw - state.value)
        state.lastTimeMs = nowMs
        return state.value
    }

    fun ensureInit(numLandmarks: Int, nowMs: Long) {
        if (states == null || states!!.size != numLandmarks) {
            states = Array(numLandmarks) { null }
        }
    }

    fun reset() {
        states = null
    }

    /**
     * PERF-2 — Update filter parameters at runtime.
     * Resets internal state so the filter reinitialises on the next frame.
     * The new parameters take effect immediately on the next [smoothPerJoint] call.
     */
    fun setParams(newMinCutoffXY: Float, newBetaXY: Float, newMinCutoffZ: Float, newBetaZ: Float) {
        minCutoffXY = newMinCutoffXY
        betaXY      = newBetaXY
        minCutoffZ  = newMinCutoffZ
        betaZ       = newBetaZ
        reset()
    }

    /**
     * Smooth a list of 21 landmarks. Returns new smoothed list.
     */
    fun smooth(lms: List<Landmark>, nowMs: Long): List<Landmark> =
        smooth(lms, nowMs, minCutoffXY, betaXY)

    /**
     * A2 — Confidence-weighted variant: caller supplies effective cutoff/beta.
     */
    fun smooth(lms: List<Landmark>, nowMs: Long, effectiveCutoffXY: Float, effectiveBetaXY: Float): List<Landmark> {
        ensureInit(lms.size, nowMs)
        val st = states!!
        return lms.mapIndexed { i, lm ->
            if (st[i] == null) {
                st[i] = LandmarkState(
                    ScalarState(lm.x, 0f, nowMs),
                    ScalarState(lm.y, 0f, nowMs),
                    ScalarState(lm.z, 0f, nowMs)
                )
                lm
            } else {
                val s = st[i]!!
                val sx = filterScalar(s.x, lm.x, nowMs, effectiveCutoffXY, effectiveBetaXY)
                val sy = filterScalar(s.y, lm.y, nowMs, effectiveCutoffXY, effectiveBetaXY)
                val sz = filterScalar(s.z, lm.z, nowMs, minCutoffZ,        betaZ)
                lm.copy(x = sx, y = sy, z = sz)
            }
        }
    }

    /**
     * IMP-R1 — Per-landmark dynamic params: each landmark gets its own cutoff/beta
     * derived from [speedRatios] (from TDF.lastSpeedRatios).
     *
     * Rationale: a fast-moving fingertip and a nearly-still wrist in the same frame
     * need different filter parameters. A single global cutoff is always a compromise.
     * Fast joints: lower cutoff (more responsive — don't lag behind the motion).
     * Slow joints: higher cutoff (more aggressive filtering — suppress static noise).
     *
     * @param speedRatios  Per-landmark speed ratios in [0,1] from [TemporalDepthFusion.lastSpeedRatios].
     *                     If null or wrong length, falls back to confidence-weighted global params.
     * @param baseCutoff   Base minCutoff at zero speed (high smoothing).
     * @param fastCutoff   Cutoff at full speed (pass-through).
     */
    fun smoothPerJoint(
        lms: List<Landmark>,
        nowMs: Long,
        speedRatios: FloatArray?,
        baseCutoff: Float,
        fastCutoff: Float,
        baseBeta: Float,
        fastBeta: Float
    ): List<Landmark> {
        ensureInit(lms.size, nowMs)
        val st = states!!
        return lms.mapIndexed { i, lm ->
            val ratio = speedRatios?.getOrNull(i) ?: 0f
            val eCutoff = baseCutoff + (fastCutoff - baseCutoff) * ratio
            val eBeta   = baseBeta   + (fastBeta   - baseBeta)   * ratio
            if (st[i] == null) {
                st[i] = LandmarkState(
                    ScalarState(lm.x, 0f, nowMs),
                    ScalarState(lm.y, 0f, nowMs),
                    ScalarState(lm.z, 0f, nowMs)
                )
                lm
            } else {
                val s = st[i]!!
                val sx = filterScalar(s.x, lm.x, nowMs, eCutoff, eBeta)
                val sy = filterScalar(s.y, lm.y, nowMs, eCutoff, eBeta)
                val sz = filterScalar(s.z, lm.z, nowMs, minCutoffZ, betaZ)
                lm.copy(x = sx, y = sy, z = sz)
            }
        }
    }
}
