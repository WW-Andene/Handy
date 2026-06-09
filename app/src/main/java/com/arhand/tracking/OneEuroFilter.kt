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
    private val minCutoffXY: Float,
    private val betaXY: Float,
    private val minCutoffZ: Float,
    private val betaZ: Float,
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
     * Smooth a list of 21 landmarks. Returns new smoothed list.
     */
    fun smooth(lms: List<Landmark>, nowMs: Long): List<Landmark> {
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
                val sx = filterScalar(s.x, lm.x, nowMs, minCutoffXY, betaXY)
                val sy = filterScalar(s.y, lm.y, nowMs, minCutoffXY, betaXY)
                val sz = filterScalar(s.z, lm.z, nowMs, minCutoffZ, betaZ)
                lm.copy(x = sx, y = sy, z = sz)
            }
        }
    }
}
