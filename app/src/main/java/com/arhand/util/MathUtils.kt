package com.arhand.util

import com.arhand.tracking.Landmark
import kotlin.math.sqrt

// View scale — matches HTML prototype VIEW_SCALE = 0.65
const val VIEW_SCALE = 0.65f

/**
 * Distance between two landmarks in 3D space.
 */
fun lmDist(a: Landmark, b: Landmark): Float {
    val dx = a.x - b.x
    val dy = a.y - b.y
    val dz = a.z - b.z
    return sqrt(dx * dx + dy * dy + dz * dz)
}

/**
 * Clamp per-frame landmark delta to prevent jumps.
 * Mirrors clampDelta() in the HTML prototype.
 */
fun clampDelta(prev: List<Landmark>, next: List<Landmark>, maxDelta: Float = 0.15f): List<Landmark> {
    return next.mapIndexed { i, n ->
        if (i >= prev.size) n
        else {
            val dx = (n.x - prev[i].x).coerceIn(-maxDelta, maxDelta)
            val dy = (n.y - prev[i].y).coerceIn(-maxDelta, maxDelta)
            val dz = (n.z - prev[i].z).coerceIn(-maxDelta, maxDelta)
            n.copy(x = prev[i].x + dx, y = prev[i].y + dy, z = prev[i].z + dz)
        }
    }
}

/**
 * Linear interpolation between two floats.
 */
fun lerp(a: Float, b: Float, t: Float): Float = a + (b - a) * t

/**
 * Convert normalized landmark coordinates to world space.
 * Mirrors the mapping used by HandRenderer and SCANNER.buildModel in the prototype.
 *
 * When MediaPipe world landmarks are available (worldX/worldY/worldZ all non-zero),
 * all three metric axes are used directly — eliminating the projection distortion
 * that the screen-space x/y path introduces (normalized coords encode both position
 * and hand-camera distance). The world landmark coordinate system is hand-center
 * origin, Y-up, X-right, Z toward camera; negate X for mirror and Z for right-hand
 * renderer convention.
 *
 * Falls back to the screen-space reconstruction when world landmarks are unavailable
 * (worldX == 0f && worldY == 0f), preserving full backward compatibility.
 *
 * @param lm      Normalized landmark (x,y in [0,1], z relative, worldX/Y/Z in meters)
 * @param aspect  Viewport aspect ratio (width / height) — only used in fallback path
 * @param mirrorX Whether to mirror X (front camera = true)
 * @return Triple of (worldX, worldY, worldZ)
 */
fun landmarkToWorld(lm: Landmark, aspect: Float, mirrorX: Boolean): Triple<Float, Float, Float> {
    // BUG-2 fix: use all three metric world axes when available.
    // Previous code fetched worldZ from the world landmark API but reconstructed
    // X and Y from projection-distorted screen-space coordinates, discarding the
    // metric X/Y that MediaPipe computes. worldX and worldY are now populated in
    // HandTracker.parseResult() alongside worldZ.
    if (lm.worldX != 0f || lm.worldY != 0f) {
        // Metric world-space path — no projection distortion.
        // MediaPipe world X: positive = right (camera frame). Mirror for front camera.
        // MediaPipe world Y: positive = up. No inversion needed.
        // MediaPipe world Z: positive = toward camera. Negate for renderer convention.
        val wx = if (mirrorX) -lm.worldX else lm.worldX
        return Triple(wx, -lm.worldY, -lm.worldZ)
    }
    // Fallback: screen-space reconstruction — unchanged from original.
    val sx = if (mirrorX) 1f - lm.x else lm.x
    val wx = (sx - 0.5f) * 2f * aspect * VIEW_SCALE
    val wy = -(lm.y - 0.5f) * 2f * VIEW_SCALE
    val wz = if (lm.worldZ != 0f) -lm.worldZ else lm.z * -0.6f
    return Triple(wx, wy, wz)
}

/**
 * Simple 3-component vector — used for geometry math without allocating THREE.Vector3.
 */
data class Vec3(val x: Float, val y: Float, val z: Float) {
    operator fun plus(o: Vec3) = Vec3(x + o.x, y + o.y, z + o.z)
    operator fun minus(o: Vec3) = Vec3(x - o.x, y - o.y, z - o.z)
    operator fun times(s: Float) = Vec3(x * s, y * s, z * s)
    fun dot(o: Vec3) = x * o.x + y * o.y + z * o.z
    fun cross(o: Vec3) = Vec3(y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x)
    fun length() = sqrt(x * x + y * y + z * z)
    fun normalized(): Vec3 {
        val l = length()
        return if (l < 1e-8f) Vec3(0f, 1f, 0f) else this * (1f / l)
    }
    fun lerp(o: Vec3, t: Float) = Vec3(
        x + (o.x - x) * t,
        y + (o.y - y) * t,
        z + (o.z - z) * t
    )
}
