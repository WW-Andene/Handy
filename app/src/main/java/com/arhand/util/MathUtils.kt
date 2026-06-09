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
 * @param lm      Normalized landmark (x,y in [0,1], z relative)
 * @param aspect  Viewport aspect ratio (width / height)
 * @param mirrorX Whether to mirror X (front camera = true)
 * @return Triple of (worldX, worldY, worldZ)
 */
fun landmarkToWorld(lm: Landmark, aspect: Float, mirrorX: Boolean): Triple<Float, Float, Float> {
    val sx = if (mirrorX) 1f - lm.x else lm.x
    val wx = (sx - 0.5f) * 2f * aspect * VIEW_SCALE
    val wy = -(lm.y - 0.5f) * 2f * VIEW_SCALE
    val wz = lm.z * -0.6f
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
