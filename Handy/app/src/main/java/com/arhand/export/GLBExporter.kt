package com.arhand.export

import android.content.ContentValues
import android.content.Context
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import com.arhand.util.Vec3
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * GLB Exporter — writes a binary glTF 2.0 file from the reconstructed mesh.
 * Port of SCANNER._exportGLB() from the HTML prototype.
 *
 * Format: GLB = 12-byte header + JSON chunk + BIN chunk
 *
 * The mesh is stored as:
 *   - POSITION accessor (VEC3 FLOAT)
 *   - NORMAL accessor (VEC3 FLOAT) — flat normals
 *   - MeshPrimitive with TRIANGLES mode
 */
object GLBExporter {

    /**
     * Export a triangle mesh to a GLB file.
     * @param positions Flat float array of triangle vertices (x,y,z × 3 per triangle)
     * @param cloudPoints Point cloud used as secondary geometry
     * @return File reference to the exported GLB
     */
    fun export(context: Context, positions: FloatArray, cloudPoints: List<Vec3>): File {
        val vertexCount = positions.size / 3

        // Compute flat normals
        val normals = FloatArray(positions.size)
        var i = 0
        while (i + 8 < positions.size) {
            val ax = positions[i];   val ay = positions[i+1]; val az = positions[i+2]
            val bx = positions[i+3]; val by = positions[i+4]; val bz = positions[i+5]
            val cx = positions[i+6]; val cy = positions[i+7]; val cz = positions[i+8]
            val ux = bx-ax; val uy = by-ay; val uz = bz-az
            val vx = cx-ax; val vy = cy-ay; val vz = cz-az
            var nx = uy*vz - uz*vy; var ny = uz*vx - ux*vz; var nz = ux*vy - uy*vx
            val len = Math.sqrt((nx*nx + ny*ny + nz*nz).toDouble()).toFloat().coerceAtLeast(1e-6f)
            nx /= len; ny /= len; nz /= len
            for (j in 0 until 3) { normals[i+j*3] = nx; normals[i+j*3+1] = ny; normals[i+j*3+2] = nz }
            i += 9
        }

        // Compute AABB for accessor min/max
        var minX = Float.MAX_VALUE; var minY = Float.MAX_VALUE; var minZ = Float.MAX_VALUE
        var maxX = -Float.MAX_VALUE; var maxY = -Float.MAX_VALUE; var maxZ = -Float.MAX_VALUE
        for (vi in 0 until vertexCount) {
            val x = positions[vi*3]; val y = positions[vi*3+1]; val z = positions[vi*3+2]
            if (x < minX) minX = x; if (y < minY) minY = y; if (z < minZ) minZ = z
            if (x > maxX) maxX = x; if (y > maxY) maxY = y; if (z > maxZ) maxZ = z
        }

        // Build BIN buffer: positions then normals
        val binSize = (positions.size + normals.size) * 4
        val bin = ByteBuffer.allocate(binSize).order(ByteOrder.LITTLE_ENDIAN)
        positions.forEach { bin.putFloat(it) }
        normals.forEach   { bin.putFloat(it) }
        val binBytes = bin.array()

        val posOffset  = 0
        val normOffset = positions.size * 4
        val posBytes   = positions.size * 4
        val normBytes  = normals.size * 4

        // JSON chunk
        val json = """{"asset":{"version":"2.0","generator":"Handy AR Hand Scanner"},"scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0}],"meshes":[{"name":"hand","primitives":[{"attributes":{"POSITION":0,"NORMAL":1},"mode":4}]}],"accessors":[{"bufferView":0,"componentType":5126,"count":$vertexCount,"type":"VEC3","min":[$minX,$minY,$minZ],"max":[$maxX,$maxY,$maxZ]},{"bufferView":1,"componentType":5126,"count":$vertexCount,"type":"VEC3"}],"bufferViews":[{"buffer":0,"byteOffset":$posOffset,"byteLength":$posBytes},{"buffer":0,"byteOffset":$normOffset,"byteLength":$normBytes}],"buffers":[{"byteLength":$binSize}]}"""

        val jsonBytes = json.toByteArray(Charsets.US_ASCII)
        val jsonPadded = jsonBytes + ByteArray((4 - jsonBytes.size % 4) % 4) { 0x20 }
        val binPadded  = binBytes  + ByteArray((4 - binBytes.size  % 4) % 4) { 0x00 }

        val totalLength = 12 + 8 + jsonPadded.size + 8 + binPadded.size

        val glb = ByteBuffer.allocate(totalLength).order(ByteOrder.LITTLE_ENDIAN)
        // GLB header
        glb.putInt(0x46546C67)               // magic "glTF"
        glb.putInt(2)                         // version
        glb.putInt(totalLength)               // total length
        // JSON chunk
        glb.putInt(jsonPadded.size)
        glb.putInt(0x4E4F534A)               // "JSON"
        glb.put(jsonPadded)
        // BIN chunk
        glb.putInt(binPadded.size)
        glb.putInt(0x004E4942)               // "BIN\0"
        glb.put(binPadded)

        val fileName = "handy_${System.currentTimeMillis()}.glb"
        val file = File(context.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS), fileName)
        FileOutputStream(file).use { it.write(glb.array()) }

        return file
    }
}
