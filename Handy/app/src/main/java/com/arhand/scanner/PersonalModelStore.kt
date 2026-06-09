package com.arhand.scanner

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.*
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map
import java.io.File

/**
 * Persists the personal hand model to DataStore + local file.
 * Replaces localStorage from the HTML prototype.
 *
 * Stores:
 *   - Scan metadata (timestamp, point count, pose quality scores)
 *   - GLB export path (local file URI)
 *   - Per-pose quality map
 */

private val Context.modelDataStore: DataStore<Preferences> by preferencesDataStore(name = "handy_model")

class PersonalModelStore(private val context: Context) {

    companion object {
        val KEY_TIMESTAMP    = stringPreferencesKey("scan_timestamp")
        val KEY_POINT_COUNT  = intPreferencesKey("point_count")
        val KEY_GLB_PATH     = stringPreferencesKey("glb_path")
        val KEY_POSE_SCORES  = stringPreferencesKey("pose_scores_json")
        val KEY_HAS_MODEL    = booleanPreferencesKey("has_model")
    }

    suspend fun saveModel(
        glbFile: File,
        pointCount: Int,
        poseScoresJson: String
    ) {
        context.modelDataStore.edit { prefs ->
            prefs[KEY_TIMESTAMP]   = System.currentTimeMillis().toString()
            prefs[KEY_POINT_COUNT] = pointCount
            prefs[KEY_GLB_PATH]    = glbFile.absolutePath
            prefs[KEY_POSE_SCORES] = poseScoresJson
            prefs[KEY_HAS_MODEL]   = true
        }
    }

    suspend fun loadModel(): ModelMeta? {
        val prefs = context.modelDataStore.data.first()
        if (prefs[KEY_HAS_MODEL] != true) return null
        return ModelMeta(
            timestamp   = prefs[KEY_TIMESTAMP]   ?: "",
            pointCount  = prefs[KEY_POINT_COUNT] ?: 0,
            glbPath     = prefs[KEY_GLB_PATH]    ?: "",
            poseScores  = prefs[KEY_POSE_SCORES] ?: "{}"
        )
    }

    suspend fun clearModel() {
        context.modelDataStore.edit { it.clear() }
    }

    suspend fun hasModel(): Boolean =
        context.modelDataStore.data.map { it[KEY_HAS_MODEL] == true }.first()
}

data class ModelMeta(
    val timestamp: String,
    val pointCount: Int,
    val glbPath: String,
    val poseScores: String
)
