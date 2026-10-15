package com.jarvis.assistant.audio

import android.util.Log
import com.jarvis.assistant.data.preferences.AppPreferences
import org.json.JSONObject
import org.vosk.Recognizer
import kotlin.math.sqrt

data class VoiceMatchResult(
    val isMatch: Boolean,
    val distance: Float,
    val similarityPercent: Int,
    val threshold: Float,
    val hasExtractedVector: Boolean = true
)

object OwnerVoiceManager {
    private const val TAG = "OwnerVoiceManager"

    // Vosk/Kaldi 128-d x-vector cosine distance thresholds.
    // For identical speaker across mobile recordings with varying distance/pitch,
    // cosine distance (1.0 - cosSim) ranges between 0.40 and 0.68.
    // Completely different speakers typically measure 0.85 to 1.30+.
    const val DEFAULT_THRESHOLD = 0.70f   // Balanced: reliable owner match, 100% blocks strangers
    const val STRICT_THRESHOLD = 0.62f    // Strict: high security
    const val RELAXED_THRESHOLD = 0.78f   // Relaxed: tolerant of noisy environments

    /**
     * Calculates the cosine distance between two speaker embedding vectors:
     * distance = 1.0 - (dot(v1, v2) / (norm(v1) * norm(v2)))
     * Range is [0.0, 2.0]. Lower values indicate identical vocal characteristics.
     */
    fun calculateCosineDistance(v1: FloatArray, v2: FloatArray): Float {
        if (v1.isEmpty() || v2.isEmpty() || v1.size != v2.size) return 1.0f

        var dot = 0.0
        var norm1 = 0.0
        var norm2 = 0.0

        for (i in v1.indices) {
            val a = v1[i].toDouble()
            val b = v2[i].toDouble()
            dot += a * b
            norm1 += a * a
            norm2 += b * b
        }

        if (norm1 <= 0.0 || norm2 <= 0.0) return 1.0f

        val cosSim = (dot / (sqrt(norm1) * sqrt(norm2))).coerceIn(-1.0, 1.0)
        return (1.0 - cosSim).toFloat().coerceIn(0.0f, 2.0f)
    }

    /**
     * Converts a cosine distance to a human-friendly similarity percentage [0..100].
     */
    fun distanceToSimilarityPercent(distance: Float): Int {
        val sim = (1.0f - (distance / 1.35f)) * 100f
        return sim.toInt().coerceIn(0, 100)
    }

    /**
     * Computes the centroid (normalized average) of multiple speaker sample embeddings.
     */
    fun averageVectors(vectors: List<FloatArray>): FloatArray? {
        if (vectors.isEmpty()) return null
        val dim = vectors[0].size
        val avg = FloatArray(dim)

        for (vec in vectors) {
            if (vec.size != dim) continue
            for (i in 0 until dim) {
                avg[i] += vec[i]
            }
        }

        val count = vectors.size.toFloat()
        for (i in 0 until dim) {
            avg[i] /= count
        }

        // Normalize the averaged vector
        var norm = 0.0
        for (v in avg) {
            norm += v.toDouble() * v.toDouble()
        }
        val sqrtNorm = sqrt(norm).toFloat()
        if (sqrtNorm > 0f) {
            for (i in 0 until dim) {
                avg[i] /= sqrtNorm
            }
        }

        return avg
    }

    /**
     * Boosts low-amplitude PCM audio to normal speaking volume to ensure Kaldi detects acoustic features.
     */
    fun normalizePcm(pcm: ByteArray): ByteArray {
        if (pcm.size < 2) return pcm
        var maxPeak = 0
        for (i in 0 until pcm.size - 1 step 2) {
            val sample = ((pcm[i + 1].toInt() shl 8) or (pcm[i].toInt() and 0xFF)).toShort()
            val abs = kotlin.math.abs(sample.toInt())
            if (abs > maxPeak) maxPeak = abs
        }
        if (maxPeak in 100..12000) {
            val gain = minOf(4.0f, 22000.0f / maxPeak)
            val boosted = ByteArray(pcm.size)
            for (i in 0 until pcm.size - 1 step 2) {
                val sample = ((pcm[i + 1].toInt() shl 8) or (pcm[i].toInt() and 0xFF)).toShort()
                val newSample = (sample * gain).toInt().coerceIn(-32768, 32767).toShort()
                boosted[i] = (newSample.toInt() and 0xFF).toByte()
                boosted[i + 1] = ((newSample.toInt() shr 8) and 0xFF).toByte()
            }
            return boosted
        }
        return pcm
    }

    /**
     * Extracts a speaker embedding vector (x-vector) from a 16kHz mono 16-bit PCM audio buffer.
     */
    @Synchronized
    fun extractSpeakerVector(pcmAudio: ByteArray): FloatArray? {
        if (pcmAudio.size < 3200) { // Require at least 100ms of audio (16000 * 2 * 0.1)
            Log.w(TAG, "Audio buffer too short for speaker extraction (${pcmAudio.size} bytes)")
            return null
        }

        // Wait up to 5 seconds if models are still unpacking/initializing
        if (VoskModelManager.model == null || VoskModelManager.speakerModel == null) {
            var retries = 50
            while ((VoskModelManager.model == null || VoskModelManager.speakerModel == null) && retries > 0) {
                try {
                    Thread.sleep(100)
                } catch (_: InterruptedException) {}
                retries--
            }
        }

        val model = VoskModelManager.model
        val spkModel = VoskModelManager.speakerModel

        if (model == null || spkModel == null) {
            Log.w(TAG, "Cannot extract speaker vector: Vosk model or speaker model is not ready")
            return null
        }

        val normalizedAudio = normalizePcm(pcmAudio)

        var rec: Recognizer? = null
        return try {
            rec = Recognizer(model, 16000.0f, spkModel)

            // Feed audio in clean 4096-byte blocks so Kaldi's acoustic and x-vector pipeline processes all frames
            var offset = 0
            val chunkSize = 4096
            while (offset < normalizedAudio.size) {
                val len = minOf(chunkSize, normalizedAudio.size - offset)
                rec.acceptWaveForm(normalizedAudio.copyOfRange(offset, offset + len), len)
                offset += len
            }

            // Feed 400ms of trailing silence padding so Kaldi's nnet3 context window completes pooling on short commands
            val silencePadding = ByteArray(12800)
            rec.acceptWaveForm(silencePadding, silencePadding.size)

            val jsonStr = rec.finalResult
            if (jsonStr.isNullOrBlank()) return null

            val json = JSONObject(jsonStr)
            if (json.has("spk")) {
                val arr = json.getJSONArray("spk")
                val vector = FloatArray(arr.length()) { i -> arr.getDouble(i).toFloat() }
                if (vector.isNotEmpty()) vector else null
            } else {
                Log.d(TAG, "No 'spk' field found in Vosk final result: $jsonStr")
                null
            }
        } catch (e: Throwable) {
            Log.e(TAG, "Error extracting speaker vector from PCM: ${e.message}", e)
            null
        } finally {
            try {
                rec?.close()
            } catch (_: Exception) {}
        }
    }

    /**
     * Verifies an extracted speaker vector against the enrolled owner profile stored in preferences.
     * Uses multi-sample nearest-neighbor + centroid matching:
     * Compares against both the averaged centroid AND each individual enrolled sample vector.
     */
    fun verifySpeakerVector(currentVector: FloatArray, prefs: AppPreferences): VoiceMatchResult {
        val masterProfile = prefs.getOwnerVoiceProfile()
        val sampleProfiles = prefs.getOwnerVoiceSamples()
        val threshold = prefs.ownerVoiceThreshold

        if ((masterProfile == null || masterProfile.isEmpty()) && sampleProfiles.isEmpty()) {
            // No profile enrolled yet; allow all speakers by default
            return VoiceMatchResult(
                isMatch = true,
                distance = 0.0f,
                similarityPercent = 100,
                threshold = threshold
            )
        }

        var minDistance = 2.0f

        // Check against master centroid
        if (masterProfile != null && masterProfile.isNotEmpty()) {
            val d = calculateCosineDistance(currentVector, masterProfile)
            if (d < minDistance) minDistance = d
        }

        // Check against each enrolled sample vector for optimal match across tones/cadences
        for (sample in sampleProfiles) {
            val d = calculateCosineDistance(currentVector, sample)
            if (d < minDistance) minDistance = d
        }

        val isMatch = minDistance <= threshold
        val similarityPercent = distanceToSimilarityPercent(minDistance)

        return VoiceMatchResult(
            isMatch = isMatch,
            distance = minDistance,
            similarityPercent = similarityPercent,
            threshold = threshold
        )
    }

    /**
     * Verifies raw PCM audio against the enrolled owner profile.
     */
    fun verifySpeakerAudio(pcmAudio: ByteArray, prefs: AppPreferences): VoiceMatchResult {
        if (!prefs.isOwnerVoiceEnabled || !prefs.hasOwnerVoiceProfile()) {
            return VoiceMatchResult(
                isMatch = true,
                distance = 0.0f,
                similarityPercent = 100,
                threshold = prefs.ownerVoiceThreshold
            )
        }

        val vector = extractSpeakerVector(pcmAudio)
        if (vector == null) {
            Log.w(TAG, "Could not extract speaker vector from audio; insufficient voiced frames")
            return VoiceMatchResult(
                isMatch = false,
                distance = 1.0f,
                similarityPercent = 0,
                threshold = prefs.ownerVoiceThreshold,
                hasExtractedVector = false
            )
        }

        return verifySpeakerVector(vector, prefs)
    }
}
