package br.com.cpzao.mobile

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ImageDecoder
import android.graphics.Paint
import android.graphics.RectF
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.Gravity
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.ByteOrder
import java.nio.ByteBuffer
import kotlin.math.max
import kotlin.math.min

class MainActivity : ComponentActivity() {
    private val labels by lazy { assets.open("labels.txt").bufferedReader().useLines { it.toList() } }
    private val threshold = 0.1f
    private val ortEnvironment by lazy { OrtEnvironment.getEnvironment() }
    private lateinit var ortSession: OrtSession
    private lateinit var preview: ImageView
    private lateinit var status: TextView

    private val picker = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        uri?.let {
            status.text = "Processando imagem…"
            try {
                runDetection(loadBitmap(it))
            } catch (error: Exception) {
                Log.e("CheckpointVisao", "Falha ao processar imagem selecionada", error)
                status.text = "Não foi possível analisar esta imagem: ${error.localizedMessage ?: error.javaClass.simpleName}"
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        ortSession = ortEnvironment.createSession(loadModel(), OrtSession.SessionOptions().apply { setIntraOpNumThreads(4) })
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(24, 32, 24, 24)
            gravity = Gravity.CENTER_HORIZONTAL
        }
        val title = TextView(this).apply { text = "Checkpoint Visão"; textSize = 22f; setTextColor(Color.BLACK) }
        val choose = Button(this).apply { text = "Selecionar imagem"; setOnClickListener { picker.launch("image/*") } }
        preview = ImageView(this).apply { adjustViewBounds = true; scaleType = ImageView.ScaleType.FIT_CENTER }
        status = TextView(this).apply { text = "Escolha uma foto para detectar buracos, obstáculos e piso tátil."; textSize = 16f; setTextColor(Color.DKGRAY) }
        root.addView(title)
        root.addView(choose)
        root.addView(preview, LinearLayout.LayoutParams(-1, 0, 1f))
        root.addView(status)
        setContentView(root)
    }

    private fun loadModel(): ByteArray = assets.open("best.onnx").use { it.readBytes() }

    private fun loadBitmap(uri: Uri): Bitmap = ImageDecoder.decodeBitmap(ImageDecoder.createSource(contentResolver, uri)) { decoder, _, _ ->
        decoder.isMutableRequired = false
    }.copy(Bitmap.Config.ARGB_8888, false)

    private fun runDetection(bitmap: Bitmap) {
        val scale = min(640f / bitmap.width, 640f / bitmap.height)
        val resizedWidth = (bitmap.width * scale).toInt()
        val resizedHeight = (bitmap.height * scale).toInt()
        val padLeft = (640 - resizedWidth) / 2
        val padTop = (640 - resizedHeight) / 2
        val letterboxed = Bitmap.createBitmap(640, 640, Bitmap.Config.ARGB_8888)
        val inputCanvas = Canvas(letterboxed)
        inputCanvas.drawColor(Color.rgb(114, 114, 114))
        inputCanvas.drawBitmap(
            Bitmap.createScaledBitmap(bitmap, resizedWidth, resizedHeight, true),
            padLeft.toFloat(), padTop.toFloat(), null
        )
        val input = ByteBuffer.allocateDirect(1 * 3 * 640 * 640 * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
        for (channel in 0..2) for (y in 0 until 640) for (x in 0 until 640) {
            val pixel = letterboxed.getPixel(x, y)
            val value = when (channel) {
                0 -> Color.red(pixel)
                1 -> Color.green(pixel)
                else -> Color.blue(pixel)
            }
            input.put(value / 255f)
        }
        input.rewind()
        val inputName = ortSession.inputNames.iterator().next()
        val channels = OnnxTensor.createTensor(ortEnvironment, input, longArrayOf(1, 3, 640, 640)).use { tensor ->
            ortSession.run(mapOf(inputName to tensor)).use { result ->
                val value = result[0].value
                when (value) {
                    is Array<*> -> {
                        val firstBatch = value.firstOrNull() as? Array<*>
                            ?: error("Formato de saída ONNX sem dimensão de classes")
                        Array(firstBatch.size) { classIndex ->
                            firstBatch[classIndex] as? FloatArray
                                ?: error("Formato inesperado na classe $classIndex da saída ONNX")
                        }
                    }
                    else -> error("Formato de saída ONNX inesperado: ${value?.javaClass?.name}")
                }
            }
        }
        val candidates = mutableListOf<Detection>()
        var maxScore = 0f
        var maxClass = 0
        if (channels.size != 7 || channels.any { it.size != 8400 }) {
            error("Formato ONNX inesperado: ${channels.size} canais x ${channels.firstOrNull()?.size ?: 0} candidatos")
        }
        // Ultralytics exports [1, 4 + classes, candidates], here [1, 7, 8400].
        for (i in 0 until channels[0].size) {
            var classId = 0
            var score = channels[4][i]
            for (c in 1..2) if (channels[4 + c][i] > score) { score = channels[4 + c][i]; classId = c }
            if (score > maxScore) { maxScore = score; maxClass = classId }
            if (score >= threshold) {
                val cx = channels[0][i]
                val cy = channels[1][i]
                val w = channels[2][i]
                val h = channels[3][i]
                val x1 = ((cx - w / 2 - padLeft) / scale).coerceIn(0f, bitmap.width.toFloat())
                val y1 = ((cy - h / 2 - padTop) / scale).coerceIn(0f, bitmap.height.toFloat())
                val x2 = ((cx + w / 2 - padLeft) / scale).coerceIn(0f, bitmap.width.toFloat())
                val y2 = ((cy + h / 2 - padTop) / scale).coerceIn(0f, bitmap.height.toFloat())
                candidates.add(Detection(classId, score, x1, y1, x2, y2))
            }
        }
        val kept = nms(candidates)
        preview.setImageBitmap(drawDetections(bitmap, kept))
        status.text = if (kept.isEmpty()) {
            "Nenhum objeto acima de 0,10. Maior score: ${labels[maxClass]} (${(maxScore * 100).toInt()}%)."
        } else kept.joinToString("\n") { "${labels[it.classId]} • ${(it.score * 100).toInt()}%" }
    }

    private fun drawDetections(bitmap: Bitmap, detections: List<Detection>): Bitmap {
        if (detections.isEmpty()) return bitmap
        val result = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(result)
        val stroke = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.GREEN
            style = Paint.Style.STROKE
            strokeWidth = max(3f, bitmap.width / 240f)
        }
        val text = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.GREEN
            textSize = max(28f, bitmap.width / 24f)
            style = Paint.Style.FILL
        }
        for (detection in detections) {
            canvas.drawRect(RectF(detection.x1, detection.y1, detection.x2, detection.y2), stroke)
            canvas.drawText("${labels[detection.classId]} ${(detection.score * 100).toInt()}%", detection.x1, max(text.textSize, detection.y1), text)
        }
        return result
    }

    private fun nms(items: List<Detection>): List<Detection> {
        val selected = mutableListOf<Detection>()
        for (classId in labels.indices) {
            val pending = items.filter { it.classId == classId }.sortedByDescending { it.score }.toMutableList()
            while (pending.isNotEmpty()) {
                val best = pending.removeAt(0); selected.add(best)
                pending.removeAll { intersectionOverUnion(best, it) > 0.45f }
            }
        }
        return selected.sortedByDescending { it.score }.take(50)
    }

    private fun intersectionOverUnion(a: Detection, b: Detection): Float {
        val iw = max(0f, min(a.x2, b.x2) - max(a.x1, b.x1)); val ih = max(0f, min(a.y2, b.y2) - max(a.y1, b.y1))
        val inter = iw * ih; val areaA = max(0f, a.x2-a.x1) * max(0f, a.y2-a.y1); val areaB = max(0f, b.x2-b.x1) * max(0f, b.y2-b.y1)
        return inter / (areaA + areaB - inter + 1e-6f)
    }

    override fun onDestroy() { if (::ortSession.isInitialized) ortSession.close(); super.onDestroy() }
    private data class Detection(val classId: Int, val score: Float, val x1: Float, val y1: Float, val x2: Float, val y2: Float)
}
