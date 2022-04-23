package com.example.imageclassification

import android.content.res.AssetManager
import android.graphics.Bitmap
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp
import java.io.FileDescriptor
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class TFLiteHelper(assetManager: AssetManager, modelPath: String, labelPath: String) {
    private var imageSizeX = 0
    private var imageSizeY = 0

    private lateinit var inputImageBuffer: TensorImage
    private lateinit var labeledProbability: Array<FloatArray>

    private var interpreter: Interpreter
    private var labelList: List<String>

    private val imageMean = 0.0f
    private val imageStd = 255.0f

    data class Recognition(
        var id: Int = 0,
        var label: String = "",
        var probability: Float = 0.0f,
    ) {
        override fun toString(): String {
            return "Label: $label, Probability: $probability"
        }
    }

    init {
        val options = Interpreter.Options()
        options.setNumThreads(5)
//        options.setUseNNAPI(true)

        interpreter = Interpreter(loadModel(assetManager, modelPath), options)
        labelList = loadLabels(assetManager, labelPath)
    }

    private fun loadModel(assetManager: AssetManager, modelPath: String): MappedByteBuffer {
        val fileDescriptor = assetManager.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength

        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun loadLabels(assetManager: AssetManager, labelPath: String): List<String> {
        return assetManager.open(labelPath).bufferedReader().readLines()
    }

    private fun loadImage(bitmap: Bitmap): TensorImage {
        inputImageBuffer.load(bitmap)

        val cropSize = Math.min(bitmap.height, bitmap.width)

        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeWithCropOrPadOp(cropSize, cropSize))
            .add(ResizeOp(imageSizeY, imageSizeX, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
            .add(NormalizeOp(imageMean, imageStd))
            .build()
        return imageProcessor.process(inputImageBuffer)
    }

    fun classifyImage(bitmap: Bitmap): List<Recognition> {
        val imageTensorIndex = 0

        val imageDataShape = interpreter.getInputTensor(imageTensorIndex).shape()
        val imageDataType = interpreter.getInputTensor(imageTensorIndex).dataType()

        imageSizeY = imageDataShape[1]
        imageSizeX = imageDataShape[2]

        inputImageBuffer = TensorImage(imageDataType)
        inputImageBuffer = loadImage(bitmap)

        labeledProbability = Array(1) { FloatArray(labelList.size) }
        interpreter.run(inputImageBuffer.buffer, labeledProbability)

        return sortResult(labeledProbability)
    }

    private fun sortResult(labelProb: Array<FloatArray>):List<Recognition>{
        val recognition = ArrayList<Recognition>()
        for(i in labelList.indices){
            val confidence = labelProb[0][i]
            recognition.add(Recognition(i, labelList[i], confidence))
        }

        return recognition.sortedByDescending {
            it.probability
        }
    }
}