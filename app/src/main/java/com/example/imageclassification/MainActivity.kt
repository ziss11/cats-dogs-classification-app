package com.example.imageclassification

import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.annotation.Nullable
import androidx.appcompat.app.AppCompatActivity
import com.example.imageclassification.databinding.ActivityMainBinding
import java.io.IOException


class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding

    private lateinit var tfLite: TFLiteHelper
    private lateinit var bitmap: Bitmap
    private lateinit var imageUrl: Uri

    private val modelPath = "converted_model.tflite"
    private val labelsPath = "label.txt"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        val view = binding.root
        setContentView(view)

        tfLite = TFLiteHelper(assets, modelPath, labelsPath)

        binding.classifyBtn.setOnClickListener(classifyImage)
        binding.classifyImage.setOnClickListener(selectImageListener)
    }


    override fun onActivityResult(requestCode: Int, resultCode: Int, @Nullable data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == 12 && resultCode == RESULT_OK && data != null) {
            imageUrl = data.data!!
            try {
                bitmap = MediaStore.Images.Media.getBitmap(contentResolver, imageUrl)
                binding.classifyImage.setImageBitmap(bitmap)
            } catch (e: IOException) {
                e.printStackTrace()
            }
        }
    }

    var selectImageListener = View.OnClickListener {
        binding.result.text = ""

        val SELECT_TYPE = "image/*"
        val SELECT_PICTURE = "Select Picture"
        val intent = Intent()
        intent.type = SELECT_TYPE
        intent.action = Intent.ACTION_GET_CONTENT
        startActivityForResult(Intent.createChooser(intent, SELECT_PICTURE), 12)
    }

    val classifyImage = View.OnClickListener {
        val result = tfLite.classifyImage(bitmap)

        binding.result.text = "${result[0].label}: ${(result[0].probability*100).toInt()}%"
        Log.d("Result", "$result")
    }
}