/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
package com.example.recognition.tflite

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.RectF
import android.os.Trace
import android.util.Pair
import com.example.recognition.env.Logger
import com.google.firebase.firestore.FirebaseFirestore
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.*
import kotlin.collections.HashMap

import android.net.Uri
import com.google.android.gms.tasks.Continuation
import com.google.android.gms.tasks.OnCompleteListener
import com.google.android.gms.tasks.Task
import com.google.firebase.firestore.ktx.getField
import com.google.firebase.storage.FirebaseStorage
import com.google.firebase.storage.StorageTask
import com.google.firebase.storage.UploadTask
import java.io.*

import android.content.Context
import android.graphics.Color
import android.graphics.drawable.Drawable
import com.squareup.picasso.Picasso
import com.squareup.picasso.Target
import java.security.KeyRep
import kotlin.collections.ArrayList
import kotlin.coroutines.coroutineContext


class TFLiteObjectDetectionAPIModel private constructor() : SimilarityClassifier {
    private var isModelQuantized = false

    // Config values.
    private var inputSize = 0

    // Pre-allocated buffers.
    private val labels = Vector<String>()
    lateinit var intValues: IntArray

    // outputLocations: array of shape [Batchsize, NUM_DETECTIONS,4]
    // Algılanan kutuların yerleri
    lateinit var outputLocations: Array<Array<FloatArray>>

    // outputClasses: array of shape [Batchsize, NUM_DETECTIONS]
    // algılanan kutuların sınıflarını içerir
    lateinit var outputClasses: Array<FloatArray>

    // outputScores: array of shape [Batchsize, NUM_DETECTIONS]
    // algılanan kutuların puanlarını içerir
    lateinit var outputScores: Array<FloatArray>

    // numDetections: array of shape [Batchsize]
    // algılanan kutuların sayısını içerir
    lateinit var numDetections: FloatArray
    lateinit var embeedings: Array<FloatArray>
    private var imgData: ByteBuffer? = null
    private var tfLite: Interpreter? = null

    // Face Mask Detector Output
    lateinit var output: Array<FloatArray>

    private val registered = HashMap<String?, SimilarityClassifier.Recognition?>()

    override fun register(name: String?, rec: SimilarityClassifier.Recognition?,new : Boolean?) {

        if (new == true){
            val fileReference = FirebaseStorage.getInstance().reference.child("Pics")
                .child(System.currentTimeMillis().toString()+".jpg")

            val baos1 = ByteArrayOutputStream()
            rec?.crop?.compress(Bitmap.CompressFormat.JPEG, 100, baos1)
            val data1 = baos1.toByteArray()
            val uploadTask1: StorageTask<*>
            uploadTask1 = fileReference.putBytes(data1)
            uploadTask1.continueWithTask(Continuation<UploadTask.TaskSnapshot, Task<Uri>>{ task->
                if (!task.isSuccessful){
                    task.exception?.let {
                        throw it
                    }
                }
                return@Continuation fileReference.downloadUrl
            }).addOnCompleteListener ( OnCompleteListener<Uri>{task ->
                if (task.isSuccessful){
                    val downloadUrl = task.result
                    val imageUrl = downloadUrl.toString()
                    val userHashMap = HashMap<String,Any>()
                    userHashMap.put("name", name!!)
                    userHashMap.put("image",imageUrl)
                    userHashMap.put("distance", rec?.distance!!)
                    userHashMap.put("id",rec.id!!)

                    for (number in 0 .. 10){
                        userHashMap.put("$number",(rec!!.extra as Array<FloatArray>)[0][number])
                    }

                    userHashMap.put("extra",rec.extra.toString())
                    FirebaseFirestore.getInstance().collection("users")
                        .document().set(userHashMap).addOnSuccessListener { task1-> }
                }
            })
        }
        registered[name] = rec

    }

    override fun getPeople(context : Context){
        FirebaseFirestore.getInstance().collection("users")
            .addSnapshotListener { value, error ->
                if (value != null){
                    for (document in value.documents){

                        val name1 = document.getField<String>("name")
                        val image = document.getField<String>("image")
                        val distance = document.getField<Float>("distance")
                        val id = document.getField<String>("id")
                        val extra = document.getField<String>("extra")


                        Picasso.get().load(image).into(object : Target {
                            override fun onBitmapLoaded(
                                bitmap: Bitmap?,
                                from: Picasso.LoadedFrom?
                            ) {
                                var color = 0
                                if (id == "0"){
                                    color = Color.GREEN
                                }else{
                                    color = Color.RED
                                }

                                val result = SimilarityClassifier.Recognition(
                                    id,
                                    name1,
                                    distance,
                                    RectF())
                                result.crop = bitmap
                                result.color = color

                                var ex : Array<FloatArray>? = null

                                for(number in 0 .. 10){
                                    ex!![0][number] = document.getField<Float>("$number")!!.toFloat()
                                }

                                result.extra = ex
                                registered[name1] = result
                            }

                            override fun onBitmapFailed(
                                e: java.lang.Exception?,
                                errorDrawable: Drawable?
                            ) {}

                            override fun onPrepareLoad(placeHolderDrawable: Drawable?) {}

                        })

                        /*Glide.with(context).asBitmap().load(image).into(object : CustomTarget<Bitmap>(){
                            override fun onResourceReady(
                                resource: Bitmap,
                                transition: Transition<in Bitmap>?
                            ) {
                                var color = 0
                                if (id == "0"){
                                    color = Color.GREEN
                                }else{
                                    color = Color.RED
                                }

                                val result = SimilarityClassifier.Recognition(id,name1,distance,RectF())
                                result.crop = resource
                                result.color = color
                                result.extra = extra as Array<FloatArray>
                                registered[name1] = result
                            }

                            override fun onLoadCleared(placeholder: Drawable?) {}
                        })*/
                    }
                }
            }
    }

    private fun findNearest(emb: FloatArray): Pair<String?, Float>? {
        var ret: Pair<String?, Float>? = null
        for ((name, value) in registered) {
            val knownEmb = (value!!.extra as Array<FloatArray>?)!![0]
            var distance = 0f
            for (i in emb.indices) {
                val diff = emb[i] - knownEmb[i]
                distance += diff * diff
            }
            distance = Math.sqrt(distance.toDouble()).toFloat()
            if (ret == null || distance < ret.second) {
                ret = Pair(name, distance)
            }
        }
        return ret
    }

    override fun recognizeImage(
        bitmap: Bitmap?,
        storeExtra: Boolean
    ): List<SimilarityClassifier.Recognition?> {
        // Log this method so that it can be analyzed with systrace.
        Trace.beginSection("recognizeImage")
        Trace.beginSection("preprocessBitmap")

        /* Sağlanan parametrelere göre 0-255 int'den
        normalleştirilmiş kayan noktalı görüntü verilerini önceden işleyin.*/

        intValues = IntArray(bitmap!!.width * bitmap.height)

        bitmap!!.getPixels(
            intValues,
            0,
            bitmap.width,
            0,
            0,
            bitmap.width,
            bitmap.height
        )

        if (storeExtra == false){
            imgData!!.rewind()
            for (i in 0 until inputSize) {
                for (j in 0 until inputSize) {
                    val pixelValue = intValues[i * inputSize + j]
                    if (isModelQuantized) {
                        // Quantized model
                        imgData!!.put((pixelValue shr 16 and 0xFF).toByte())
                        imgData!!.put((pixelValue shr 8 and 0xFF).toByte())
                        imgData!!.put((pixelValue and 0xFF).toByte())
                    } else { // Float model
                        imgData!!.putFloat(((pixelValue shr 16 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                        imgData!!.putFloat(((pixelValue shr 8 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                        imgData!!.putFloat(((pixelValue and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                    }
                }
            }
            Trace.endSection() // Ön işlem bitmap
        }


        // Giriş verilerini TensorFlow'a kopyalanır.
        Trace.beginSection("feed")
        val inputArray = arrayOf<Any?>(imgData)
        Trace.endSection()

        // Burada outputMap, Yüz Maskesi dedektörüne uyacak şekilde değiştirilir
        val outputMap: MutableMap<Int, Any> = HashMap()
        embeedings = Array(1) { FloatArray(OUTPUT_SIZE) }
        outputMap[0] = embeedings

        // Çıkarım Çağrısı
        Trace.beginSection("run")

        //tfLite.runForMultipleInputsOutputs(inputArray, outputMapBack);
        tfLite!!.runForMultipleInputsOutputs(inputArray, outputMap)
        Trace.endSection()

        var distance = Float.MAX_VALUE
        val id = "0"
        var label: String? = "?"
        if (registered.size > 0) {
            val nearest = findNearest(embeedings[0])
            if (nearest != null) {
                val name = nearest.first
                label = name
                distance = nearest.second
            }
        }
        val numDetectionsOutput = 1
        val recognitions = ArrayList<SimilarityClassifier.Recognition?>(numDetectionsOutput)
        val rec = SimilarityClassifier.Recognition(
            id,
            label,
            distance,
            RectF()
        )
        recognitions.add(rec)
        if (storeExtra) {
            rec.extra = embeedings
        }
        Trace.endSection()
        return recognitions
    }

    override fun enableStatLogging(logStats: Boolean) {}
    override val statString: String
        get() = ""

    override fun close() {}
    override fun setNumThreads(num_threads: Int) {
        if (tfLite != null) tfLite!!.setNumThreads(num_threads)
    }

    override fun setUseNNAPI(isChecked: Boolean) {
        if (tfLite != null) tfLite!!.setUseNNAPI(isChecked)
    }

    companion object {
        private val LOGGER =
            Logger()

        //private static final int OUTPUT_SIZE = 512;
        private const val OUTPUT_SIZE = 192

        // Only return this many results.
        private const val NUM_DETECTIONS = 1

        // Float model
        private const val IMAGE_MEAN = 128.0f
        private const val IMAGE_STD = 128.0f

        // Number of threads in the java app
        private const val NUM_THREADS = 4

        /** Memory-map the model file in Assets.  */
        @Throws(IOException::class)
        private fun loadModelFile(
            assets: AssetManager,
            modelFilename: String
        ): MappedByteBuffer {
            val fileDescriptor = assets.openFd(modelFilename)
            val inputStream =
                FileInputStream(fileDescriptor.fileDescriptor)
            val fileChannel = inputStream.channel
            val startOffset = fileDescriptor.startOffset
            val declaredLength = fileDescriptor.declaredLength
            return fileChannel.map(
                FileChannel.MapMode.READ_ONLY,
                startOffset,
                declaredLength
            )
        }

        /**
         * Initializes a native TensorFlow session for classifying images.
         *
         * @param assetManager The asset manager to be used to load assets.
         * @param modelFilename The filepath of the model GraphDef protocol buffer.
         * @param labelFilename The filepath of label file for classes.
         * @param inputSize The size of image input
         * @param isQuantized Boolean representing model is quantized or not
         */
        @Throws(IOException::class)
        fun create(
            assetManager: AssetManager,
            modelFilename: String,
            labelFilename: String,
            inputSize: Int,
            isQuantized: Boolean
        ): SimilarityClassifier {
            var d = TFLiteObjectDetectionAPIModel()
            val actualFilename = labelFilename.split("file:///android_asset/")[1]
            val labelsInput = assetManager.open(actualFilename)
            val br = BufferedReader(InputStreamReader(labelsInput))

            var line: String?
            while (br.readLine().also { din ->
                    line = din } != null) {
                LOGGER.w(line!!)
                d.labels.add(line)
            }
            br.close()
            d.inputSize = inputSize
            try {
                d.tfLite = Interpreter(
                    loadModelFile(
                        assetManager,
                        modelFilename
                    )
                )
            } catch (e: Exception) {
                throw RuntimeException(e)
            }
            d.isModelQuantized = isQuantized
            // Pre-allocate buffers.
            val numBytesPerChannel: Int = if (isQuantized) {
                1 // Quantized
            } else {
                4 // Floating point
            }
            d.imgData = ByteBuffer.allocateDirect(1 * d.inputSize * d.inputSize * 3 * numBytesPerChannel)
            d.imgData!!.order(ByteOrder.nativeOrder())
            d.intValues = IntArray(d.inputSize * d.inputSize)
            d.tfLite!!.setNumThreads(NUM_THREADS)
            d.outputLocations = Array(
                1
            ) {
                Array(NUM_DETECTIONS) { FloatArray(4) }
            }
            d.outputClasses = Array(
                1
            ) { FloatArray(NUM_DETECTIONS) }
            d.outputScores = Array(
                1
            ) { FloatArray(NUM_DETECTIONS) }
            d.numDetections = FloatArray(1)
            return d
        }
    }
}