package com.example.recognition

import android.app.AlertDialog
import android.content.Context
import android.content.DialogInterface
import android.graphics.*
import android.graphics.drawable.Drawable
import android.hardware.camera2.CameraCharacteristics
import android.media.ImageReader.OnImageAvailableListener
import android.os.Bundle
import android.os.SystemClock
import android.os.Trace
import android.util.AttributeSet
import android.util.Size
import android.util.TypedValue
import android.view.View
import android.widget.EditText
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.core.graphics.scale
import com.bumptech.glide.Glide
import com.bumptech.glide.request.target.CustomTarget
import com.bumptech.glide.request.transition.Transition
import com.example.recognition.customview.OverlayView
import com.example.recognition.env.BorderedText
import com.example.recognition.env.ImageUtils
import com.example.recognition.env.Logger
import com.example.recognition.tflite.SimilarityClassifier
import com.example.recognition.tflite.TFLiteObjectDetectionAPIModel
import com.example.recognition.tracking.MultiBoxTracker
import com.google.android.gms.tasks.OnSuccessListener
import com.google.android.material.floatingactionbutton.FloatingActionButton
import com.google.firebase.firestore.FirebaseFirestore
import com.google.firebase.firestore.ktx.getField
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.squareup.picasso.Picasso
import com.squareup.picasso.Target
import java.nio.ByteBuffer
import java.util.*
import android.graphics.BitmapFactory

import android.graphics.Bitmap
import java.io.IOException
import java.net.URL
import java.nio.ByteOrder


open class DetectorActivity : CameraActivity(), OnImageAvailableListener {

    var trackingOverlay: OverlayView? = null
    private var sensorOrientation: Int? = null

    lateinit var detector: SimilarityClassifier

    lateinit var embeedings: Array<FloatArray>

    private val IMAGE_MEAN = 128.0f
    private val IMAGE_STD = 128.0f

    private var lastProcessingTimeMs: Long = 0
    private var rgbFrameBitmap: Bitmap? = null
    private var croppedBitmap: Bitmap? = null
    lateinit var cropCopyBitmap: Bitmap
    private var computingDetection = false

    private var addPending = false

    private var intValues: IntArray? = null
    private var imgData: ByteBuffer? = null

    //private boolean adding = false;
    private var timestamp: Long = 0
    private var frameToCropTransform: Matrix? = null
    private var cropToFrameTransform: Matrix? = null

    //private Matrix cropToPortraitTransform;
    private var tracker: MultiBoxTracker? = null
    private var borderedText: BorderedText? = null

    // Face detector
    private var faceDetector: FaceDetector? = null

    // Ön izleme resmi portre şeklimde seçilir
    private var portraitBmp: Bitmap? = null

    // here the face is cropped and drawn
    private var faceBmp: Bitmap? = null
    private var fas : Bitmap? = null
    lateinit var fabAdd: FloatingActionButton

    //private HashMap<String, Classifier.Recognition> knownFaces = new HashMap<>();
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        //Kişi Ekle
        fabAdd = findViewById(R.id.fab_add)
        fabAdd.setOnClickListener {
            onAddClick()
        }

        // Birden fazla yüzün tespiti
        val options = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .setContourMode(FaceDetectorOptions.LANDMARK_MODE_NONE)
            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_NONE)
            .build()
        val detector = FaceDetection.getClient(options)
        faceDetector = detector
    }

    private fun onAddClick() {
        addPending = true
    }

    public override fun onPreviewSizeChosen(size: Size?, rotation: Int) {

        val textSizePx = TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP,
            TEXT_SIZE_DIP,
            resources.displayMetrics
        )
        borderedText = BorderedText(textSizePx)
        borderedText!!.setTypeface(Typeface.MONOSPACE)
        tracker = MultiBoxTracker(this)
        try {
            detector = TFLiteObjectDetectionAPIModel.create(
                assets,
                TF_OD_API_MODEL_FILE,
                TF_OD_API_LABELS_FILE,
                TF_OD_API_INPUT_SIZE,
                TF_OD_API_IS_QUANTIZED
            )
            //cropSize = TF_OD_API_INPUT_SIZE;
        } catch (e: Exception) {
            e.printStackTrace()
            LOGGER.e(e, "Exception initializing classifier!")
            val toast = Toast.makeText(
                applicationContext, "Classifier could not be initialized", Toast.LENGTH_SHORT
            )
            toast.show()
            finish()
        }
        previewWidth = size!!.width
        previewHeight = size.height
        sensorOrientation = rotation - screenOrientation
        LOGGER.i(
            "Camera orientation relative to screen canvas: %d",
            sensorOrientation
        )
        LOGGER.i(
            "Initializing at size %dx%d",
            previewWidth,
            previewHeight
        )
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888)
        val targetW: Int
        val targetH: Int
        if (sensorOrientation == 90 || sensorOrientation == 270) {
            targetH = previewWidth
            targetW = previewHeight
        } else {
            targetW = previewWidth
            targetH = previewHeight
        }
        val cropW = (targetW / 2.0).toInt()
        val cropH = (targetH / 2.0).toInt()
        croppedBitmap = Bitmap.createBitmap(cropW, cropH, Bitmap.Config.ARGB_8888)
        portraitBmp = Bitmap.createBitmap(targetW, targetH, Bitmap.Config.ARGB_8888)
        faceBmp = Bitmap.createBitmap(
            TF_OD_API_INPUT_SIZE,
            TF_OD_API_INPUT_SIZE,
            Bitmap.Config.ARGB_8888
        )

        frameToCropTransform =
            ImageUtils.getTransformationMatrix(
                previewWidth, previewHeight,
                cropW, cropH,
                sensorOrientation!!, MAINTAIN_ASPECT
            )

        cropToFrameTransform = Matrix()
        frameToCropTransform!!.invert(cropToFrameTransform)
        val frameToPortraitTransform =
            ImageUtils.getTransformationMatrix(
                previewWidth, previewHeight,
                targetW, targetH,
                sensorOrientation!!, MAINTAIN_ASPECT
            )
        trackingOverlay = findViewById<View>(R.id.tracking_overlay) as OverlayView
        trackingOverlay!!.addCallback(
            object : OverlayView.DrawCallback{
                override fun drawCallback(canvas: Canvas?) {
                    tracker!!.draw(canvas!!)
                    if (isDebug) {
                        tracker!!.drawDebug(canvas)
                    }
                }
            })
        tracker!!.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation!!)
    }

    override fun processImage() {
        ++timestamp
        val currTimestamp = timestamp
        trackingOverlay!!.postInvalidate()

        // No mutex needed as this method is not reentrant.
        if (computingDetection) {
            readyForNextImage()
            return
        }
        computingDetection = true
        LOGGER.i("Preparing image $currTimestamp for detection in bg thread.")
        rgbFrameBitmap!!.setPixels(
            getRgbBytes(),
            0,
            previewWidth,
            0,
            0,
            previewWidth,
            previewHeight
        )
        readyForNextImage()
        val canvas = Canvas(croppedBitmap!!)
        canvas.drawBitmap(rgbFrameBitmap!!, frameToCropTransform!!, null)
        // For examining the actual TF input.
        if (SAVE_PREVIEW_BITMAP) {
            ImageUtils.saveBitmap(croppedBitmap!!)
        }
        val image = InputImage.fromBitmap(croppedBitmap!!, 0)
        faceDetector!!
            .process(image)
            .addOnSuccessListener(OnSuccessListener { faces ->
                if (faces.size == 0) {
                    updateResults(currTimestamp, LinkedList())
                    return@OnSuccessListener
                }
                runInBackground(
                    Runnable {
                        onFacesDetected(currTimestamp, faces, addPending)
                        addPending = false
                    })
            })
    }

    override val layoutId: Int
        get() = R.layout.tfe_od_camera_connection_fragment_tracking
    override val desiredPreviewFrameSize: Size?
        get() = Size(640, 480)

    private enum class DetectorMode {
        TF_OD_API
    }

    override fun setUseNNAPI(isChecked: Boolean) {
        runInBackground(Runnable { detector.setUseNNAPI(isChecked) })
    }

    override fun setNumThreads(numThreads: Int) {
        runInBackground(Runnable { detector.setNumThreads(numThreads) })
    }

    // Yüz İşlemleri
    private fun createTransform(srcWidth: Int, srcHeight: Int, dstWidth: Int, dstHeight: Int, applyRotation: Int): Matrix {
        val matrix = Matrix()
        if (applyRotation != 0) {
            if (applyRotation % 90 != 0) {
                LOGGER.w("Rotation of %d % 90 != 0", applyRotation)
            }

            // Görüntünün merkezini orjinde olacak şekilde ayarlar
            matrix.postTranslate(-srcWidth / 2.0f, -srcHeight / 2.0f)

            // Orjin etrafında döndürün
            matrix.postRotate(applyRotation.toFloat())
        }

        if (applyRotation != 0) {

            // GÖrüntü merkezini eski haline getirir
            matrix.postTranslate(dstWidth / 2.0f, dstHeight / 2.0f)
        }
        return matrix
    }

    //Resim ekleme Diyaloğunu Göster
    private fun showAddFaceDialog(rec: SimilarityClassifier.Recognition) {
        val builder = AlertDialog.Builder(this)
        val inflater = layoutInflater
        val dialogLayout = inflater.inflate(R.layout.image_edit_dialog, null)
        val ivFace = dialogLayout.findViewById<ImageView>(R.id.dlg_image)
        val tvTitle = dialogLayout.findViewById<TextView>(R.id.dlg_title)
        val etName = dialogLayout.findViewById<EditText>(R.id.dlg_input)
        tvTitle.text = "Yüz Ekle"
        ivFace.setImageBitmap(rec.crop)
        etName.hint = "Input name"
        builder.setPositiveButton("OK", DialogInterface.OnClickListener { dlg, i ->
            val name = etName.text.toString()
            if (name.isEmpty()) {
                return@OnClickListener
            }
            detector.register(name, rec,true)
            dlg.dismiss()
        })
        builder.setView(dialogLayout)
        builder.show()
    }


    private fun updateResults(currTimestamp: Long, mappedRecognitions: List<SimilarityClassifier.Recognition>) {

        tracker!!.trackResults(mappedRecognitions, currTimestamp)
        trackingOverlay!!.postInvalidate()
        computingDetection = false

        if (mappedRecognitions.isNotEmpty()) {
            val rec = mappedRecognitions[0]
            if (rec.extra != null) {
                showAddFaceDialog(rec)
            }
        }
    }


    private fun onFacesDetected(currTimestamp: Long, faces: List<Face>, add: Boolean) {

        addUser()

        cropCopyBitmap = Bitmap.createBitmap(croppedBitmap!!)
        val canvas = Canvas(cropCopyBitmap)
        val paint = Paint()
        paint.color = Color.RED
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = 2.0f
        var minimumConfidence =
            MINIMUM_CONFIDENCE_TF_OD_API
        minimumConfidence = when (MODE) {
            DetectorMode.TF_OD_API -> MINIMUM_CONFIDENCE_TF_OD_API
        }
        val mappedRecognitions: MutableList<SimilarityClassifier.Recognition> = LinkedList()


        //final List<Classifier.Recognition> results = new ArrayList<>();

        // Note this can be done only once
        val sourceW = rgbFrameBitmap!!.width
        val sourceH = rgbFrameBitmap!!.height
        val targetW = portraitBmp!!.width
        val targetH = portraitBmp!!.height
        val transform = createTransform(
            sourceW,
            sourceH,
            targetW,
            targetH,
            sensorOrientation!!
        )
        val cv = Canvas(portraitBmp!!)

        // Orjinal görüntüyü portre modunda çizer
        cv.drawBitmap(rgbFrameBitmap!!, transform, null)
        val cvFace = Canvas(faceBmp!!)
        val saved = false
        for (face in faces) {
            LOGGER.i("FACE $face")
            LOGGER.i("Running detection on face $currTimestamp")
            //results = detector.recognizeImage(croppedBitmap);
            val boundingBox = RectF(face.boundingBox)

            //final boolean goodConfidence = result.getConfidence() >= minimumConfidence;
            val goodConfidence = true //face.get;
            if (boundingBox != null && goodConfidence) {

                // Kırpma koordinatlarını orjinaline eşler
                cropToFrameTransform!!.mapRect(boundingBox)

                // Orjinal koordinatları dikey koordinatlar ile eşler
                val faceBB = RectF(boundingBox)
                transform.mapRect(faceBB)

                // Portreyi orjine çevirir ve boyutları ölçekler
                //cv.drawRect(faceBB, paint)
                val sx = TF_OD_API_INPUT_SIZE.toFloat() / faceBB.width()
                val sy = TF_OD_API_INPUT_SIZE.toFloat() / faceBB.height()
                val matrix = Matrix()
                matrix.postTranslate(-faceBB.left, -faceBB.top)
                matrix.postScale(sx, sy)
                cvFace.drawBitmap(portraitBmp!!, matrix, null)


                //canvas.drawRect(faceBB, paint);
                var label = ""
                var confidence = -1f
                var color = Color.BLUE
                var extra: Any? = null
                var crop: Bitmap? = null
                if (add) {
                    crop = Bitmap.createBitmap(
                        portraitBmp!!,
                        faceBB.left.toInt(),
                        faceBB.top.toInt(),
                        faceBB.width().toInt(),
                        faceBB.height().toInt()
                    )
                }
                val startTime = SystemClock.uptimeMillis()
                val resultsAux = detector.recognizeImage(faceBmp, add)
                lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime
                if (resultsAux!!.size > 0) {

                    val result = resultsAux[0]
                    extra = result!!.extra

                    val conf = result.distance!!
                    if (conf < 1.0f) {
                        confidence = conf
                        label = result.name!!
                        color = if (result.id == "0") {
                            Color.GREEN
                        } else {
                            Color.RED
                        }
                    }
                }
                if (cameraFacing == CameraCharacteristics.LENS_FACING_FRONT) {

                    // camera is frontal so the image is flipped horizontally
                    // flips horizontally
                    val flip = Matrix()
                    if (sensorOrientation == 90 || sensorOrientation == 270) {
                        flip.postScale(1f, -1f, previewWidth / 2.0f, previewHeight / 2.0f)
                    } else {
                        flip.postScale(-1f, 1f, previewWidth / 2.0f, previewHeight / 2.0f)
                    }
                    //flip.postScale(1, -1, targetW / 2.0f, targetH / 2.0f);
                    flip.mapRect(boundingBox)
                }

                //Result here
                val result = SimilarityClassifier.Recognition(
                    "0", label, confidence, boundingBox
                )

                result.color = color
                result.setLocation(boundingBox)
                result.extra = extra
                result.crop = crop
                mappedRecognitions.add(result)
            }
        }

        updateResults(currTimestamp, mappedRecognitions)
    }


    private fun addUser(){
        FirebaseFirestore.getInstance().collection("users")
            .addSnapshotListener { value, error ->
                if (value != null){
                    for (document in value.documents){
                        val image = document.getField<String>("image")
                        val name = document.getField<String>("name")
                        val distance = document.getField<Float>("distance")
                        val id = document.getField<String>("id")

                        Thread {
                            try {
                                val url = URL(image)
                                val bitmap = BitmapFactory.decodeStream(url.openConnection().getInputStream())

                                var color = Color.BLUE
                                var confidence = -1f
                                var crop: Bitmap? = null
                                var extra: Any? = null

                                val intValues = IntArray(TF_OD_API_INPUT_SIZE * TF_OD_API_INPUT_SIZE)
                                bitmap.setPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

                                extra =arrayOf<Any?>(convertBitmapToByteBuffer(bitmap))

                                /*val resultsAux = detector.recognizeImage(bitmap, true)
                                if (resultsAux!!.size > 0) {
                                    val result = resultsAux[0]
                                    extra = result!!.extra
                                    if (distance!! < 1.0f) {
                                        color = if (id == "0") {
                                            Color.GREEN
                                        } else {
                                            Color.RED
                                        }
                                    }
                                }*/

                                val result = SimilarityClassifier.Recognition(
                                    "0", name, distance, RectF()
                                )

                                result.color = color
                                result.setLocation(RectF())
                                result.extra = extra
                                result.crop = bitmap
                                detector.register(name, result, false)

                            } catch (e: IOException) {
                                System.out.println(e)
                            }
                        }.start()

                    }
                }
            }
    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer? {
        val byteBuffer = ByteBuffer.allocateDirect(4 * 1 * TF_OD_API_INPUT_SIZE * TF_OD_API_INPUT_SIZE * 3)
        byteBuffer.order(ByteOrder.nativeOrder())
        val intValues = IntArray(TF_OD_API_INPUT_SIZE * TF_OD_API_INPUT_SIZE)
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        var pixel = 0
        for (i in 0 until 192) {
            for (j in 0 until 192) {
                val `val` = intValues[pixel++]
                byteBuffer.putFloat(((`val` shr 16 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                byteBuffer.putFloat(((`val` shr 8 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                byteBuffer.putFloat(((`val` and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
            }
        }


        return byteBuffer
    }

    companion object {
        private val LOGGER = Logger()

        // FaceNet
        //  private static final int TF_OD_API_INPUT_SIZE = 160;
        //  private static final boolean TF_OD_API_IS_QUANTIZED = false;
        //  private static final String TF_OD_API_MODEL_FILE = "facenet.tflite";
        //  //private static final String TF_OD_API_MODEL_FILE = "facenet_hiroki.tflite";
        // MobileFaceNet
        private const val TF_OD_API_INPUT_SIZE = 112
        private const val TF_OD_API_IS_QUANTIZED = false
        private const val TF_OD_API_MODEL_FILE = "mobile_face_net.tflite"
        private const val TF_OD_API_LABELS_FILE = "file:///android_asset/labelmap.txt"
        private val MODE = DetectorMode.TF_OD_API

        // Minimum detection confidence to track a detection.
        private const val MINIMUM_CONFIDENCE_TF_OD_API = 0.5f
        private const val MAINTAIN_ASPECT = false

        //private static final int CROP_SIZE = 320;
        //private static final Size CROP_SIZE = new Size(320, 320);
        private const val SAVE_PREVIEW_BITMAP = false
        private const val TEXT_SIZE_DIP = 10f
    }
}