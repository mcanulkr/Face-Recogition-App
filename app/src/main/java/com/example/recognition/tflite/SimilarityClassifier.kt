package com.example.recognition.tflite

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF

// Farklı tanıma motorlarıyla etkileşim kurmak için genel arayüz.

interface SimilarityClassifier {
    fun register(name: String?, recognition: Recognition?,new : Boolean?)
    fun recognizeImage(bitmap: Bitmap?, getExtra: Boolean): List<Recognition?>?

    fun enableStatLogging(debug: Boolean)
    val statString: String?
    fun close()
    fun setNumThreads(num_threads: Int)
    fun setUseNNAPI(isChecked: Boolean)
    fun getPeople(context : Context)

    // Neyin tanındığını açıklayan bir Sınıflandırıcı tarafından döndürülen değişmez bir sonuç.
    class Recognition(

        //Tanınanlar için benzersiz bir tanımlayıcı. Nesnenin örneğine değil, sınıfa özgüdür.
        val id: String?,

        //Tanımlamada görünecek isim
        val name: String?,

        /*Tanınmanın diğerlerine göre ne kadar iyi olduğuna ilişkin sıralanabilir bir puan.
        Daha düşük daha iyi olmalı.*/
        val distance: Float?,

        // Tanınan nesnenin konumu için kaynak görüntü içinde isteğe bağlı konum.
        private var location: RectF?

        ) {

        var extra: Any? = null
        var color: Int? = null
        var crop: Bitmap? = null

        fun getLocation(): RectF {
            return RectF(location)
        }

        fun setLocation(location: RectF?) {
            this.location = location
        }

        override fun toString(): String {
            var resultString = ""
            if (id != null) {
                resultString += "[$id] "
            }
            if (name != null) {
                resultString += "$name "
            }
            if (distance != null) {
                resultString += String.format("(%.1f%%) ", distance * 100.0f)
            }
            if (location != null) {
                resultString += location.toString() + " "
            }
            return resultString.trim { it <= ' ' }
        }

    }
}