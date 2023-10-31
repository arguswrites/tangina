package com.example.new_tester

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.SurfaceTexture
import android.hardware.camera2.*
import android.support.v7.app.AppCompatActivity
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.support.v4.app.ActivityCompat
import android.support.v4.content.ContextCompat
import android.util.Log
import android.view.Surface
import android.view.TextureView
import android.widget.Button
import android.widget.TextView
import com.example.new_tester.ml.Detect
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer


class MainActivity : AppCompatActivity() {

    private lateinit var textureView: TextureView
    private lateinit var captureButton: Button
    private lateinit var resultTextView: TextView
    private lateinit var cameraDevice: CameraDevice
    private lateinit var cameraCaptureSession: CameraCaptureSession
    private lateinit var captureRequestBuilder: CaptureRequest.Builder
    private lateinit var handler: Handler
    private lateinit var backgroundThread: HandlerThread
    private lateinit var model: Detect
    private val cameraManager: CameraManager by lazy {
        getSystemService(Context.CAMERA_SERVICE) as CameraManager
    }

    // Example: Create a ByteBuffer (replace this with your actual input data)
    private val byteBuffer = ByteBuffer.allocateDirect(4 * 320 * 320 * 3) // Adjust the size based on your input dimensions
    private val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 320, 320, 3), DataType.FLOAT32)

    // TextureListener
    private val textureListener = object : TextureView.SurfaceTextureListener {
        override fun onSurfaceTextureAvailable(surface: SurfaceTexture, width: Int, height: Int) {
            openCamera()
            Log.d("Module Opened:", "Camera has been opened")
        }

        override fun onSurfaceTextureSizeChanged(surface: SurfaceTexture, width: Int, height: Int) {}

        override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean {
            return false
        }

        override fun onSurfaceTextureUpdated(surface: SurfaceTexture) {}
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        textureView = findViewById(R.id.textureView)
        captureButton = findViewById(R.id.captureButton)
        resultTextView = findViewById(R.id.resultTextView)
        val openCameraButton: Button = findViewById(R.id.captureButton)

        // Initialize TFLite model
        model = Detect.newInstance(this)

        openCameraButton.setOnClickListener {
            if (ContextCompat.checkSelfPermission(
                    this,
                    Manifest.permission.CAMERA
                ) == PackageManager.PERMISSION_GRANTED
            ) {
                openCamera()
            } else {
                ActivityCompat.requestPermissions(
                    this,
                    arrayOf(Manifest.permission.CAMERA),
                    REQUEST_CAMERA_PERMISSION
                )
            }
        }
    }

    private val stateCallback = object : CameraDevice.StateCallback() {
        override fun onOpened(camera: CameraDevice) {
            cameraDevice = camera
            createCameraPreviewSession()

            inputFeature0.loadBuffer(byteBuffer)

            val outputs = model.process(inputFeature0)


        }

        override fun onDisconnected(camera: CameraDevice) {
            cameraDevice.close()
        }

        override fun onError(camera: CameraDevice, error: Int) {
            cameraDevice.close()
            this@MainActivity.finish()
        }
    }

    private fun openCamera() {
        val cameraId = cameraManager.cameraIdList[0]
        try {
            if (ContextCompat.checkSelfPermission(
                    this,
                    Manifest.permission.CAMERA
                ) == PackageManager.PERMISSION_GRANTED
            ) {
                cameraManager.openCamera(cameraId, stateCallback, null)
            }
        } catch (e: CameraAccessException) {
            e.printStackTrace()
        }
    }

    @Suppress("DEPRECATION")
    private fun createCameraPreviewSession() {
        try {
            val texture = textureView.surfaceTexture
            if (texture != null) {
                texture.setDefaultBufferSize(1080, 1920)
            }

            val surface = Surface(texture)

            captureRequestBuilder =
                cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
            captureRequestBuilder.addTarget(surface)

            cameraDevice.createCaptureSession(
                listOf(surface),
                object : CameraCaptureSession.StateCallback() {
                    override fun onConfigured(session: CameraCaptureSession) {
                        cameraCaptureSession = session
                        updatePreview()
                    }

                    override fun onConfigureFailed(session: CameraCaptureSession) {}
                }, null
            )
        } catch (e: CameraAccessException) {
            e.printStackTrace()
        }
    }

    private fun updatePreview() {
        captureRequestBuilder.set(CaptureRequest.CONTROL_MODE, CaptureRequest.CONTROL_MODE_AUTO)

        try {
            cameraCaptureSession.setRepeatingRequest(
                captureRequestBuilder.build(),
                null,
                handler
            )
        } catch (e: CameraAccessException) {
            e.printStackTrace()
        }
    }

    private fun processObjectDetectionResults(outputs: Detect.Outputs) {
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer
        val outputFeature1 = outputs.outputFeature1AsTensorBuffer
        val outputFeature2 = outputs.outputFeature2AsTensorBuffer
        val outputFeature3 = outputs.outputFeature3AsTensorBuffer

        val detectedObjects = interpretOutput(outputFeature0, outputFeature1)
        updateUI(detectedObjects)
    }

    private fun interpretOutput(
        outputFeature0: TensorBuffer,
        outputFeature1: TensorBuffer
    ): List<DetectedObject> {
        val detectedObjects = mutableListOf<DetectedObject>()

        val numDetections = outputFeature0.intArray[0]
        for (i in 0 until numDetections) {
            val classLabel = outputFeature1.getFloatValue(i * 4)
            val confidence = outputFeature1.getFloatValue(i * 4 + 1)
            val boundingBox = RectF(
                outputFeature1.getFloatValue(i * 4 + 2),
                outputFeature1.getFloatValue(i * 4 + 3),
                outputFeature1.getFloatValue(i * 4 + 4),
                outputFeature1.getFloatValue(i * 4 + 5)
            )

            val detectedObject = DetectedObject(classLabel.toString(), confidence, boundingBox)
            detectedObjects.add(detectedObject)
        }

        return detectedObjects
    }

    data class DetectedObject(val classLabel: String, val confidence: Float, val boundingBox: RectF)


    private fun updateUI(detectedObjects: List<DetectedObject>) {
        runOnUiThread {
            textureView.invalidate()

            val canvas = textureView.lockCanvas()
            for (detectedObject in detectedObjects) {
                val rect = RectF(
                    /* left = */
                    detectedObject.boundingBox.left * textureView.width.toFloat(),
                    /* top = */
                    (detectedObject.boundingBox.top * textureView.height).toFloat(),
                    /* right = */
                    (detectedObject.boundingBox.right * textureView.width).toFloat(),
                    /* bottom = */
                    (detectedObject.boundingBox.bottom * textureView.height).toFloat()
                )
                if (canvas != null) {
                    canvas.drawRect(rect, Paint().apply {
                        color = Color.RED
                        style = Paint.Style.STROKE
                        strokeWidth = 2.0f
                    })
                }
            }
            if (canvas != null) {
                textureView.unlockCanvasAndPost(canvas)
            }
        }
    }

    private fun startBackgroundThread() {
        backgroundThread = HandlerThread("CameraBackground")
        backgroundThread.start()
        handler = Handler(backgroundThread.looper)
    }

    private fun stopBackgroundThread() {
        backgroundThread.quitSafely()
        try {
            backgroundThread.join()
        } catch (e: InterruptedException) {
            e.printStackTrace()
        }
    }

    override fun onResume() {
        super.onResume()
        startBackgroundThread()

        if (textureView.isAvailable) {
            openCamera()
        } else {
            textureView.surfaceTextureListener = textureListener
        }
    }

    override fun onPause() {
        closeCamera()
        stopBackgroundThread()

        // Close TFLite model
        model.close()

        super.onPause()
    }

    private fun closeCamera() {
        cameraCaptureSession.close()
        cameraDevice.close()
    }

    companion object {
        private const val REQUEST_CAMERA_PERMISSION = 1
    }
}