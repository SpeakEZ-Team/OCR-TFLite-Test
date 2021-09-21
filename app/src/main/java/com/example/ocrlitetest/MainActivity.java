package com.example.ocrlitetest;

import androidx.appcompat.app.AppCompatActivity;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.os.Bundle;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Locale;

import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

public class MainActivity extends AppCompatActivity {

    protected Interpreter tflite;

    ImageProcessor imageProcessor;
    ByteBuffer tImage;
    Bitmap bitmap;
    AssetManager assetManager;
    InputStream istr;
    private TensorBuffer outputProbabilityBuffer;
    TextView lbl;
    TextToSpeech speaker;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // tImage = new TensorImage(DataType.FLOAT32);
        assetManager = getAssets();

        imageProcessor =
                new ImageProcessor.Builder()
                        .build();

        final Button button = findViewById(R.id.button);
        lbl = findViewById(R.id.label);
        button.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                runInference();
            }
        });

        speaker=new TextToSpeech(getApplicationContext(), new TextToSpeech.OnInitListener() {
            @Override
            public void onInit(int status) {
                if(status != TextToSpeech.ERROR) {
                    speaker.setLanguage(Locale.US);
                }
            }
        });

    }

    private void runInference() {
        try {
            tflite = new Interpreter(loadModelFile());
        } catch (Exception ex) {
            ex.printStackTrace();
        }

        try {
            istr = assetManager.open("hello_ez.png");
        } catch (Exception ex) {
            ex.printStackTrace();
        }

        bitmap = BitmapFactory.decodeStream(istr);
        tImage = getGSBuffer(bitmap);


        /** Output probability TensorBuffer. */
        //...
        // Get the array size for the output buffer from the TensorFlow Lite model file
        int probabilityTensorIndex = 0;
        int[] probabilityShape =
                tflite.getOutputTensor(probabilityTensorIndex).shape();
        DataType probabilityDataType =
                tflite.getOutputTensor(probabilityTensorIndex).dataType();

        // Creates the output tensor and its processor.
        outputProbabilityBuffer =
                TensorBuffer.createFixedSize(probabilityShape, probabilityDataType);

        tflite.run(tImage,
                outputProbabilityBuffer.getBuffer().rewind());

        String alphabet = "0123456789abcdefghijklmnopqrstuvwxyz";
        char[] predict = new char[48];

        for (int j=0; j<48; ++j) {
            for (int i = 0; i < 37; ++i) {
                if (i != 36 && outputProbabilityBuffer.getFloatValue((i + 37 * j)) > 0.9) {
                    predict[j] = alphabet.charAt(i);
                }

            }
        }

        StringBuilder sb = new StringBuilder();

        for (int i=0; i<predict.length; ++i) {
            if (predict[i] != '\0') {
                sb.append(predict[i]);
            }
        }

        String predict_s = sb.toString();

        lbl.setText(predict_s);
        speaker.speak(predict_s, TextToSpeech.QUEUE_FLUSH, null);

    }

    /** Memory-map the model file in Assets. */
    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd("keras_ocr_dr.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private ByteBuffer getGSBuffer(Bitmap bitmap){
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        ByteBuffer mImgData = ByteBuffer
                .allocateDirect(4 * width * height);
        mImgData.order(ByteOrder.nativeOrder());
        int[] pixels = new int[width*height];
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);
        for (int pixel : pixels) {
            float value = (float) Color.red(pixel)/255.0f;
            mImgData.putFloat(value);
        }

        return mImgData;
    }

}