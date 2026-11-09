package com.example.imagepro;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.util.Log;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;

/**
 * YOLO26n TFLite CPU detector. Call recognizeImage() sequentially on one camera thread.
 * Input: RGBA OpenCV camera Mat; model: float32 NCHW [1,3,320,320].
 */
public class objectDetectorClass {
    private static final String PERF_TAG = "YOLO_PERF";
    private static final int INPUT_SIZE = 320;
    private static final int NUM_CLASSES = 4;
    private static final int NUM_PREDICTIONS = 2100;
    private static final int PLANE_SIZE = INPUT_SIZE * INPUT_SIZE;
    private static final float CONF_THRESHOLD = 0.1f;
    private static final float IOU_THRESHOLD = 0.5f;
    private static final int PERF_LOG_EVERY = 30;

    // Calibrated for 1280-pixel-wide original camera image, not YOLO input 320.
    private static final float FOCAL_LENGTH_PX_AT_1280 = 625.0f;
    private static final float CALIBRATION_IMAGE_WIDTH = 1280.0f;
    // Width assumptions in meters: car, motorbike, bus, truck.
    private static final float[] VEHICLE_WIDTH_METERS = {1.6f, 0.7f, 2.5f, 2.0f};
    private volatile NearestVehicle nearestVehicle = null;

    // Real lane boundaries from TwinLiteNet+; no virtual/fixed lane lines.
    private List<Point> leftLanePoints=Collections.emptyList();
    private List<Point> rightLanePoints=Collections.emptyList();
    private static final float LANE_MARGIN_PX=8f;
    public synchronized void setEgoLanePoints(List<Point> left,List<Point> right) {
        leftLanePoints=left==null?Collections.<Point>emptyList():new ArrayList<>(left);
        rightLanePoints=right==null?Collections.<Point>emptyList():new ArrayList<>(right);
    }

    // OpenCV Mat is RGBA here.
    private static final Scalar CAR_COLOR = new Scalar(0, 255, 0, 255);
    private static final Scalar MOTORBIKE_COLOR = new Scalar(255, 165, 0, 255);
    private static final Scalar BUS_COLOR = new Scalar(255, 0, 255, 255);
    private static final Scalar TRUCK_COLOR = new Scalar(0, 165, 255, 255);
    private static final Scalar OUTSIDE_COLOR = new Scalar(150, 150, 150, 255);
    private static final Scalar UNKNOWN_COLOR = new Scalar(255, 255, 0, 255);
    private static final Scalar LEAD_COLOR = new Scalar(255, 0, 0, 255);

    private final Interpreter interpreter;
    private final List<String> labelList;

    // Reused buffers: no per-frame direct-buffer, tensor or image-array allocations.
    private final ByteBuffer inputBuffer = ByteBuffer.allocateDirect(3 * PLANE_SIZE * 4)
            .order(ByteOrder.nativeOrder());
    private final byte[] rgbBytes = new byte[3 * PLANE_SIZE];
    private final float[][][] output = new float[1][NUM_CLASSES + 4][NUM_PREDICTIONS];

    // Reused native OpenCV buffers. Never release these inside recognizeImage().
    private final Mat workingImage = new Mat();
    private final Mat resizedImage = new Mat();
    private final Mat letterboxImage = new Mat();
    private final Mat rgbImage = new Mat();

    private int perfFrameCount = 0;
    private double perfConvertNs, perfPreprocessNs, perfInferenceNs;
    private double perfDecodeNs, perfNmsNs, perfDrawNs, perfTotalNs;
    private boolean closed = false;

    public objectDetectorClass(AssetManager assetManager, String modelPath,
                               String labelPath, int inputSize) throws IOException {
        if (inputSize != INPUT_SIZE) {
            throw new IllegalArgumentException("YOLO26 model requires inputSize=320");
        }
        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(2);
        interpreter = new Interpreter(loadModelFile(assetManager, modelPath), options);
        labelList = loadLabelList(assetManager, labelPath);

        if (labelList.size() != NUM_CLASSES) {
            interpreter.close();
            throw new IllegalArgumentException("Expected 4 labels, got " + labelList.size());
        }
        int[] in = interpreter.getInputTensor(0).shape();
        int[] out = interpreter.getOutputTensor(0).shape();
        if (in.length != 4 || in[0] != 1 || in[1] != 3
                || in[2] != INPUT_SIZE || in[3] != INPUT_SIZE) {
            interpreter.close();
            throw new IllegalArgumentException("Unexpected model input shape");
        }
        if (out.length != 3 || out[0] != 1 || out[1] != NUM_CLASSES + 4
                || out[2] != NUM_PREDICTIONS) {
            interpreter.close();
            throw new IllegalArgumentException("Unexpected model output shape");
        }
    }

    private static ByteBuffer loadModelFile(AssetManager assets, String path) throws IOException {
        try (AssetFileDescriptor fd = assets.openFd(path);
             FileInputStream stream = new FileInputStream(fd.getFileDescriptor());
             FileChannel channel = stream.getChannel()) {
            return channel.map(FileChannel.MapMode.READ_ONLY,
                    fd.getStartOffset(), fd.getDeclaredLength());
        }
    }

    private static List<String> loadLabelList(AssetManager assets, String path) throws IOException {
        List<String> labels = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(assets.open(path)))) {
            String line;
            while ((line = reader.readLine()) != null) {
                String value = line.trim();
                if (!value.isEmpty()) labels.add(value);
            }
        }
        return labels;
    }

    // This method must not be called concurrently on the same detector instance.
    public synchronized Mat recognizeImage(Mat matImage) {
        if (closed) throw new IllegalStateException("Detector already closed");
        final long start = System.nanoTime();

        // Orientation is normalized by CameraActivity.
        matImage.copyTo(workingImage);
        final long afterConvert = System.nanoTime();

        int originalWidth = workingImage.cols();
        int originalHeight = workingImage.rows();
        float scale = Math.min(INPUT_SIZE / (float) originalWidth,
                INPUT_SIZE / (float) originalHeight);
        int resizedWidth = Math.round(originalWidth * scale);
        int resizedHeight = Math.round(originalHeight * scale);
        int padLeft = (INPUT_SIZE - resizedWidth) / 2;
        int padTop = (INPUT_SIZE - resizedHeight) / 2;
        int padRight = INPUT_SIZE - resizedWidth - padLeft;
        int padBottom = INPUT_SIZE - resizedHeight - padTop;

        // Resize and letterbox in native OpenCV, without Bitmap or Canvas.
        Imgproc.resize(workingImage, resizedImage,
                new Size(resizedWidth, resizedHeight), 0, 0, Imgproc.INTER_LINEAR);
        Core.copyMakeBorder(resizedImage, letterboxImage,
                padTop, padBottom, padLeft, padRight,
                Core.BORDER_CONSTANT, new Scalar(114, 114, 114, 255));
        Imgproc.cvtColor(letterboxImage, rgbImage, Imgproc.COLOR_RGBA2RGB);
        rgbImage.get(0, 0, rgbBytes);

        // Convert interleaved unsigned RGB bytes into normalized planar RGB floats.
        inputBuffer.clear();
        for (int i = 0; i < PLANE_SIZE; i++) {
            inputBuffer.putFloat((rgbBytes[3 * i] & 0xFF) / 255.0f);
        }
        for (int i = 0; i < PLANE_SIZE; i++) {
            inputBuffer.putFloat((rgbBytes[3 * i + 1] & 0xFF) / 255.0f);
        }
        for (int i = 0; i < PLANE_SIZE; i++) {
            inputBuffer.putFloat((rgbBytes[3 * i + 2] & 0xFF) / 255.0f);
        }
        inputBuffer.rewind();
        final long afterPreprocess = System.nanoTime();

        // Run TFLite inference.
        interpreter.run(inputBuffer, output);
        final long afterInference = System.nanoTime();

        List<Detection> detections = decodeOutput(output[0], scale, padLeft, padTop,
                originalWidth, originalHeight);
        final long afterDecode = System.nanoTime();
        detections = applyNms(detections);
        final long afterNms = System.nanoTime();


        Detection closestDetection = null;
        float closestDistance = Float.POSITIVE_INFINITY;
        int insideCount = 0;
        // Convert focal length to original frame width; never use 320 directly.
        float focalPx = FOCAL_LENGTH_PX_AT_1280 * originalWidth / CALIBRATION_IMAGE_WIDTH;
        for (Detection d : detections) {
            boolean inside = classifyLane(d) == LaneStatus.INSIDE;
            if (!inside) continue;
            insideCount++;
            float bboxWidth = d.x2 - d.x1;
            if (bboxWidth <= 0) continue;
            float distance = focalPx * VEHICLE_WIDTH_METERS[d.classId] / bboxWidth;
            if ((!Float.isNaN(distance) && !Float.isInfinite(distance)) && distance > 0 && distance < closestDistance) {
                closestDistance = distance;
                closestDetection = d;
            }
        }
        nearestVehicle = closestDetection == null ? null : new NearestVehicle(
                labelList.get(closestDetection.classId), closestDetection.classId,
                closestDistance, closestDetection.confidence, insideCount);

        for (Detection d : detections) {
            // The bottom-center of the bounding box approximates tire/road contact.
            boolean inside = classifyLane(d) == LaneStatus.INSIDE;
            LaneStatus status=classifyLane(d);
            Scalar color=inside?colorForClass(d.classId):(status==LaneStatus.OUTSIDE?OUTSIDE_COLOR:UNKNOWN_COLOR);
            Imgproc.rectangle(workingImage, new Point(d.x1, d.y1),
                    new Point(d.x2, d.y2), color, 2);
            float distance = Float.NaN;
            if (inside && d.x2 > d.x1) {
                distance = focalPx * VEHICLE_WIDTH_METERS[d.classId] / (d.x2 - d.x1);
            }
            String label = labelList.get(d.classId) + " "
                    + String.format(Locale.US, "%.2f", d.confidence);
            if (inside && (!Float.isNaN(distance) && !Float.isInfinite(distance))) {
                label += String.format(Locale.US, " %.1fm", distance);
            }
            if (status==LaneStatus.UNKNOWN) label += " | LANE UNKNOWN";
            if (d == closestDetection) {
                Imgproc.rectangle(workingImage, new Point(d.x1 - 3, d.y1 - 3),
                        new Point(d.x2 + 3, d.y2 + 3), LEAD_COLOR, 3);
                label = "NEAREST " + label;
            }
            Imgproc.putText(workingImage, label,
                    new Point(d.x1, Math.max(20, (int) d.y1)),
                    Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
        }
        final long afterDraw = System.nanoTime();

        workingImage.copyTo(matImage);
        final long end = System.nanoTime();
        recordPerformance(afterConvert - start,
                afterPreprocess - afterConvert,
                afterInference - afterPreprocess,
                afterDecode - afterInference,
                afterNms - afterDecode,
                afterDraw - afterNms,
                end - start);
        return matImage;
    }

    public static final class NearestVehicle {
        public final String label;
        public final int classId;
        public final float distanceMeters;
        public final float confidence;
        public final int vehiclesInLane;

        private NearestVehicle(String label, int classId, float distanceMeters,
                               float confidence, int vehiclesInLane) {
            this.label = label;
            this.classId = classId;
            this.distanceMeters = distanceMeters;
            this.confidence = confidence;
            this.vehiclesInLane = vehiclesInLane;
        }
    }

    public NearestVehicle getNearestVehicle() {
        return nearestVehicle;
    }

    private static Scalar colorForClass(int classId) {
        switch (classId) {
            case 0: return CAR_COLOR;
            case 1: return MOTORBIKE_COLOR;
            case 2: return BUS_COLOR;
            case 3: return TRUCK_COLOR;
            default: return OUTSIDE_COLOR;
        }
    }

    private enum LaneStatus {INSIDE,OUTSIDE,UNKNOWN}

    /** Only interpolate within observed support. No lane estimate => UNKNOWN. */
    private static float laneXAtY(List<Point> points,float y) {
        if (points==null || points.size()<2 || Float.isNaN(y) || Float.isInfinite(y)) return Float.NaN;
        Point lo=null,hi=null;
        for (Point p:points) {
            if (Double.isNaN(p.x)||Double.isInfinite(p.x)||Double.isNaN(p.y)||Double.isInfinite(p.y)) continue;
            if (p.y<=y && (lo==null || p.y>lo.y)) lo=p;
            if (p.y>=y && (hi==null || p.y<hi.y)) hi=p;
        }
        if (lo==null || hi==null) return Float.NaN;
        if (Math.abs(hi.y-lo.y)<.00001) return (float)lo.x;
        return (float)(lo.x+(y-lo.y)*(hi.x-lo.x)/(hi.y-lo.y));
    }
    private LaneStatus classifyLane(Detection d) {
        // Cropped boxes have unreliable ground contact points.
        if (d.x1<=0 || d.y1<=0 || d.x2>=workingImage.cols()-1 ||
                d.y2>=workingImage.rows()-1) return LaneStatus.UNKNOWN;
        float lx=laneXAtY(leftLanePoints,d.y2),rx=laneXAtY(rightLanePoints,d.y2);
        if (Float.isNaN(lx)||Float.isNaN(rx)||rx-lx<=Math.max(2*LANE_MARGIN_PX,.015f*workingImage.cols()) ||
                rx-lx>.95f*workingImage.cols()) return LaneStatus.UNKNOWN;
        float x=(d.x1+d.x2)*.5f;
        if (Math.abs(x-lx)<=LANE_MARGIN_PX || Math.abs(x-rx)<=LANE_MARGIN_PX) return LaneStatus.UNKNOWN;
        return x>lx+LANE_MARGIN_PX && x<rx-LANE_MARGIN_PX?LaneStatus.INSIDE:LaneStatus.OUTSIDE;
    }

    private void recordPerformance(long convert, long preprocess, long inference,
                                   long decode, long nms, long draw, long total) {
        perfFrameCount++;
        perfConvertNs += convert;
        perfPreprocessNs += preprocess;
        perfInferenceNs += inference;
        perfDecodeNs += decode;
        perfNmsNs += nms;
        perfDrawNs += draw;
        perfTotalNs += total;
        if (perfFrameCount >= PERF_LOG_EVERY) {
            double n = perfFrameCount;
            Log.d(PERF_TAG, String.format(Locale.US,
                    "Average over %.0f frames (ms): convert=%.1f | preprocess=%.1f | inference=%.1f | decode=%.1f | NMS=%.1f | draw=%.1f | total=%.1f | approxFPS=%.2f",
                    n, perfConvertNs / n / 1e6, perfPreprocessNs / n / 1e6,
                    perfInferenceNs / n / 1e6, perfDecodeNs / n / 1e6,
                    perfNmsNs / n / 1e6, perfDrawNs / n / 1e6,
                    perfTotalNs / n / 1e6, 1e9 / (perfTotalNs / n)));
            perfFrameCount = 0;
            perfConvertNs = perfPreprocessNs = perfInferenceNs = 0;
            perfDecodeNs = perfNmsNs = perfDrawNs = perfTotalNs = 0;
        }
    }

    private List<Detection> decodeOutput(float[][] raw, float scale,
                                         int padLeft, int padTop,
                                         int originalWidth, int originalHeight) {
        List<Detection> detections = new ArrayList<>();
        for (int i = 0; i < NUM_PREDICTIONS; i++) {
            int bestClass = 0;
            float bestScore = raw[4][i];
            for (int c = 1; c < NUM_CLASSES; c++) {
                if (raw[4 + c][i] > bestScore) {
                    bestScore = raw[4 + c][i];
                    bestClass = c;
                }
            }
            if (bestScore < CONF_THRESHOLD) continue;
            float cx = raw[0][i] * INPUT_SIZE;
            float cy = raw[1][i] * INPUT_SIZE;
            float w = raw[2][i] * INPUT_SIZE;
            float h = raw[3][i] * INPUT_SIZE;
            float x1 = clamp((cx - w / 2f - padLeft) / scale, 0, originalWidth - 1);
            float y1 = clamp((cy - h / 2f - padTop) / scale, 0, originalHeight - 1);
            float x2 = clamp((cx + w / 2f - padLeft) / scale, 0, originalWidth - 1);
            float y2 = clamp((cy + h / 2f - padTop) / scale, 0, originalHeight - 1);
            if (x2 <= x1 || y2 <= y1) continue;
            detections.add(new Detection(x1, y1, x2, y2, bestScore, bestClass));
        }
        return detections;
    }

    private List<Detection> applyNms(List<Detection> detections) {
        List<Detection> kept = new ArrayList<>();
        for (int classId = 0; classId < NUM_CLASSES; classId++) {
            List<Detection> candidates = new ArrayList<>();
            for (Detection d : detections) {
                if (d.classId == classId) candidates.add(d);
            }
            Collections.sort(candidates, new Comparator<Detection>() {
                @Override public int compare(Detection a, Detection b) {
                    return Float.compare(b.confidence, a.confidence);
                }
            });
            boolean[] suppressed = new boolean[candidates.size()];
            for (int i = 0; i < candidates.size(); i++) {
                if (suppressed[i]) continue;
                Detection best = candidates.get(i);
                kept.add(best);
                for (int j = i + 1; j < candidates.size(); j++) {
                    if (!suppressed[j] && iou(best, candidates.get(j)) > IOU_THRESHOLD) {
                        suppressed[j] = true;
                    }
                }
            }
        }
        return kept;
    }

    private static float iou(Detection a, Detection b) {
        float left = Math.max(a.x1, b.x1);
        float top = Math.max(a.y1, b.y1);
        float right = Math.min(a.x2, b.x2);
        float bottom = Math.min(a.y2, b.y2);
        float intersection = Math.max(0, right - left) * Math.max(0, bottom - top);
        float areaA = Math.max(0, a.x2 - a.x1) * Math.max(0, a.y2 - a.y1);
        float areaB = Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);
        float union = areaA + areaB - intersection;
        return union <= 0 ? 0 : intersection / union;
    }

    private static float clamp(float v, float min, float max) {
        return Math.max(min, Math.min(max, v));
    }

    private static class Detection {
        final float x1, y1, x2, y2, confidence;
        final int classId;
        Detection(float x1, float y1, float x2, float y2, float confidence, int classId) {
            this.x1 = x1;
            this.y1 = y1;
            this.x2 = x2;
            this.y2 = y2;
            this.confidence = confidence;
            this.classId = classId;
        }
    }
    public synchronized void close() {
        if (closed) return;
        closed = true;
        nearestVehicle = null;
        interpreter.close();
        workingImage.release();
        resizedImage.release();
        letterboxImage.release();
        rgbImage.release();
    }
}
