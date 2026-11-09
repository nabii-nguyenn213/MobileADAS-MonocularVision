package com.example.imagepro;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.pm.PackageManager;
import android.location.Location;
import android.location.LocationListener;
import android.location.LocationManager;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.view.OrientationEventListener;
import android.view.SurfaceView;
import android.view.Window;
import android.view.WindowManager;

import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.atomic.AtomicBoolean;

public class CameraActivity extends Activity
        implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "CameraActivity";

    private static final int CAMERA_PERMISSION_REQUEST = 1;
    private static final int LOCATION_PERMISSION_REQUEST = 2;

    private static final long LOG_INTERVAL_NS =
            3_000_000_000L;

    private static final long SPEED_STALE_MS =
            4_000L;

    // Tối đa 5 lần lane inference mỗi giây.
    private static final long LANE_SUBMIT_INTERVAL_NS =
            200_000_000L;

    // Không dùng kết quả lane đã quá cũ.
    private static final long LANE_RESULT_MAX_AGE_NS =
            800_000_000L;

    private CameraBridgeViewBase cameraView;
    private objectDetectorClass vehicleDetector;

    // Chỉ laneWorker được truy cập LaneDetector.
    private LaneDetector laneDetector;

    private final ExecutorService laneWorker =
            Executors.newSingleThreadExecutor();

    private final AtomicBoolean laneBusy =
            new AtomicBoolean(false);

    private volatile boolean laneActive = false;
    private volatile int laneSession = 0;

    private volatile long lastLaneSubmitNs = 0;

    private volatile LaneSnapshot laneSnapshot =
            LaneSnapshot.EMPTY;

    private OrientationEventListener orientationListener;
    private volatile int deviceRotation = 0;

    private LocationManager locationManager;
    private LocationListener locationListener;

    private volatile float speedKmh = Float.NaN;
    private volatile long lastSpeedElapsedMs = 0L;

    private long previousFrameTime;
    private long lastLogTime;

    private float fps;
    private long latestLaneMs;

    // Snapshot bất biến: camera chỉ đọc, worker chỉ ghi mới.
    private static final class LaneSnapshot {

        static final LaneSnapshot EMPTY = new LaneSnapshot(
                Collections.<Point>emptyList(),
                Collections.<Point>emptyList(),
                Collections.<Point>emptyList(),
                Collections.<Point>emptyList(),
                0,
                "not_run",
                0,
                0,
                0,
                0
        );

        final List<Point> displayLeft;
        final List<Point> displayRight;

        final List<Point> reliableLeft;
        final List<Point> reliableRight;

        final long timestampNs;
        final String failure;

        final int maskPixels;
        final int roiPixels;
        final int lineCount;

        final long laneMs;

        LaneSnapshot(
                List<Point> dl,
                List<Point> dr,
                List<Point> rl,
                List<Point> rr,
                long timestampNs,
                String failure,
                int maskPixels,
                int roiPixels,
                int lineCount,
                long laneMs) {

            this.displayLeft = copyPoints(dl);
            this.displayRight = copyPoints(dr);

            this.reliableLeft = copyPoints(rl);
            this.reliableRight = copyPoints(rr);

            this.timestampNs = timestampNs;
            this.failure = failure;

            this.maskPixels = maskPixels;
            this.roiPixels = roiPixels;
            this.lineCount = lineCount;

            this.laneMs = laneMs;
        }

        private static List<Point> copyPoints(
                List<Point> source) {

            ArrayList<Point> copy = new ArrayList<>();

            for (Point p : source) {
                copy.add(new Point(p.x, p.y));
            }

            return Collections.unmodifiableList(copy);
        }
    }

    private final BaseLoaderCallback loaderCallback =
            new BaseLoaderCallback(this) {

                @Override
                public void onManagerConnected(int status) {

                    if (status == LoaderCallbackInterface.SUCCESS) {

                        Log.i(TAG, "OpenCV initialized");

                        if (cameraView != null && hasCameraPermission()) {
                            cameraView.enableView();
                        }

                    } else {
                        super.onManagerConnected(status);
                    }
                }
            };

    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);

        requestWindowFeature(Window.FEATURE_NO_TITLE);

        getWindow().addFlags(
                WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.activity_camera);

        cameraView = findViewById(R.id.frame_Surface);
        cameraView.setVisibility(SurfaceView.VISIBLE);
        cameraView.setCvCameraViewListener(this);

        orientationListener =
                new OrientationEventListener(this) {

                    @Override
                    public void onOrientationChanged(int orientation) {

                        if (orientation == ORIENTATION_UNKNOWN)
                            return;

                        if (orientation >= 315 || orientation < 45) {
                            deviceRotation = 0;
                        } else if (orientation < 135) {
                            deviceRotation = 90;
                        } else if (orientation < 225) {
                            deviceRotation = 180;
                        } else {
                            deviceRotation = 270;
                        }
                    }
                };

        locationManager =
                (LocationManager) getSystemService(
                        Context.LOCATION_SERVICE);

        locationListener = new LocationListener() {

            @Override
            public void onLocationChanged(Location location) {

                if (location != null &&
                        location.hasSpeed() &&
                        location.getSpeed() >= 0f) {

                    speedKmh = location.getSpeed() * 3.6f;

                    lastSpeedElapsedMs =
                            SystemClock.elapsedRealtime();

                } else {

                    speedKmh = Float.NaN;
                    lastSpeedElapsedMs = 0L;
                }
            }

            @Override
            public void onStatusChanged(
                    String provider,
                    int status,
                    Bundle extras) {
            }

            @Override
            public void onProviderEnabled(String provider) {
                startGpsUpdates();
            }

            @Override
            public void onProviderDisabled(String provider) {
                speedKmh = Float.NaN;
                lastSpeedElapsedMs = 0L;
            }
        };

        try {

            vehicleDetector = new objectDetectorClass(
                    getAssets(),
                    "yolo26n_mosaic_320_w8a32.tflite",
                    "vehicle_labels.txt",
                    320
            );

        } catch (IOException | IllegalArgumentException e) {

            Log.e(TAG, "Failed to load vehicle model", e);
        }

        if (!hasCameraPermission()) {

            ActivityCompat.requestPermissions(
                    this,
                    new String[]{Manifest.permission.CAMERA},
                    CAMERA_PERMISSION_REQUEST
            );

        } else if (!hasLocationPermission()) {

            requestLocationPermission();
        }
    }

    private boolean hasCameraPermission() {

        return ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED;
    }

    private boolean hasLocationPermission() {

        return ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.ACCESS_FINE_LOCATION
        ) == PackageManager.PERMISSION_GRANTED;
    }

    private void requestLocationPermission() {

        ActivityCompat.requestPermissions(
                this,
                new String[]{Manifest.permission.ACCESS_FINE_LOCATION},
                LOCATION_PERMISSION_REQUEST
        );
    }

    private void startGpsUpdates() {

        if (locationManager == null || !hasLocationPermission())
            return;

        try {

            if (locationManager.isProviderEnabled(
                    LocationManager.GPS_PROVIDER)) {

                locationManager.requestLocationUpdates(
                        LocationManager.GPS_PROVIDER,
                        1000L,
                        0f,
                        locationListener
                );

            } else {

                speedKmh = Float.NaN;
                lastSpeedElapsedMs = 0L;

                Log.w(TAG, "GPS is disabled");
            }

        } catch (SecurityException | IllegalArgumentException e) {

            Log.w(TAG, "Unable to start GPS updates", e);
        }
    }

    private void stopGpsUpdates() {

        if (locationManager != null && locationListener != null) {
            locationManager.removeUpdates(locationListener);
        }

        speedKmh = Float.NaN;
        lastSpeedElapsedMs = 0L;
    }

    @Override
    protected void onResume() {

        super.onResume();

        if (orientationListener != null &&
                orientationListener.canDetectOrientation()) {

            orientationListener.enable();
        }

        if (hasLocationPermission()) {
            startGpsUpdates();
        }

        if (!hasCameraPermission())
            return;

        if (OpenCVLoader.initDebug()) {

            loaderCallback.onManagerConnected(
                    LoaderCallbackInterface.SUCCESS);

        } else {

            OpenCVLoader.initAsync(
                    OpenCVLoader.OPENCV_VERSION_3_4_0,
                    this,
                    loaderCallback
            );
        }
    }

    @Override
    public void onRequestPermissionsResult(
            int requestCode,
            String[] permissions,
            int[] grantResults) {

        super.onRequestPermissionsResult(
                requestCode,
                permissions,
                grantResults);

        boolean granted = grantResults.length > 0 &&
                grantResults[0] ==
                        PackageManager.PERMISSION_GRANTED;

        if (requestCode == CAMERA_PERMISSION_REQUEST && granted) {

            if (OpenCVLoader.initDebug()) {

                loaderCallback.onManagerConnected(
                        LoaderCallbackInterface.SUCCESS);

            } else {

                OpenCVLoader.initAsync(
                        OpenCVLoader.OPENCV_VERSION_3_4_0,
                        this,
                        loaderCallback);
            }

            if (!hasLocationPermission()) {
                requestLocationPermission();
            } else {
                startGpsUpdates();
            }

        } else if (requestCode ==
                LOCATION_PERMISSION_REQUEST && granted) {

            startGpsUpdates();
        }
    }

    @Override
    protected void onPause() {

        if (orientationListener != null)
            orientationListener.disable();

        stopGpsUpdates();

        if (cameraView != null)
            cameraView.disableView();

        closeLane();

        previousFrameTime = 0;
        fps = 0;

        super.onPause();
    }

    @Override
    protected void onDestroy() {

        stopGpsUpdates();

        if (cameraView != null)
            cameraView.disableView();

        closeLane();

        laneWorker.shutdown();

        if (vehicleDetector != null)
            vehicleDetector.close();

        super.onDestroy();
    }

    @Override
    public void onCameraViewStarted(int width, int height) {

        previousFrameTime = 0;
        fps = 0;

        final int session = ++laneSession;

        laneActive = true;
        laneSnapshot = LaneSnapshot.EMPTY;
        lastLaneSubmitNs = 0;

        laneWorker.execute(() -> {

            if (!laneActive || session != laneSession)
                return;

            if (laneDetector == null) {

                try {

                    laneDetector = new LaneDetector(
                            getAssets(),
                            "medium_float32.tflite"
                    );

                    Log.i(TAG, "LaneDetector initialized");

                } catch (Exception e) {

                    Log.e(TAG,
                            "Failed to initialize LaneDetector", e);
                }
            }
        });
    }

    @Override
    public void onCameraViewStopped() {
        closeLane();
    }

    private void closeLane() {

        laneActive = false;
        ++laneSession;

        laneSnapshot = LaneSnapshot.EMPTY;

        try {

            laneWorker.execute(() -> {

                if (laneDetector != null) {

                    laneDetector.close();
                    laneDetector = null;
                }
            });

        } catch (RejectedExecutionException ignored) {
        }
    }

    private void submitLaneFrame(
            Mat uprightFrame,
            long nowNs) {

        if (!laneActive)
            return;

        if (laneBusy.get())
            return;

        if (nowNs - lastLaneSubmitNs <
                LANE_SUBMIT_INTERVAL_NS)
            return;

        if (!laneBusy.compareAndSet(false, true))
            return;

        lastLaneSubmitNs = nowNs;

        final int session = laneSession;

        // Phải copy trước khi YOLO vẽ lên frame.
        final Mat copy = uprightFrame.clone();

        try {

            laneWorker.execute(() -> {

                try {

                    if (!laneActive ||
                            session != laneSession ||
                            laneDetector == null) {
                        return;
                    }

                    long t0 = System.nanoTime();

                    laneDetector.detect(copy);

                    long elapsedMs =
                            (System.nanoTime() - t0) / 1_000_000L;

                    LaneSnapshot result = new LaneSnapshot(
                            laneDetector.getDisplayLeftPoints(),
                            laneDetector.getDisplayRightPoints(),
                            laneDetector.getLeftPoints(),
                            laneDetector.getRightPoints(),
                            System.nanoTime(),
                            laneDetector.getFailure(),
                            laneDetector.getMaskPixels(),
                            laneDetector.getRoiPixels(),
                            laneDetector.getLineCount(),
                            elapsedMs
                    );

                    if (laneActive && session == laneSession) {
                        laneSnapshot = result;
                    }

                } catch (Exception e) {

                    Log.e(TAG, "Lane worker failed", e);

                    if (laneDetector != null) {
                        laneDetector.invalidate();
                    }

                } finally {

                    copy.release();
                    laneBusy.set(false);
                }
            });

        } catch (RejectedExecutionException e) {

            copy.release();
            laneBusy.set(false);
        }
    }

    private static int getFrameRotation(int rotation) {

        switch (rotation) {

            case 0:
                return Core.ROTATE_90_CLOCKWISE;

            case 180:
                return Core.ROTATE_90_COUNTERCLOCKWISE;

            case 90:
                return Core.ROTATE_180;

            case 270:
            default:
                return -1;
        }
    }

    private static int inverseRotation(int rotation) {

        if (rotation == Core.ROTATE_90_CLOCKWISE)
            return Core.ROTATE_90_COUNTERCLOCKWISE;

        if (rotation == Core.ROTATE_90_COUNTERCLOCKWISE)
            return Core.ROTATE_90_CLOCKWISE;

        return Core.ROTATE_180;
    }

    private void drawTelemetry(
            Mat image,
            long yoloMs,
            long totalMs,
            objectDetectorClass.NearestVehicle nearest) {

        int w = image.cols();
        int h = image.rows();

        if (w <= 0 || h <= 0)
            return;

        double scale = Math.max(
                0.55,
                Math.min(w / 720.0, h / 480.0));

        int margin = (int) Math.round(16 * scale);

        int panelW = Math.min(
                w - 2 * margin,
                (int) Math.round(390 * scale));

        int panelH = (int) Math.round(184 * scale);

        int left = margin;
        int top = margin;

        if (panelW <= 0 || panelH <= 0)
            return;

        Rect panelRect = new Rect(
                left,
                top,
                Math.min(panelW, w - left),
                Math.min(panelH, h - top)
        );

        Mat roi = image.submat(panelRect);

        Mat shade = new Mat(
                roi.size(),
                roi.type(),
                new Scalar(18, 32, 40, 255)
        );

        Core.addWeighted(
                shade,
                0.70,
                roi,
                0.30,
                0.0,
                roi
        );

        shade.release();
        roi.release();

        long age =
                SystemClock.elapsedRealtime() -
                        lastSpeedElapsedMs;

        boolean valid =
                lastSpeedElapsedMs > 0 &&
                        age >= 0 &&
                        age <= SPEED_STALE_MS &&
                        !Float.isNaN(speedKmh);

        String speedText = valid
                ? String.format(Locale.US, "%.0f", speedKmh)
                : "--";

        Imgproc.putText(
                image,
                "GPS SPEED",
                new Point(
                        left + 12 * scale,
                        top + 23 * scale),
                Imgproc.FONT_HERSHEY_SIMPLEX,
                0.52 * scale,
                new Scalar(200, 220, 225, 255),
                1
        );

        Imgproc.putText(
                image,
                speedText,
                new Point(
                        left + 12 * scale,
                        top + 79 * scale),
                Imgproc.FONT_HERSHEY_SIMPLEX,
                1.65 * scale,
                new Scalar(255, 255, 255, 255),
                Math.max(2, (int) (3 * scale))
        );

        Imgproc.putText(
                image,
                "km/h",
                new Point(
                        left + 135 * scale,
                        top + 77 * scale),
                Imgproc.FONT_HERSHEY_SIMPLEX,
                0.65 * scale,
                new Scalar(210, 220, 225, 255),
                2
        );

        Imgproc.putText(
                image,
                String.format(Locale.US, "FPS %.1f", fps),
                new Point(
                        left + 235 * scale,
                        top + 40 * scale),
                Imgproc.FONT_HERSHEY_SIMPLEX,
                0.60 * scale,
                new Scalar(65, 230, 120, 255),
                2
        );

        Imgproc.putText(
                image,
                String.format(
                        Locale.US,
                        "YOLO %d ms",
                        yoloMs),
                new Point(
                        left + 235 * scale,
                        top + 70 * scale),
                Imgproc.FONT_HERSHEY_SIMPLEX,
                0.45 * scale,
                new Scalar(225, 225, 225, 255),
                1
        );

        Imgproc.putText(
                image,
                String.format(
                        Locale.US,
                        "Total %d ms",
                        totalMs),
                new Point(
                        left + 235 * scale,
                        top + 95 * scale),
                Imgproc.FONT_HERSHEY_SIMPLEX,
                0.45 * scale,
                new Scalar(225, 225, 225, 255),
                1
        );

        Imgproc.putText(
                image,
                String.format(
                        Locale.US,
                        "Lane %d ms",
                        latestLaneMs),
                new Point(
                        left + 235 * scale,
                        top + 120 * scale),
                Imgproc.FONT_HERSHEY_SIMPLEX,
                0.45 * scale,
                new Scalar(225, 225, 225, 255),
                1
        );

        String nearestText = nearest == null
                ? "NEAREST: --"
                : String.format(
                Locale.US,
                "NEAREST: %s %.1f m",
                nearest.label,
                nearest.distanceMeters);

        Imgproc.putText(
                image,
                nearestText,
                new Point(
                        left + 12 * scale,
                        top + 157 * scale),
                Imgproc.FONT_HERSHEY_SIMPLEX,
                0.56 * scale,
                new Scalar(255, 225, 80, 255),
                2
        );
    }

    @Override
    public Mat onCameraFrame(
            CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        final long start = System.nanoTime();

        if (previousFrameTime != 0) {

            long delta = start - previousFrameTime;

            if (delta > 0) {

                float instantFps =
                        1_000_000_000f / delta;

                fps = fps == 0
                        ? instantFps
                        : 0.8f * fps + 0.2f * instantFps;
            }
        }

        previousFrameTime = start;

        Mat frame = inputFrame.rgba();

        int rotationCode =
                getFrameRotation(deviceRotation);

        Mat modelFrame = new Mat();

        try {

            if (rotationCode == -1) {

                frame.copyTo(modelFrame);

            } else {

                Core.rotate(
                        frame,
                        modelFrame,
                        rotationCode);
            }

            // Không chờ lane inference.
            submitLaneFrame(
                    modelFrame,
                    System.nanoTime());

            // Đọc snapshot gần nhất.
            LaneSnapshot snapshot = laneSnapshot;

            boolean fresh =
                    snapshot.timestampNs != 0 &&
                            System.nanoTime() - snapshot.timestampNs
                                    <= LANE_RESULT_MAX_AGE_NS;

            latestLaneMs =
                    fresh ? snapshot.laneMs : 0;

            if (vehicleDetector != null) {

                vehicleDetector.setEgoLanePoints(
                        fresh ? snapshot.reliableLeft : null,
                        fresh ? snapshot.reliableRight : null
                );
            }

            // Tô lane trước YOLO.
            if (fresh) {

                LaneOverlayRenderer.draw(
                        modelFrame,
                        snapshot.displayLeft,
                        snapshot.displayRight
                );
            }

            long yoloStart = System.nanoTime();

            if (vehicleDetector != null) {

                vehicleDetector.recognizeImage(modelFrame);
            }

            long yoloMs =
                    (System.nanoTime() - yoloStart)
                            / 1_000_000L;

            long totalMs =
                    (System.nanoTime() - start)
                            / 1_000_000L;

            objectDetectorClass.NearestVehicle nearest =
                    vehicleDetector == null
                            ? null
                            : vehicleDetector.getNearestVehicle();

            drawTelemetry(
                    modelFrame,
                    yoloMs,
                    totalMs,
                    nearest
            );

            if (rotationCode == -1) {

                modelFrame.copyTo(frame);

            } else {

                Core.rotate(
                        modelFrame,
                        frame,
                        inverseRotation(rotationCode));
            }

            if (start - lastLogTime >= LOG_INTERVAL_NS) {

                Log.d(
                        TAG,
                        String.format(
                                Locale.US,
                                "rotation=%d | frame=%dx%d | " +
                                        "fps=%.1f | YOLO=%d ms | " +
                                        "speed=%.1f km/h | nearest=%s | " +
                                        "laneMs=%d | lane=%s | " +
                                        "mask=%d roi=%d lines=%d",
                                deviceRotation,
                                frame.cols(),
                                frame.rows(),
                                fps,
                                yoloMs,
                                speedKmh,
                                nearest == null
                                        ? "none"
                                        : String.format(
                                        Locale.US,
                                        "%s %.1fm",
                                        nearest.label,
                                        nearest.distanceMeters),
                                latestLaneMs,
                                fresh
                                        ? snapshot.failure
                                        : "unavailable/stale",
                                fresh
                                        ? snapshot.maskPixels : 0,
                                fresh
                                        ? snapshot.roiPixels : 0,
                                fresh
                                        ? snapshot.lineCount : 0
                        )
                );

                lastLogTime = start;
            }

            return frame;

        } finally {

            modelFrame.release();
        }
    }
}