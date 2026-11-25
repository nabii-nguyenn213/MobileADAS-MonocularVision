package com.example.imagepro;

import android.content.Context;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraManager;
import android.util.SizeF;
import android.util.Log;

public final class CameraProfile {

    private final float focalMm;
    private final float sensorWidthMm;
    private final float sensorHeightMm;

    private CameraProfile(float focalMm,
                          float sensorWidthMm,
                          float sensorHeightMm) {

        this.focalMm = focalMm;
        this.sensorWidthMm = sensorWidthMm;
        this.sensorHeightMm = sensorHeightMm;
    }

    public static CameraProfile fromDevice(Context context) {

        CameraManager manager = (CameraManager)
                context.getSystemService(Context.CAMERA_SERVICE);

        if (manager == null) return null;

        try {
            for (String id : manager.getCameraIdList()) {

                CameraCharacteristics c =
                        manager.getCameraCharacteristics(id);

                Integer facing = c.get(
                        CameraCharacteristics.LENS_FACING);

                if (facing == null ||
                        facing != CameraCharacteristics.LENS_FACING_BACK) {
                    continue;
                }

                float[] focalLengths = c.get(
                        CameraCharacteristics
                                .LENS_INFO_AVAILABLE_FOCAL_LENGTHS);

                SizeF sensor = c.get(
                        CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE);

                if (focalLengths == null ||
                        focalLengths.length == 0 ||
                        sensor == null ||
                        focalLengths[0] <= 0 ||
                        sensor.getWidth() <= 0 ||
                        sensor.getHeight() <= 0) {
                    continue;
                }

                CameraProfile profile = new CameraProfile(
                        focalLengths[0],
                        sensor.getWidth(),
                        sensor.getHeight()
                );

                Log.i("CameraProfile",
                        "Camera ID: " + id
                                + " | Focal: " + focalLengths[0]
                                + " mm | Sensor: " + sensor
                                + " | Estimated fx1280: "
                                + profile.getFocalPx(1280, 720, false));

                return profile;
            }

        } catch (Exception e) {
            Log.e("CameraProfile", "Cannot read camera", e);
        }

        return null;
    }

    public float getFocalPx(int width,
                            int height,
                            boolean rotated90) {

        float sensorW = rotated90
                ? sensorHeightMm : sensorWidthMm;

        float sensorH = rotated90
                ? sensorWidthMm : sensorHeightMm;

        float imageRatio = width / (float) height;
        float sensorRatio = sensorW / sensorH;

        // Approximate centered aspect-ratio crop.
        float effectiveWidth = imageRatio < sensorRatio
                ? sensorH * imageRatio
                : sensorW;

        return focalMm * width / effectiveWidth;
    }
}