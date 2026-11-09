package com.example.imagepro;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public final class LaneOverlayRenderer {

    private LaneOverlayRenderer() {}

    private static final Scalar LANE_FILL_RGBA =
            new Scalar(155, 220, 255, 255);

    private static final double LANE_FILL_ALPHA = 0.25;

    private static double interpolateX(
            List<Point> points, double y) {

        if (y <= points.get(0).y)
            return points.get(0).x;

        for (int i = 1; i < points.size(); i++) {
            Point a = points.get(i - 1);
            Point b = points.get(i);

            if (y <= b.y) {
                double dy = b.y - a.y;

                if (Math.abs(dy) < 1e-9)
                    return b.x;

                return a.x + (y - a.y)
                        * (b.x - a.x) / dy;
            }
        }

        return points.get(points.size() - 1).x;
    }

    public static void draw(
            Mat frame,
            List<Point> displayLeft,
            List<Point> displayRight) {

        if (frame == null || frame.empty()) return;
        if (displayLeft == null || displayRight == null) return;
        if (displayLeft.size() < 2 || displayRight.size() < 2)
            return;

        ArrayList<Point> left =
                new ArrayList<>(displayLeft);

        ArrayList<Point> right =
                new ArrayList<>(displayRight);

        Collections.sort(left,
                (a, b) -> Double.compare(a.y, b.y));

        Collections.sort(right,
                (a, b) -> Double.compare(a.y, b.y));

        double lo = Math.max(
                left.get(0).y,
                right.get(0).y);

        double hi = Math.min(
                left.get(left.size() - 1).y,
                right.get(right.size() - 1).y);

        if (hi <= lo) return;

        final int n = 60;
        final int w = frame.cols();
        final int h = frame.rows();

        Point[] polygon = new Point[n * 2];

        for (int i = 0; i < n; i++) {
            double y = lo + (hi - lo)
                    * i / (n - 1.0);

            double lx = interpolateX(left, y);
            double rx = interpolateX(right, y);

            if (rx <= lx) return;

            double py = Math.max(
                    0, Math.min(h - 1, Math.round(y)));

            polygon[i] = new Point(
                    Math.max(0,
                            Math.min(w - 1, Math.round(lx))),
                    py);

            polygon[2 * n - 1 - i] = new Point(
                    Math.max(0,
                            Math.min(w - 1, Math.round(rx))),
                    py);
        }

        int minX = w - 1;
        int maxX = 0;
        int minY = h - 1;
        int maxY = 0;

        for (Point p : polygon) {
            minX = Math.min(minX, (int) p.x);
            maxX = Math.max(maxX, (int) p.x);
            minY = Math.min(minY, (int) p.y);
            maxY = Math.max(maxY, (int) p.y);
        }

        if (maxX <= minX || maxY <= minY)
            return;

        Rect bounds = new Rect(
                minX,
                minY,
                maxX - minX + 1,
                maxY - minY + 1);

        Mat roi = frame.submat(bounds);
        Mat overlay = roi.clone();
        Mat mask = Mat.zeros(
                roi.size(), CvType.CV_8UC1);

        Point[] local = new Point[polygon.length];

        for (int i = 0; i < polygon.length; i++) {
            local[i] = new Point(
                    polygon[i].x - minX,
                    polygon[i].y - minY);
        }

        MatOfPoint polygonMat =
                new MatOfPoint(local);

        Mat blended = new Mat();

        try {
            Imgproc.fillPoly(
                    mask,
                    Collections.singletonList(polygonMat),
                    new Scalar(255));

            overlay.setTo(LANE_FILL_RGBA, mask);

            Core.addWeighted(
                    overlay,
                    LANE_FILL_ALPHA,
                    roi,
                    1.0 - LANE_FILL_ALPHA,
                    0.0,
                    blended);

            blended.copyTo(roi, mask);

        } finally {
            blended.release();
            polygonMat.release();
            mask.release();
            overlay.release();
            roi.release();
        }
    }
}