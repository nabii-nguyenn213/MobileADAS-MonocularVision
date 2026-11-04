"""Width-based optical-axis depth; no detector or ML dependencies."""
import math


class DistanceEstimator:
    # Approximate placeholders, not measured dimensions of detected vehicles.
    DEFAULT_WIDTHS = {"car": 1.8, "truck": 2.5, "bus": 2.5,
                      "motorcycle": 0.8, "pickup_truck": 2.0}
    ALIASES = {"motorbike": "motorcycle"}

    def __init__(self, *, focal_length_px=None, horizontal_fov_deg=None,
                 reference_resolution=None, vehicle_widths=None):
        if focal_length_px is not None and horizontal_fov_deg is not None:
            raise ValueError("Supply focal length or horizontal FOV, not both")
        if focal_length_px is not None:
            self._positive(focal_length_px, "focal_length_px")
        if horizontal_fov_deg is not None:
            if not math.isfinite(horizontal_fov_deg) or not 0 < horizontal_fov_deg < 180:
                raise ValueError("horizontal_fov_deg must be between 0 and 180")
        self.focal_length_px = focal_length_px
        self.horizontal_fov_deg = horizontal_fov_deg
        self.reference_resolution = (self._resolution(reference_resolution)
                                     if reference_resolution is not None else None)
        self.vehicle_widths = dict(self.DEFAULT_WIDTHS)
        for name, width in (vehicle_widths or {}).items():
            self._positive(width, "vehicle width")
            self.vehicle_widths[self._class(name)] = float(width)

    @staticmethod
    def _positive(value, name):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")

    @classmethod
    def _resolution(cls, resolution):
        width, height = map(float, resolution)
        cls._positive(width, "image width")
        cls._positive(height, "image height")
        return width, height

    @classmethod
    def _class(cls, name):
        name = str(name).strip().lower()
        return cls.ALIASES.get(name, name)

    def effective_focal_length(self, image_resolution, *, source_resolution=None,
                               crop_box=None):
        """Sizes are (width, height). Crop is xyxy in the uncropped source.

        Pipeline: reference image -> source resize -> crop -> output resize.
        An explicit focal length without a reference size is in source pixels.
        FOV describes the full uncropped source, or reference if supplied.
        Letterbox padding must be removed before using this API.
        """
        iw, ih = self._resolution(image_resolution)
        sw, sh = self._resolution(source_resolution or image_resolution)
        rw, rh = self.reference_resolution or (sw, sh)
        if crop_box is None:
            cw = sw
        else:
            x1, y1, x2, y2 = map(float, crop_box)
            if not all(map(math.isfinite, (x1, y1, x2, y2))) or not (
                0 <= x1 < x2 <= sw and 0 <= y1 < y2 <= sh
            ):
                raise ValueError("crop_box must lie within source_resolution")
            cw = x2 - x1
        fx = self.focal_length_px
        if fx is None:
            if self.horizontal_fov_deg is None:
                return None
            fx = rw / (2 * math.tan(math.radians(self.horizontal_fov_deg) / 2))
        return float(fx) * sw / rw * iw / cw

    def estimate(self, bbox, image_resolution, vehicle_class, *,
                 source_resolution=None, crop_box=None):
        """Return meters or None for uncalibrated, unknown or invalid boxes.

        Reject boxes touching ANY image boundary: a truncated physical width
        would systematically overestimate depth. Coordinates are output pixels.
        """
        fx = self.effective_focal_length(
            image_resolution, source_resolution=source_resolution, crop_box=crop_box)
        width = self.vehicle_widths.get(self._class(vehicle_class))
        iw, ih = self._resolution(image_resolution)
        try:
            x1, y1, x2, y2 = map(float, bbox)
        except (ValueError, TypeError):
            return None
        if not all(map(math.isfinite, (x1, y1, x2, y2))) or not (
            0 < x1 < x2 < iw and 0 < y1 < y2 < ih
        ):
            return None
        if fx is None or width is None:
            return None
        distance = fx * width / (x2 - x1)
        return distance if math.isfinite(distance) else None

    def calibrate(self, bbox, image_resolution, known_distance_m,
                  vehicle_class="car", *, real_width_m=None):
        """Set focal length at this resolution from an uncropped reference.

        Use measured vehicle width when possible. Calibration replaces FOV.
        """
        self._positive(known_distance_m, "known_distance_m")
        width = (real_width_m if real_width_m is not None
                 else self.vehicle_widths.get(self._class(vehicle_class)))
        if width is None:
            raise ValueError("Unknown vehicle class; supply real_width_m")
        self._positive(width, "real_width_m")
        iw, ih = self._resolution(image_resolution)
        x1, y1, x2, y2 = map(float, bbox)
        if not all(map(math.isfinite, (x1, y1, x2, y2))) or not (
            0 < x1 < x2 < iw and 0 < y1 < y2 < ih
        ):
            raise ValueError("Calibration requires an unclipped valid box")
        self.focal_length_px = known_distance_m * (x2 - x1) / width
        self.horizontal_fov_deg = None
        self.reference_resolution = (iw, ih)
        return self.focal_length_px
