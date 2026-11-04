import math
import unittest

from ai.src.distance_estimation import DistanceEstimator


class DistanceTests(unittest.TestCase):
    def setUp(self):
        self.est = DistanceEstimator(focal_length_px=1000,
                                     reference_resolution=(1000, 600))
        self.box = [100, 100, 200, 200]

    def test_known_focal(self):
        self.assertAlmostEqual(self.est.estimate(self.box, (1000, 600), 'car'), 18)

    def test_fov(self):
        est = DistanceEstimator(horizontal_fov_deg=90)
        self.assertAlmostEqual(est.effective_focal_length((1000, 600)), 500)
        self.assertAlmostEqual(est.estimate(self.box, (1000, 600), 'car'), 9)

    def test_inverse_width(self):
        self.assertGreater(self.est.estimate([100, 100, 150, 200], (1000, 600), 'car'), 18)

    def test_resize(self):
        self.assertAlmostEqual(self.est.estimate([50, 50, 100, 100], (500, 300), 'car'), 18)

    def test_calibration(self):
        est = DistanceEstimator()
        self.assertAlmostEqual(est.calibrate(self.box, (1000, 600), 18, real_width_m=1.8), 1000)
        self.assertAlmostEqual(est.estimate(self.box, (1000, 600), 'car'), 18)
        self.assertAlmostEqual(est.estimate([50, 50, 100, 100], (500, 300), 'car'), 18)

    def test_classes(self):
        for name, distance in [('car', 18), ('truck', 25), ('bus', 25),
                               ('motorcycle', 8), ('motorbike', 8), ('pickup_truck', 20)]:
            self.assertAlmostEqual(self.est.estimate(self.box, (1000, 600), name), distance)
        est = DistanceEstimator(focal_length_px=1000, vehicle_widths={'motorbike': 1})
        self.assertEqual(est.estimate(self.box, (1000, 600), 'motorcycle'), 10)

    def test_invalid(self):
        for box in [None, [], [1, 1, 1, 2], [2, 1, 1, 2], [1, 2, 3, 1],
                    [1, 1, math.nan, 2], [1, 1, math.inf, 2]]:
            self.assertIsNone(self.est.estimate(box, (1000, 600), 'car'))

    def test_clipped(self):
        for box in [[0, 10, 100, 100], [-10, 10, 100, 100],
                    [900, 10, 1000, 100], [10, 0, 100, 100],
                    [10, 10, 100, 601], [1100, 10, 1200, 100]]:
            self.assertIsNone(self.est.estimate(box, (1000, 600), 'car'))

    def test_unknown_uncalibrated(self):
        self.assertIsNone(self.est.estimate(self.box, (1000, 600), 'person'))
        self.assertIsNone(DistanceEstimator().estimate(self.box, (1000, 600), 'car'))

    def test_crop_and_resize(self):
        # Crop 1000 -> 500 pixels, then resize 500 -> 250 pixels: fx=500.
        kw = dict(source_resolution=(1000, 600), crop_box=(200, 0, 700, 600))
        self.assertAlmostEqual(self.est.estimate([50, 10, 100, 100], (250, 300), 'car', **kw), 18)
        fov = DistanceEstimator(horizontal_fov_deg=90)
        self.assertAlmostEqual(fov.effective_focal_length((500, 600), **kw), 500)
        with self.assertRaises(ValueError):
            self.est.estimate(self.box, (1000, 600), 'car', crop_box=(-1, 0, 500, 600))
        with self.assertRaises(ValueError):
            self.est.estimate(self.box, (0, 600), 'car')

    def test_bad_configuration(self):
        for kwargs in [dict(focal_length_px=0), dict(horizontal_fov_deg=180),
                       dict(focal_length_px=1, horizontal_fov_deg=90),
                       dict(vehicle_widths={'car': -1})]:
            with self.assertRaises(ValueError):
                DistanceEstimator(**kwargs)
