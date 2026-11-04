"""Scheduling, continuous convergence, expiry and distance safety regressions."""
import unittest
from unittest.mock import Mock
import numpy as np
from ai.src.fusion.adas_pipeline import ADASPipeline
from ai.src.distance_estimation.model import DistanceEstimator
from ai.src.lane_detection.temporal_tracker import TemporalLaneTracker
from ai.src.lane_detection.tests.test_temporal_tracker import observations


class IntervalTests(unittest.TestCase):
    def test_scheduling_and_safe_membership_on_skips(self):
        for interval in (1,5,10):
            lane=Mock();lane.extractor.diagnostics={}
            lane.predict.return_value=(np.zeros((600,1000),np.uint8),observations())
            vehicle=Mock();vehicle.predict.return_value=[dict(bbox=[400,320,600,450],class_name='car',confidence=.9)]
            estimator=DistanceEstimator(focal_length_px=600,reference_resolution=(1000,600))
            estimator.estimate=Mock(wraps=estimator.estimate)
            pipeline=ADASPipeline(vehicle_tracking_enabled=False,lane_model=lane,vehicle_model=vehicle,distance_estimator=estimator,
                                  full_frame_vehicles=True,lane_detection_interval=interval)
            pipeline.tracking_extractor.extract=Mock(return_value=observations())
            frame=np.zeros((600,1000,3),np.uint8)
            for i in range(31):
                _,result=pipeline.process_frame(frame,i/30)
                self.assertEqual(result['lane_detection_ran'],i%interval==0)
                self.assertTrue(all(result['lanes'].values()))
                self.assertEqual(result['vehicles'][0]['lane_status'],'inside')
            self.assertEqual(lane.predict.call_count,1+30//interval)
            self.assertEqual(vehicle.predict.call_count,31)
            self.assertEqual(estimator.estimate.call_count,31)
            pipeline.reset()
            pipeline.process_frame(frame,0.)
            self.assertEqual(pipeline.lane_inference_calls,1)

    def test_skips_converge_without_refreshing_observation(self):
        tracker=TemporalLaneTracker(acquisition_hits=1,classification_max_age=.3)
        tracker.update(observations(),(1000,600),0.)
        visible,_,_=tracker.update(observations(320,700),(1000,600),.1)
        self.assertEqual(visible['left_ego'][0][0],305)
        visible,safe,meta=tracker.update({},(1000,600),.2,detection_ran=False)
        self.assertGreater(visible['left_ego'][0][0],305)
        self.assertTrue(safe['left_ego'])
        self.assertAlmostEqual(meta['sides']['left_ego']['age_seconds'],.1)
        visible,safe,_=tracker.update({},(1000,600),.3,detection_ran=True)
        self.assertTrue(visible['left_ego']);self.assertFalse(safe['left_ego'])
        visible,safe,_=tracker.update({},(1000,600),.4,detection_ran=False)
        self.assertFalse(safe['left_ego'])
        visible,_,_=tracker.update({},(1000,600),.61,detection_ran=False)
        self.assertFalse(any(visible.values()))

    def test_interval_hold_validation(self):
        with self.assertRaises(ValueError):
            ADASPipeline(lane_detection_interval=20,video_fps=30,lane_max_hold_seconds=.5)
        with self.assertRaises(ValueError):
            ADASPipeline(lane_detection_interval=5,lane_tracking_enabled=False)


if __name__=='__main__':unittest.main()
