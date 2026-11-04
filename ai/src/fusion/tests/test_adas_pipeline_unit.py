"""Membership, gating, calibration, rendering and single-inference regressions."""
import unittest
from unittest.mock import Mock,patch
import tempfile
import json
from pathlib import Path
import numpy as np
from ai.src.fusion.ego_lane_classifier import EgoLaneClassifier
from ai.src.fusion.adas_pipeline import ADASPipeline,load_camera_config
from ai.src.fusion.adas_visualizer import draw_results,vehicle_label
from ai.src.distance_estimation.model import DistanceEstimator


def lanes():
    return {side:[(x,int(y)) for y in np.linspace(300,560,30)]
            for side,x in [('left_ego',300),('right_ego',700)]}


def detection(box):
    return dict(bbox=box,class_name='car',class_id=0,confidence=.95)


class MembershipTests(unittest.TestCase):
    def setUp(self):
        self.c=EgoLaneClassifier(margin=8);self.c.prepare(lanes(),(1000,600))

    def test_inside(self):
        self.assertEqual(self.c.classify([400,320,600,450],(1000,600)),'inside')

    def test_outside(self):
        self.assertEqual(self.c.classify([50,320,150,450],(1000,600)),'outside')

    def test_boundary_uncertain(self):
        self.assertEqual(self.c.classify([250,320,350,450],(1000,600)),'unknown')

    def test_partial_overlap_uses_bottom_center(self):
        self.assertEqual(self.c.classify([100,320,400,450],(1000,600)),'outside')

    def test_missing_boundaries(self):
        for side in ('left_ego','right_ego'):
            l=lanes();l[side]=[];self.c.prepare(l,(1000,600))
            self.assertEqual(self.c.classify([400,320,600,450],(1000,600)),'unknown')

    def test_crossing(self):
        l=lanes();l['right_ego']=list(zip(np.linspace(800,200,30),np.linspace(300,560,30)))
        self.c.prepare(l,(1000,600));self.assertIsNone(self.c.boundaries)

    def test_invalid_fit(self):
        l=lanes();l['left_ego']=[(float('nan'),400)]*30
        self.c.prepare(l,(1000,600));self.assertIsNone(self.c.boundaries)

    def test_outside_supported_region(self):
        for bottom in (280,580):
            self.assertEqual(self.c.classify([400,100,600,bottom],(1000,600)),'unknown')

    def test_partial_and_invalid_boxes(self):
        for box in ([0,320,600,450],[400,320,1000,450],[600,320,400,450],[400,320,float('nan'),450],None):
            self.assertEqual(self.c.classify(box,(1000,600)),'unknown')

    def test_temporal_confirmation_and_missing_clears(self):
        c=EgoLaneClassifier(smooth=True)
        d=[detection([400,320,600,450])]
        self.assertEqual(c.classify_all(d,lanes(),(1000,600))[0]['lane_status'],'unknown')
        self.assertEqual(c.classify_all(d,lanes(),(1000,600))[0]['lane_status'],'inside')
        self.assertEqual(c.classify_all(d,{},(1000,600))[0]['lane_status'],'unknown')
        self.assertEqual(c.classify_all(d,lanes(),(1000,600))[0]['lane_status'],'unknown')


class PipelineTests(unittest.TestCase):
    def build(self,est=None,full=True):
        lane=Mock();lane.predict.return_value=(np.zeros((600,1000),np.uint8),lanes())
        vehicle=Mock();vehicle.predict.return_value=[detection([400,320,600,450]),detection([50,320,150,450]),detection([250,320,350,450])]
        est=est if est is not None else DistanceEstimator(focal_length_px=600,reference_resolution=(1000,600))
        est.estimate=Mock(wraps=est.estimate)
        p=ADASPipeline(vehicle_tracking_enabled=False,lane_model=lane,vehicle_model=vehicle,distance_estimator=est,full_frame_vehicles=full,lane_tracking_enabled=False,lane_detection_interval=1)
        return p,lane,vehicle,est

    def test_only_inside_calls_existing_estimator_once(self):
        p,lane,vehicle,est=self.build()
        _,r=p.process_frame(np.zeros((600,1000,3),np.uint8))
        self.assertEqual([d['lane_status'] for d in r['vehicles']],['inside','outside','unknown'])
        est.estimate.assert_called_once_with([400,320,600,450],(1000,600),'car')
        self.assertAlmostEqual(r['vehicles'][0]['distance'],5.4)
        self.assertEqual([v['distance'] for v in r['vehicles'][1:]],[None,None])
        lane.predict.assert_called_once();vehicle.predict.assert_called_once()
        self.assertEqual(p.distance_calls,1)

    def test_invalid_distance(self):
        for value in (None,float('nan'),float('inf'),-1,0):
            p,_,_,est=self.build();est.estimate.return_value=value;est.estimate.side_effect=None
            _,r=p.process_frame(np.zeros((600,1000,3),np.uint8))
            self.assertIsNone(r['vehicles'][0]['distance'])
            self.assertTrue(vehicle_label(r['vehicles'][0]).endswith('N/A'))

    def test_estimator_failure(self):
        p,_,_,est=self.build();est.estimate.side_effect=RuntimeError('failure')
        _,r=p.process_frame(np.zeros((600,1000,3),np.uint8));self.assertIsNone(r['vehicles'][0]['distance'])

    def test_no_automatic_calibration(self):
        p,_,_,est=self.build(DistanceEstimator())
        with patch.object(est,'calibrate',side_effect=AssertionError('must not calibrate')):
            _,r=p.process_frame(np.zeros((600,1000,3),np.uint8))
        self.assertFalse(r['calibrated']);self.assertIsNone(r['vehicles'][0]['distance'])
        self.assertEqual(load_camera_config(None),{})

    def test_existing_roi_preset_and_mapping(self):
        p,_,vehicle,est=self.build(full=False)
        self.assertEqual(p.roi,[.25,.25,.75,.65])
        vehicle.predict.return_value=[detection([150,170,350,220])]
        _,r=p.process_frame(np.zeros((600,1000,3),np.uint8))
        self.assertEqual(vehicle.predict.call_args.args[0].shape,(240,500,3))
        self.assertEqual(r['vehicles'][0]['bbox'],[400,320,600,370])
        est.estimate.assert_called_once_with([400,320,600,370],(1000,600),'car')

    def test_crop_truncated_is_unknown(self):
        p,_,vehicle,est=self.build(full=False)
        vehicle.predict.return_value=[detection([150,170,350,240])]
        _,r=p.process_frame(np.zeros((600,1000,3),np.uint8))
        self.assertEqual(r['vehicles'][0]['lane_status'],'unknown');est.estimate.assert_not_called()

    def test_saved_calibration_loads_without_recalibration(self):
        from ai.src.fusion.adas_pipeline import PRESET
        config=load_camera_config(PRESET)
        est=DistanceEstimator(**config)
        self.assertEqual(est.reference_resolution,(1280,961))
        self.assertAlmostEqual(est.focal_length_px,616.6666666666666)
        # Numeric behavior of the existing estimator; not camera applicability proof.
        self.assertAlmostEqual(est.estimate([100,100,322,300],(1280,961),'car'),5.)

    def test_invalid_camera_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'camera.json';path.write_text(json.dumps(dict(focal_length_px=600)))
            with self.assertRaises(ValueError):load_camera_config(path)

    def test_visual_colors_and_labels(self):
        frame=np.zeros((600,1000,3),np.uint8)
        vehicles=[dict(detection(box),lane_status=status,distance=12.3) for box,status in [
            ([400,320,600,450],'inside'),([50,320,150,450],'outside'),([750,320,850,450],'unknown')]]
        result=dict(lanes=lanes(),vehicles=vehicles,calibrated=True)
        with patch('cv2.fillPoly',side_effect=AssertionError('no polygons')):
            image=draw_results(frame,result)
        self.assertEqual(image[400,400].tolist(),[0,255,0])
        self.assertEqual(image[400,50].tolist(),[128,128,128])
        self.assertEqual(image[400,750].tolist(),[128,128,128])
        self.assertEqual(image[300,300].tolist(),[0,255,0])
        self.assertEqual(image[300,700].tolist(),[0,0,255])
        self.assertEqual(image[500,500].tolist(),[0,0,0])
        self.assertIn('12.3 m',vehicle_label(vehicles[0]))
        self.assertNotIn('12.3',vehicle_label(vehicles[1]));self.assertNotIn('12.3',vehicle_label(vehicles[2]))
        self.assertEqual(frame.sum(),0)


if __name__=='__main__':unittest.main()
