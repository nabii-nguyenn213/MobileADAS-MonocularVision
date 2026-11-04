"""Identity, hysteresis, geometry, filtering and pipeline safety regressions."""
import unittest
from unittest.mock import Mock, patch
import math
import numpy as np
from ai.src.fusion.vehicle_lane_tracker import VehicleLaneTracker
from ai.src.fusion.distance_smoother import DistanceSmoother
from ai.src.fusion.ego_lane_classifier import EgoLaneClassifier
from ai.src.fusion.adas_pipeline import ADASPipeline
from ai.src.fusion.adas_visualizer import draw_results, vehicle_label
from ai.src.distance_estimation.model import DistanceEstimator


def detection(x=400):
    return dict(bbox=[x,320,x+200,450],class_name='car',class_id=0,confidence=.95)


def lanes():
    return {side:[(x,float(y)) for y in np.linspace(300,560,30)]
            for side,x in [('left_ego',300),('right_ego',700)]}


def evidence(obs='inside',**kwargs):
    return dict(lane_observation=obs,lane_u=.5 if obs=='inside' else 1.2,
                lane_reason='valid',**kwargs)


class VehicleTrackingTests(unittest.TestCase):
    def test_order_independence_and_separate_history(self):
        t=VehicleLaneTracker()
        a,b=t.associate([detection(100),detection(340)],0.)
        t.smooth_distance(a,10,0.);t.smooth_distance(b,50,0.)
        b2,a2=t.associate([detection(345),detection(105)],1/30)
        self.assertEqual(a['track_id'],a2['track_id'])
        self.assertEqual(b['track_id'],b2['track_id'])
        self.assertNotEqual(a['track_id'],b['track_id'])
        self.assertEqual(t.smooth_distance(a2,10,1/30),10)
        self.assertEqual(t.smooth_distance(b2,50,1/30),50)

    def test_expiry_and_new_id(self):
        t=VehicleLaneTracker(track_timeout_seconds=.1)
        a=t.associate([detection()],0.)[0]
        t.smooth_distance(a,25,0.)
        t.associate([],.05);t.associate([],.11)
        self.assertNotIn(a['track_id'],t.tracks)
        b=t.associate([detection()],.12)[0]
        self.assertNotEqual(a['track_id'],b['track_id'])
        self.assertIsNone(t.tracks[b['track_id']].distance.smoothed)

    def test_missing_detection_resets_state_on_reacquisition(self):
        t=VehicleLaneTracker(inside_confirm_frames=1)
        a=t.associate([detection()],0.)[0]
        t.classify(a,evidence(),0.);t.smooth_distance(a,25,0.)
        self.assertEqual(t.associate([],1/30),[])
        b=t.associate([detection()],2/30)[0]
        self.assertEqual(a['track_id'],b['track_id'])
        self.assertEqual(b['tracking_reason'],'reacquired_reset')
        track=t.tracks[b['track_id']]
        self.assertEqual(track.lane_status,'unknown')
        self.assertIsNone(track.distance.smoothed)

    def test_ambiguous_match_does_not_transfer_state(self):
        t=VehicleLaneTracker()
        a=t.associate([detection(400),detection(410)],0.)
        for item in a:t.smooth_distance(item,20,0.)
        b=t.associate([detection(405),detection(405)],1/30)
        self.assertTrue(set(v['track_id'] for v in a).isdisjoint(v['track_id'] for v in b))
        self.assertTrue(all(t.tracks[v['track_id']].distance.smoothed is None for v in b))

    def test_class_or_implausible_scale_change_starts_new_id(self):
        for change in ('class','scale'):
            t=VehicleLaneTracker();a=t.associate([detection()],0.)[0]
            b=detection()
            if change=='class':b['class_name']='truck'
            else:b['bbox']=[0,0,1000,600]
            b=t.associate([b],1/30)[0]
            self.assertNotEqual(a['track_id'],b['track_id'])

    def test_reset_never_reuses_id(self):
        t=VehicleLaneTracker();a=t.associate([detection()],0.)[0]
        t.reset();b=t.associate([detection()],0.)[0]
        self.assertGreater(b['track_id'],a['track_id'])

    def test_invalid_box_not_tracked(self):
        t=VehicleLaneTracker()
        for box in (None,[1,2,3],[1,2,float('nan'),4],[4,3,2,1]):
            item=t.associate([dict(detection(),bbox=box)],0.)[0]
            self.assertIsNone(item['track_id'])


class HysteresisTests(unittest.TestCase):
    def setUp(self):
        self.t=VehicleLaneTracker();self.i=0

    def step(self,obs='inside',**kwargs):
        timestamp=self.i/30;self.i+=1
        item=self.t.associate([detection()],timestamp)[0]
        return self.t.classify(item,evidence(obs,**kwargs),timestamp)

    def inside(self):
        self.assertEqual(self.step()['lane_status'],'unknown')
        self.assertEqual(self.step()['lane_status'],'unknown')
        item=self.step();self.assertEqual(item['lane_status'],'inside');return item

    def test_initial_outside_confirmation(self):
        for _ in range(5):self.assertEqual(self.step('outside')['lane_status'],'unknown')
        self.assertEqual(self.step('outside')['lane_status'],'outside')

    def test_one_frame_error_keeps_color_but_gates_measurement(self):
        self.inside()
        item=self.step('outside')
        self.assertEqual(item['lane_status'],'inside')
        self.assertFalse(item['distance_eligible'])
        item=self.step();self.assertTrue(item['distance_eligible'])

    def test_real_lane_change_in_both_directions(self):
        self.inside()
        for _ in range(5):self.assertEqual(self.step('outside')['lane_status'],'inside')
        self.assertEqual(self.step('outside')['lane_status'],'outside')
        for _ in range(2):self.assertEqual(self.step()['lane_status'],'outside')
        self.assertEqual(self.step()['lane_status'],'inside')

    def test_uncertain_hold_is_bounded(self):
        self.inside()
        for _ in range(9):self.assertEqual(self.step('uncertain')['lane_status'],'inside')
        item=self.step('uncertain')
        self.assertEqual(item['lane_status'],'unknown')
        self.assertFalse(item['distance_eligible'])

    def test_alternating_errors_do_not_hold_incorrect_inside_forever(self):
        self.inside()
        for i in range(12):item=self.step('outside' if i%2 else 'uncertain')
        self.assertEqual(item['lane_status'],'unknown')

    def test_missing_geometry_hold_then_unknown(self):
        self.inside();item=self.step('unknown')
        self.assertEqual(item['lane_status'],'inside');self.assertFalse(item['distance_eligible'])
        for _ in range(10):item=self.step('unknown')
        self.assertEqual(item['lane_status'],'unknown')

    def test_tracking_failure_and_expired_geometry_immediate_unknown(self):
        self.inside();item=self.step('unknown',hard_invalid=True)
        self.assertEqual(item['lane_status'],'unknown')
        self.assertFalse(item['distance_eligible'])

    def test_implausible_association_revokes_inside(self):
        self.inside();item=self.step('outside',implausible_association=True)
        self.assertEqual(item['lane_status'],'unknown')
        for _ in range(5):item=self.step('outside',implausible_association=True)
        self.assertEqual(item['lane_status'],'outside')

    def test_distance_history_survives_brief_uncertainty(self):
        item=self.inside();self.t.smooth_distance(item,25,(self.i-1)/30)
        item=self.step('uncertain')
        self.t.smooth_distance(item,None,(self.i-1)/30)
        self.assertEqual(self.t.tracks[item['track_id']].distance.smoothed,25)
        item=self.step();self.assertTrue(item['distance_eligible'])

    def test_high_fps_confirmation_has_minimum_duration(self):
        t=VehicleLaneTracker(fps=60)
        for i in range(4):
            item=t.associate([detection()],i/60)[0]
            self.assertEqual(t.classify(item,evidence(),i/60)['lane_status'],'unknown')
        item=t.associate([detection()],4/60)[0]
        self.assertEqual(t.classify(item,evidence(),4/60)['lane_status'],'inside')


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.c=EgoLaneClassifier();self.c.prepare(lanes(),(1000,600))

    def observe_x(self,x):
        return self.c.observe([x-50,320,x+50,450],(1000,600))

    def test_normalized_regions(self):
        self.assertAlmostEqual(self.observe_x(500)['lane_u'],.5)
        self.assertEqual(self.observe_x(500)['lane_observation'],'inside')
        for x in (300,320,680,700):
            self.assertEqual(self.observe_x(x)['lane_observation'],'uncertain')
        for x in (250,750):
            self.assertEqual(self.observe_x(x)['lane_observation'],'outside')

    def test_no_extrapolation_or_clipped_box(self):
        self.assertEqual(self.c.observe([400,320,600,565],(1000,600))['lane_reason'],'bottom_outside_supported_y')
        self.assertTrue(self.c.observe(detection()['bbox'],(1000,600),crop_clipped=True)['hard_invalid'])

    def test_missing_and_crossing_geometry_unknown(self):
        for l in ({},dict(left_ego=lanes()['right_ego'],right_ego=lanes()['left_ego'])):
            self.c.prepare(l,(1000,600))
            self.assertEqual(self.observe_x(500)['lane_observation'],'unknown')

    def test_gradual_geometric_lane_change_keeps_id_and_confirms_outside(self):
        t=VehicleLaneTracker();c=EgoLaneClassifier();c.prepare(lanes(),(1000,600))
        ids=[];states=[]
        positions=[500]*10+list(range(520,761,20))+[760]*10
        for i,x in enumerate(positions):
            item=t.associate([detection(x-100)],i/30)[0]
            observation=c.observe(item['bbox'],(1000,600))
            item=t.classify(item,observation,i/30)
            ids.append(item['track_id']);states.append(item['lane_status'])
        self.assertEqual(len(set(ids)),1)
        self.assertEqual(states[2],'inside')
        self.assertEqual(states[-1],'outside')
        self.assertNotIn('unknown',states[2:])


class DistanceTests(unittest.TestCase):
    def test_median_then_ema(self):
        s=DistanceSmoother();self.assertEqual(s.update(25,0),25)
        self.assertAlmostEqual(s.update(27,.03),25.35)
        self.assertAlmostEqual(s.update(24,.06),25.2275)

    def test_invalid_predictions_not_fabricated(self):
        s=DistanceSmoother();s.update(25,0.)
        for v in (None,float('nan'),float('inf'),-1,0,'bad'):
            self.assertIsNone(s.update(v,.03))
        self.assertEqual(s.smoothed,25)
        self.assertEqual(list(s.history),[25])

    def test_extreme_outlier_and_reacquisition(self):
        s=DistanceSmoother();s.update(25,0.)
        self.assertIsNone(s.update(2500,.03))
        self.assertEqual(s.rejection,'extreme_ratio')
        self.assertEqual(s.update(25,.06),25)
        self.assertIsNone(s.update(2500,.09))
        self.assertEqual(s.update(2501,.12),2501)

    def test_rapid_closing_motion_is_accepted(self):
        s=DistanceSmoother();values=[]
        for i,d in enumerate((50,40,30,20,10)):
            values.append(s.update(d,i/30))
        self.assertTrue(all(v is not None and math.isfinite(v) for v in values))
        self.assertTrue(all(a>b for a,b in zip(values,values[1:])))

    def test_plausible_close_range_rapid_approach_not_rejected(self):
        s=DistanceSmoother();s.update(5,0.)
        self.assertIsNotNone(s.update(1,.2))  # 20 m/s closure, factor five in range.
        self.assertIsNone(s.rejection)

    def test_timeout_resets_history(self):
        s=DistanceSmoother(max_age_seconds=.2);s.update(25,0.)
        self.assertIsNone(s.update(None,.21));self.assertIsNone(s.smoothed)
        self.assertEqual(s.update(50,.22),50)

    def test_fluctuation_reduced(self):
        s=DistanceSmoother();raw=[25,27,24,26,25,23,27]*4
        filtered=[s.update(v,i/30) for i,v in enumerate(raw)]
        self.assertLess(np.std(np.diff(filtered)),np.std(np.diff(raw)))


class PipelineTrackingTests(unittest.TestCase):
    def build(self,**kwargs):
        lane=Mock();lane.extractor.diagnostics={}
        lane.predict.return_value=(np.zeros((600,1000),np.uint8),lanes())
        vehicle=Mock();vehicle.predict.side_effect=lambda f:[detection()]
        est=DistanceEstimator(focal_length_px=600,reference_resolution=(1000,600))
        est.estimate=Mock(wraps=est.estimate)
        p=ADASPipeline(lane_model=lane,vehicle_model=vehicle,distance_estimator=est,
                       full_frame_vehicles=True,**kwargs)
        p.tracking_extractor.extract=Mock(return_value=lanes())
        return p,lane,vehicle,est

    def test_confirmed_inside_on_skips_one_vehicle_pass_and_existing_calibration(self):
        p,lane,vehicle,est=self.build();frame=np.zeros((600,1000,3),np.uint8)
        ids=[]
        for i in range(11):
            image,result=p.process_frame(frame,i/30);v=result['vehicles'][0];ids.append(v['track_id'])
            self.assertTrue(all(result['lanes'].values()))
            if i<2:self.assertEqual(v['lane_status'],'unknown')
            else:
                self.assertEqual(v['lane_status'],'inside');self.assertEqual(v['raw_distance'],5.4)
                self.assertAlmostEqual(v['distance'],5.4)
                self.assertTrue(v['distance_fresh']);self.assertTrue(v['distance_for_alerts'])
                self.assertEqual(image[400,400].tolist(),[0,255,0])
        self.assertEqual(len(set(ids)),1)
        self.assertEqual(est.estimate.call_count,9)
        self.assertEqual(lane.predict.call_count,3)
        self.assertEqual(vehicle.predict.call_count,11)
        est.estimate.assert_called_with([400,320,600,450],(1000,600),'car')

    def test_uncertain_or_unknown_never_gets_new_distance(self):
        p,_,_,est=self.build();frame=np.zeros((600,1000,3),np.uint8)
        for i in range(3):p.process_frame(frame,i/30)
        self.assertEqual(est.estimate.call_count,1)
        with patch.object(p.classifier,'observe',return_value=evidence('uncertain')):
            _,result=p.process_frame(frame,3/30)
        v=result['vehicles'][0];self.assertEqual(v['lane_status'],'inside')
        self.assertIsNone(v['distance']);self.assertFalse(v['distance_fresh'])
        self.assertEqual(est.estimate.call_count,1)
        with patch.object(p.classifier,'observe',return_value=evidence('unknown',hard_invalid=True)):
            _,result=p.process_frame(frame,4/30)
        self.assertEqual(result['vehicles'][0]['lane_status'],'unknown')
        self.assertEqual(est.estimate.call_count,1)

    def test_invalid_distance_does_not_display_old_value(self):
        p,_,_,est=self.build();frame=np.zeros((600,1000,3),np.uint8)
        for i in range(3):p.process_frame(frame,i/30)
        est.estimate.return_value=float('nan');est.estimate.side_effect=None
        _,result=p.process_frame(frame,3/30)
        self.assertIsNone(result['vehicles'][0]['distance'])
        self.assertFalse(result['vehicles'][0]['distance_fresh'])

    def test_scene_change_resets_vehicle_evidence(self):
        p,_,_,_=self.build();frame=np.zeros((600,1000,3),np.uint8)
        for i in range(3):_,result=p.process_frame(frame,i/30)
        old_id=result['vehicles'][0]['track_id']
        _,result=p.process_frame(np.full_like(frame,255),3/30)
        self.assertNotEqual(old_id,result['vehicles'][0]['track_id'])
        self.assertEqual(result['vehicles'][0]['lane_status'],'unknown')

    def test_expired_lane_immediately_invalidates_green(self):
        p,_,_,est=self.build();frame=np.zeros((600,1000,3),np.uint8)
        for i in range(3):_,result=p.process_frame(frame,i/30)
        self.assertEqual(result['vehicles'][0]['lane_status'],'inside')
        rendered={side:[] for side in ('left_ego','right_ego')}
        meta=dict(reset_reason=None,sides={side:dict(timed_out=True) for side in rendered})
        with patch.object(p.lane_tracker,'update',return_value=(rendered,rendered,meta)):
            _,result=p.process_frame(frame,3/30)
        self.assertEqual(result['vehicles'][0]['lane_status'],'unknown')
        self.assertEqual(result['vehicles'][0]['lane_reason'],'lane_track_expired')
        self.assertEqual(est.estimate.call_count,1)

    def test_confirmed_outside_gray_and_no_distance(self):
        p,_,vehicle,est=self.build();frame=np.zeros((600,1000,3),np.uint8)
        vehicle.predict.side_effect=lambda f:[detection(700)]
        for i in range(8):image,result=p.process_frame(frame,i/30)
        item=result['vehicles'][0]
        self.assertEqual(item['lane_status'],'outside')
        self.assertIsNone(item['distance']);est.estimate.assert_not_called()
        self.assertEqual(image[400,700].tolist(),[128,128,128])

    def test_renderer_keeps_policy_and_no_polygons(self):
        t=VehicleLaneTracker();items=t.associate([detection(400),detection(50)],0.)
        inside=dict(items[0],lane_status='inside',distance=25.3)
        outside=dict(items[1],lane_status='outside',distance=None)
        with patch('cv2.fillPoly',side_effect=AssertionError('no polygons')):
            image=draw_results(np.zeros((600,1000,3),np.uint8),dict(lanes=lanes(),vehicles=[inside,outside],calibrated=True))
        self.assertEqual(image[400,400].tolist(),[0,255,0])
        self.assertEqual(image[400,50].tolist(),[128,128,128])
        self.assertIn('25.3 m',vehicle_label(inside));self.assertNotIn(' m',vehicle_label(outside))


if __name__=='__main__':unittest.main()
