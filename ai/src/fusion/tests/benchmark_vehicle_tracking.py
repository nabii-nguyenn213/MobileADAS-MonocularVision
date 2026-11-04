"""Compare saved video diagnostics without another network pass.

Actual ID switches require annotated identities. This reports association churn
and a deterministic synthetic evidence sequence separately from video metrics.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
from . import test_vehicle_tracking as fixtures
from ai.src.fusion.vehicle_lane_tracker import VehicleLaneTracker


def step_std(samples):
    steps=[v-p for (t,p),(s,v) in zip(samples,samples[1:]) if 0 < s-t <= .05]
    return statistics.pstdev(steps) if len(steps)>=2 else None


def summarize(rows):
    # Use exactly the same diagnostic association for both logs. It never
    # changes the saved statuses or ranges and has no access to future frames.
    tracker=VehicleLaneTracker();previous={};counts=Counter();reasons=Counter()
    raw,smooth={},{};ids=set();unknown_frames=0
    for row in rows:
        timestamp=row['timestamp_seconds']
        if (row.get('lane_tracking') or {}).get('reset_reason'):
            tracker.reset();previous={}
        items=tracker.associate(row['vehicles'],timestamp)
        now={}
        unknown_frames+=any(v['lane_status']=='unknown' for v in items)
        for v in items:
            key=v['track_id'];ids.add(key)
            counts[v['lane_status']]+=1
            reasons[v.get('lane_reason','legacy_no_reason')]+=1
            if key in previous:
                before=previous[key]=='inside';after=v['lane_status']=='inside'
                counts['green_to_gray']+=before and not after
                counts['gray_to_green']+=not before and after
            now[key]=v['lane_status']
            if v.get('raw_distance') is not None:
                raw.setdefault(key,[]).append((timestamp,v['raw_distance']))
            if v.get('distance') is not None:
                smooth.setdefault(key,[]).append((timestamp,v['distance']))
        previous=now
    observations=sum(counts[k] for k in ('inside','outside','unknown'))
    def aggregate(series):
        values=[step_std(samples) for samples in series.values()]
        values=[v for v in values if v is not None]
        return statistics.mean(values) if values else None
    return dict(frames=len(rows),vehicle_observations=observations,
        green_to_gray=counts['green_to_gray'],gray_to_green=counts['gray_to_green'],
        unknown_observations_percent=100*counts['unknown']/max(observations,1),
        frames_with_unknown_percent=100*unknown_frames/max(len(rows),1),
        diagnostic_track_ids=len(ids),id_switches=None,
        id_switch_note='Ground-truth identities unavailable; new IDs are fragmentation proxies, not measured ID switches.',
        raw_distance_step_std_m=aggregate(raw),smoothed_distance_step_std_m=aggregate(smooth),
        lane_reasons=dict(reasons))


def synthetic():
    tracker=VehicleLaneTracker();legacy=[];stable=[];raw=[];smoothed=[];first_outside=None;return_inside=None;lost_unknown=None
    for i in range(100):
        obs='inside' if i<30 or i>=60 else 'outside'
        if i in (10,20):obs='outside'  # isolated observation errors
        if i==26:obs='uncertain'
        if i>=80:obs='unknown'  # missing lane geometry, not an ID loss
        item=tracker.associate([fixtures.detection()],i/30)[0]
        item=tracker.classify(item,fixtures.evidence(obs),i/30)
        legacy.append('inside' if obs=='inside' else 'outside' if obs=='outside' else 'unknown')
        stable.append(item['lane_status'])
        if item['distance_eligible']:
            value=25+(0,1,-1,2,-2)[i%5]
            value=float(value)
            filtered=tracker.smooth_distance(item,value,i/30)
            raw.append((i/30,value));smoothed.append((i/30,filtered))
        else:tracker.smooth_distance(item,None,i/30)
        if i>=30 and first_outside is None and item['lane_status']=='outside':first_outside=i
        if i>=60 and return_inside is None and item['lane_status']=='inside':return_inside=i
        if i>=80 and lost_unknown is None and item['lane_status']=='unknown':lost_unknown=i
    def changes(values):
        return dict(green_to_gray=sum(a=='inside' and b!='inside' for a,b in zip(values,values[1:])),
                    gray_to_green=sum(a!='inside' and b=='inside' for a,b in zip(values,values[1:])))
    return dict(description='Synthetic evidence only; not a driving-video accuracy score.',
        before=changes(legacy),after=changes(stable),id_switches=0,
        raw_distance_step_std_m=step_std(raw),smoothed_distance_step_std_m=step_std(smoothed),
        lane_change_start_frame=30,confirmed_outside_frame=first_outside,
        return_inside_start_frame=60,confirmed_inside_frame=return_inside,
        missing_geometry_start_frame=80,unknown_frame=lost_unknown)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before',type=Path,required=True);p.add_argument('--after',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    before=[json.loads(line) for line in args.before.read_text().splitlines()]
    after=[json.loads(line) for line in args.after.read_text().splitlines()]
    b=json.loads(args.before.with_suffix('.benchmark.json').read_text())
    a=json.loads(args.after.with_suffix('.benchmark.json').read_text())
    report=dict(before=summarize(before),after=summarize(after),synthetic=synthetic(),
                performance=dict(before_fps=b['pipeline_fps'],after_fps=a['pipeline_fps'],
                    before_membership_ms=b['mean']['membership_ms'],after_membership_ms=a['mean']['membership_ms'],
                    added_membership_ms=a['mean']['membership_ms']-b['mean']['membership_ms'],
                    fps_change_percent=100*(a['pipeline_fps']/b['pipeline_fps']-1)),
                same_lane_points_frames=sum(x['lanes']==y['lanes'] for x,y in zip(before,after)),
                before_lane_calls=b['lane_statistics']['measured_lane_inference_calls'],
                after_lane_calls=a['lane_statistics']['measured_lane_inference_calls'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
