"""Points-only lanes and fixed inside/outside vehicle colors."""
import cv2
import math
import numpy as np


def vehicle_label(vehicle):
    text = f"{vehicle['class_name']} {vehicle['confidence']:.2f}"
    status = vehicle['lane_status']
    if status == 'inside':
        distance = vehicle.get('distance')
        valid = isinstance(distance,(int,float)) and math.isfinite(distance) and distance > 0
        text += f' | {distance:.1f} m' if valid else ' | N/A'
    elif status == 'unknown':
        text += ' | Lane unknown'
    return text


def draw_results(frame, results, point_radius=4, thickness=2):
    output = frame.copy()
    for side,color in [('left_ego',(0,255,0)),('right_ego',(0,0,255))]:
        for x,y in results['lanes'].get(side,[]):
            if np.isfinite([x,y]).all() and 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                cv2.circle(output,(round(x),round(y)),point_radius,color,-1,cv2.LINE_AA)
    occupied = []
    for vehicle in results['vehicles']:
        try:
            box = np.asarray(vehicle['bbox'],dtype=float)
            if box.shape != (4,) or not np.isfinite(box).all():
                continue
            x1,y1,x2,y2 = map(round,box)
        except (TypeError,ValueError):
            continue
        color = (0,255,0) if vehicle['lane_status'] == 'inside' else (128,128,128)
        cv2.rectangle(output,(x1,y1),(x2,y2),color,thickness)
        text = vehicle_label(vehicle)
        (tw,th),baseline = cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,.5,1)
        tx = max(0,min(x1,frame.shape[1]-tw-4))
        ty = max(th+4,min(y1-6,frame.shape[0]-baseline-2))
        for _ in range(5):
            if not any(abs(ty-py) < th+baseline+4 and tx < px+pw and px < tx+tw for px,py,pw in occupied):
                break
            ty = min(frame.shape[0]-baseline-2,ty+th+baseline+4)
        occupied.append((tx,ty,tw))
        cv2.putText(output,text,(tx,ty),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),3,cv2.LINE_AA)
        cv2.putText(output,text,(tx,ty),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1,cv2.LINE_AA)
    fps = results.get('processing_fps')
    if fps is not None:
        cv2.putText(output,f'Processing FPS: {fps:.1f}',(12,30),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2,cv2.LINE_AA)
    if not results.get('calibrated',False):
        cv2.putText(output,'Distance calibration unavailable',(12,58),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,255),2,cv2.LINE_AA)
    return output
