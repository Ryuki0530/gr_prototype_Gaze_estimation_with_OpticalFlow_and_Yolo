import cv2
import numpy
import math

FLOW_THRESH = 5.0        # 視線移動とみなす速度[pixel/frame]
# ANGLE_THRESH = math.radians(45)


def select_target(detections, dx, dy, center, prev_idx=None):

    if len(detections) == 0:
        return None
    
    cand = []
    # 各クラスの中心座標や中心軸への距離量を算出
    for idx ,det in enumerate(detections):
        x1,y1,x2,y2 = det.xyxy[0].cpu().numpy()
    
        obj_cx, obj_cy = (x1 + x2) / 2, (y1 + y2) /2

        dist_center = math.hypot(obj_cx - center[0], obj_cy - center[1])

        cand.append((idx, obj_cx, obj_cy, dist_center))

    baseline_idx = min(cand, key = lambda c: c[3])[0]

    if math.hypot(dx, dy) < FLOW_THRESH and prev_idx is not None:
        return prev_idx