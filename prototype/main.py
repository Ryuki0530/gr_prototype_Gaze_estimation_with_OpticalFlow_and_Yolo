import cv2
import numpy
import torch
from ultralytics import YOLO
from conn_cam import connCam
from calc_optical_flow import calcOpticalFlow
import numpy as np
import matplotlib.pyplot as plt
from yolo_viwer import viewYoloResult
import math

YOLO_MODEL = "yolov8n"
YOLO_ACTIVATE = True
OPTICAL_FLOW_ACTIVATE = True
OPTICAL_FLOW_PLOT = True
CAM_ID = 1
FLOW_THRESH = 5.0        # 視線移動とみなす速度[pixel/frame]
ANGLE_THRESH = math.radians(45)

# ---------------------------------------------------------------
# 1. select_target で返すのは Boxes オブジェクトではなく “index”
# ---------------------------------------------------------------
def select_target(detections, dx, dy, center, prev_idx=None):
    if len(detections) == 0:
        return None

    # ===== 前計算 =====
    cand = []  # 候補リスト: (idx, obj_cx, obj_cy, dist_center) を格納する
    for idx, det in enumerate(detections):
        # 各検出結果のバウンディングボックス座標を取得
        x1, y1, x2, y2 = det.xyxy[0].cpu().numpy()
        
        # バウンディングボックスの中心座標を計算
        obj_cx, obj_cy = (x1 + x2) / 2, (y1 + y2) / 2
        
        # 中心座標から画面中央までの距離を計算
        dist_center = math.hypot(obj_cx - center[0], obj_cy - center[1])
        
        # 候補リストに (インデックス, 中心座標, 中心からの距離) を追加
        cand.append((idx, obj_cx, obj_cy, dist_center))

    # --- 中央に最も近い ---
    baseline_idx = min(cand, key=lambda c: c[3])[0]

    # --- 速さが閾値未満 → 前回の対象を維持 ---
    if math.hypot(dx, dy) < FLOW_THRESH and prev_idx is not None:
        return prev_idx

    # --- フロー方向にある物体を探す ---
    best_idx, best_angle = baseline_idx, ANGLE_THRESH
    for idx, obj_cx, obj_cy, _ in cand:
        vec_obj  = (obj_cx-center[0], obj_cy-center[1])
        norm_obj = math.hypot(*vec_obj)
        if norm_obj == 0:
            return idx
        cosang = (dx*vec_obj[0] + dy*vec_obj[1]) / (math.hypot(dx,dy)*norm_obj)
        angle  = math.acos(max(min(cosang,1.0), -1.0))
        if angle < best_angle:
            best_idx, best_angle = idx, angle

    return best_idx

    
def main():
    
    frame_count = 0

    cap = connCam(on_device_camera_id=CAM_ID)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    model = YOLO(YOLO_MODEL)

    ret, prev_frame = cap.read()
    if not ret:
        print("Camera Error")
        return
    

    #光フローグラフの初期化
    if OPTICAL_FLOW_PLOT == True:
        plt.ion()#インタラクティブモード有効化
        fig, ax = plt.subplots()
        ax.set_title("OpticalFlowAVG (dx, dy)")
        ax.set_xlabel("Frame")
        ax.set_ylabel("OpticalFlowAVG(pixels)")
        x_data = []
        dx_data = []
        dy_data = []
        line_dx, = ax.plot([],[],label = "dx")
        line_dy, = ax.plot([],[],label = "dy")
        ax.legend()

    prev_idx = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        center = (frame_width/2, frame_height/2)
        target_det = None

        #光フロー計算
        if OPTICAL_FLOW_ACTIVATE:
            dx,dy = calcOpticalFlow(prev_frame,frame,plot_graph= True)
            cv2.putText(frame,f"dx: {dx:.2f}\ndy: {dy:.2f}",(10,30),cv2.FONT_HERSHEY_COMPLEX_SMALL,1.0,(0,0,0),2)

        # ── YOLO 推論と注視対象の描画 ──────────────────────
        if YOLO_ACTIVATE:
            yoloResults = model(frame)
            detections  = yoloResults[0].boxes

            # 注視対象は「detections の何番目か」で返す
            target_idx = select_target(detections, dx, dy, center, prev_idx)
            prev_idx = target_idx

            # 枠を描画：target_idx だけ緑、その他は赤
            for idx, det in enumerate(detections):
                colour = (0, 255, 0) if idx == target_idx else (0, 0, 255)
                viewYoloResult(frame, model, det, color=colour)



        #光フローの履歴プロット
        if OPTICAL_FLOW_PLOT and OPTICAL_FLOW_ACTIVATE:
            x_data.append(frame_count)
            dx_data.append(dx)
            dy_data.append(dy)

            line_dx.set_xdata(x_data)
            line_dx.set_ydata(dx_data)
            line_dy.set_xdata(x_data)
            line_dy.set_ydata(dy_data)

            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.01)

        #描画
        cv2.imshow("Yolo&OpticalFlow",frame)

        prev_frame = frame.copy()
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main()

