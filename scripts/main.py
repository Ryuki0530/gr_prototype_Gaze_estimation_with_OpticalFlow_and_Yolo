import cv2
import numpy
import torch
from ultralytics import YOLO
from conn_cam import connCam
from calc_optical_flow import calcOpticalFlowAvg
from calc_optical_flow import gridOpticalFlow
import numpy as np
import matplotlib.pyplot as plt
from yolo_viwer import viewYoloResult
import math
from target_selecter import select_target

# 各種パラメータ初期化
YOLO_ACTIVATE = False
OPTICAL_FLOW_AVG_ACTIVATE = False
OPTICAL_FLOW_AVG_PLOT = False
YOLO_MODEL = "yolov8n"
YOLO_ACTIVATE = False
OPTICAL_FLOW_AVG_ACTIVATE = False
OPTICAL_FLOW_GRID_ACTIVATE = False
OPTICAL_FLOW_AVG_PLOT = False
FLOW_THRESH = 0       # 視線移動とみなす速度[pixel/frame]
ANGLE_THRESH = math.radians(0)
GRIDSIZE = 100


# プロパティ
CAM_ID = 1
MODE = 1 
    # 0:光フロー ＋ YOLO
    # 1:グリッド分割 ＋ 光フロー

if MODE == 0:
    YOLO_MODEL = "yolov8n"
    YOLO_ACTIVATE = True
    OPTICAL_FLOW_AVG_ACTIVATE = True
    OPTICAL_FLOW_AVG_PLOT = True
    FLOW_THRESH = 5.0        # 視線移動とみなす速度[pixel/frame]
    ANGLE_THRESH = math.radians(45)
    
if MODE == 1:
    YOLO_ACTIVATE = False
    OPTICAL_FLOW_AVG_ACTIVATE = False
    OPTICAL_FLOW_AVG_PLOT = False
    OPTICAL_FLOW_GRID_ACTIVATE = True
    GRID_SIZE = 75


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
    if (OPTICAL_FLOW_AVG_PLOT == True):
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


    # メインループ準備
    prev_idx = None
    if OPTICAL_FLOW_GRID_ACTIVATE:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # メインループ
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        center = (frame_width/2, frame_height/2)
        target_det = None

        #光フロー計算
        if OPTICAL_FLOW_AVG_ACTIVATE:
            dx,dy = calcOpticalFlowAvg(prev_frame,frame,plot_graph= True)
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
        if OPTICAL_FLOW_AVG_PLOT and OPTICAL_FLOW_AVG_ACTIVATE:
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

        # グリッド分けを用いた方法
        if OPTICAL_FLOW_GRID_ACTIVATE:
            prev_gray, frame = gridOpticalFlow(prev_gray,frame,grid_size= GRID_SIZE)

        #描画
        cv2.imshow("GR_PROTOTYPE_GAZE_ESTIMATION",frame)

        prev_frame = frame.copy()
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main()

