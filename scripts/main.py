import cv2
import numpy
import torch
from ultralytics import YOLO
from conn_cam import connCam
from calc_optical_flow import calcOpticalFlow
import numpy as np
import matplotlib.pyplot as plt
from yolo_viwer import viewYoloResult

YOLO_MODEL = "yolov8n"
YOLO_ACTIVATE = False
OPTICAL_FLOW_ACTIVATE = True
OPTICAL_FLOW_PLOT = False
CAM_ID = 0
    
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

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        #Yolo推論実行
        if YOLO_ACTIVATE:
            yoloResults = model(frame)
            detections = yoloResults[0].boxes
            for detection in detections:
                viewYoloResult(frame,model,detection)

        #光フロー計算
        if OPTICAL_FLOW_ACTIVATE:
            dx,dy = calcOpticalFlow(prev_frame,frame,plot_graph= True)
            cv2.putText(frame,f"dx: {dx:.2f}\ndy: {dy:.2f}",(10,30),cv2.FONT_HERSHEY_COMPLEX_SMALL,1.0,(0,0,0),2)
        
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

