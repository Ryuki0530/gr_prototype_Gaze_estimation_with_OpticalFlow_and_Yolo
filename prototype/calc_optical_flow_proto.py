#!/usr/bin/env python
# coding: utf-8
"""
main.py ―― YOLO × Optical-Flow を用いた簡易 Gaze 推定デモ
--------------------------------------------------------------------
* 画面中央に最も近い物体を “注視対象” とみなす
* Farneback 法で平均フロー (dx,dy) を計算  
  └ フロー量がしきい値 FLOW_THRESH を超えたら  
    フロー方向にある物体へ注視対象を乗り換え
* 注視対象は緑枠，その他は赤枠で表示
* q キーで終了
--------------------------------------------------------------------
事前に必要な自作モジュール
    - conn_cam.py          : cv2.VideoCapture を返すラッパ
    - calc_optical_flow.py : Farneback で (dx,dy) を返す関数
    - yolo_viwer.py        : YOLO 検出結果を描画（color 引数対応）
"""

import math
import cv2
import torch          # noqa: F401（GPU 判定等で使うなら残してください）
import numpy as np    # noqa: F401（将来の拡張用）
import matplotlib.pyplot as plt
from ultralytics import YOLO

from conn_cam import connCam
from calc_optical_flow import calcOpticalFlow
from yolo_viwer  import viewYoloResult

# ────────────── 設定値 ───────────────────────────
YOLO_MODEL             = "yolov8n"   # 重み名 or パス
YOLO_ACTIVATE          = True        # 物体検出を行うか
OPTICAL_FLOW_ACTIVATE  = True        # 光フローを使うか
OPTICAL_FLOW_PLOT      = True        # グラフの ON/OFF
CAM_ID                 = 1           # 使用カメラ番号
FLOW_THRESH            = 5.0         # 注視移動判定(px/frame)
ANGLE_THRESH           = math.radians(45)  # 方向判定閾値

# ────────────── ヘルパ関数 ────────────────────────
def select_target(detections, dx, dy, center, prev_target_id=None):
    """
    物体リストから注視対象を選ぶ
    --------------------------------------------------------------
    detections : Ultralytics Boxes オブジェクト
    dx,dy      : 平均フロー
    center     : (cx,cy) 画面中心
    prev_target_id : 前フレームの注視 box 識別子（安定化用、無くても可）
    return     : 注視対象となった Boxes 要素（None の場合もあり）
    """
    if len(detections) == 0:
        return None

    # ① 距離と中心座標を前計算
    cand = []   # (det, obj_cx, obj_cy, dist_center)
    for det in detections:
        x1, y1, x2, y2 = det.xyxy[0].cpu().numpy()
        obj_cx, obj_cy = (x1 + x2) / 2, (y1 + y2) / 2
        dist_center    = math.hypot(obj_cx - center[0], obj_cy - center[1])
        cand.append((det, obj_cx, obj_cy, dist_center))

    # ② 基本ターゲット＝中央に最も近い物体
    baseline_det = min(cand, key=lambda c: c[3])[0]

    # ③ フロー速さが閾値未満 → 乗り換え不要
    speed = math.hypot(dx, dy)
    if speed < FLOW_THRESH:
        return baseline_det

    # ④ 閾値超え：フロー方向 ±ANGLE_THRESH にある物体を探す
    best_det, best_angle = None, ANGLE_THRESH
    for det, obj_cx, obj_cy, _ in cand:
        vec_obj  = (obj_cx - center[0], obj_cy - center[1])
        norm_obj = math.hypot(*vec_obj)
        if norm_obj == 0:      # 画面ど真ん中
            return det
        cos_ang = (dx * vec_obj[0] + dy * vec_obj[1]) / (speed * norm_obj)
        angle   = math.acos(max(min(cos_ang, 1.0), -1.0))
        if angle < best_angle:
            best_det, best_angle = det, angle

    return best_det if best_det is not None else baseline_det

# ────────────── メイン処理 ────────────────────────
def main() -> None:
    # 0) カメラ初期化
    cap = connCam(on_device_camera_id=CAM_ID)
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    center       = (frame_width / 2, frame_height / 2)

    # 1) YOLO モデル
    model = YOLO(YOLO_MODEL)

    # 2) 最初のフレーム取得
    ret, prev_frame = cap.read()
    if not ret:
        print("Camera Error")
        return

    # 3) 光フロー用グラフ
    if OPTICAL_FLOW_PLOT:
        plt.ion()
        fig, ax   = plt.subplots()
        ax.set_title("OpticalFlow AVG (dx, dy)")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Pixels / frame")
        x_data, dx_data, dy_data = [], [], []
        line_dx, = ax.plot([], [], label="dx")
        line_dy, = ax.plot([], [], label="dy")
        ax.legend()

    # 4) ループ前の初期値
    frame_count = 0
    dx = dy = 0.0
    target_det = None     # 直前の注視対象

    # ────── メインループ ──────
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 4-1) Optical Flow
        if OPTICAL_FLOW_ACTIVATE:
            dx, dy = calcOpticalFlow(prev_frame, frame, plot_graph=True)
            cv2.putText(frame,
                        f"dx:{dx:.1f}  dy:{dy:.1f}",
                        (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        1.0, (0, 0, 0), 2)

        # 4-2) YOLO 推論 & 注視対象決定
        if YOLO_ACTIVATE:
            yolo_results = model(frame)
            detections   = yolo_results[0].boxes

            target_det = select_target(
                detections, dx, dy, center,
                prev_target_id=getattr(target_det, 'id', None)
            )

            for det in detections:
                colour = (0, 255, 0) if det is target_det else (0, 0, 255)
                viewYoloResult(frame, model, det, color=colour)

        # 4-3) グラフ更新
        if OPTICAL_FLOW_PLOT and OPTICAL_FLOW_ACTIVATE:
            x_data.append(frame_count)
            dx_data.append(dx)
            dy_data.append(dy)

            line_dx.set_data(x_data, dx_data)
            line_dy.set_data(x_data, dy_data)
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.01)

        # 4-4) 画面表示
        cv2.imshow("YOLO & OpticalFlow Gaze Demo", frame)

        # 4-5) 次フレーム準備
        prev_frame  = frame.copy()
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 5) 終了処理
    cap.release()
    cv2.destroyAllWindows()

# ──────────── エントリポイント ────────────
if __name__ == "__main__":
    main()
