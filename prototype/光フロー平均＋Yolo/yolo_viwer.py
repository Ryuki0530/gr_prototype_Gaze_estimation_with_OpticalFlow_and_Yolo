# yolo_viwer.py などにある既存関数を書き換え
import cv2

def viewYoloResult(frame, model, det, color=(0, 0, 255), thickness=2):
    """
    1 つの検出結果をフレームに描画するユーティリティ
    --------------------------------------------------
    frame   : 描画対象の画像
    model   : YOLO モデル（クラス名取得用）
    det     : Ultralytics detection (Boxes 要素を 1 個渡す)
    color   : (B, G, R) のタプル   ← ★ 追加
    thickness : 枠線の太さ
    """
    x1, y1, x2, y2 = map(int, det.xyxy[0])       # 座標
    cls_id = int(det.cls[0])                     # クラス ID
    conf   = float(det.conf[0])                  # 信頼度

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    label = f"{model.names[cls_id]} {conf:.2f}"
    cv2.putText(frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
