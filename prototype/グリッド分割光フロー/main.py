import cv2
import numpy as np

def process_frame(prev_gray, frame, grid_size):
    """
    フレームを処理して光フローを計算し、グリッド分割とベクトル描画を行う。
    """
    # グレースケールに変換
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 光フローの計算
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # フレームの高さと幅を取得
    h, w = gray.shape

    # グリッドに分割して光フローの移動量を表示
    for y in range(0, h, grid_size):
        for x in range(0, w, grid_size):
            # グリッド内の光フローを取得
            fx, fy = flow[y:y+grid_size, x:x+grid_size].mean(axis=(0, 1))

            # ベクトルを描画
            end_x = int(x + grid_size / 2 + fx)
            end_y = int(y + grid_size / 2 + fy)
            cv2.arrowedLine(frame, (x + grid_size // 2, y + grid_size // 2), (end_x, end_y), (0, 255, 0), 1, tipLength=0.3)

    return gray, frame

# カメラの初期化
cap = cv2.VideoCapture(0)

# 光フロー計算用の初期フレーム
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# グリッドサイズ
grid_size = 20

while True:
    # 現在のフレームを取得
    ret, frame = cap.read()
    if not ret:
        break

    # フレームを処理
    prev_gray, processed_frame = process_frame(prev_gray, frame, grid_size)

    # 結果を表示
    cv2.imshow('Optical Flow Grid', processed_frame)

    # 'q'キーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# リソースの解放
cap.release()
cv2.destroyAllWindows()