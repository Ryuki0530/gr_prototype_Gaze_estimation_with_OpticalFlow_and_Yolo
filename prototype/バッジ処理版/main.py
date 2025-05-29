import cv2
import numpy as np
import sys

def process_frame(prev_gray, frame, grid_size):
    draw_frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                         0.5, 3, 15, 3, 5, 1.2, 0)
    h, w = gray.shape
    for y in range(0, h, grid_size):
        for x in range(0, w, grid_size):
            fx, fy = flow[y:y+grid_size, x:x+grid_size].mean(axis=(0, 1))
            start_point = (x + grid_size // 2, y + grid_size // 2)
            end_point = (int(start_point[0] + fx), int(start_point[1] + fy))
            cv2.arrowedLine(draw_frame, start_point, end_point,
                            (0, 255, 0), 1, tipLength=0.3)
    return gray, draw_frame

# コマンドライン引数チェック
if len(sys.argv) < 3:
    print("使い方: python main.py 入力ファイル名.mp4 出力ファイル名.mp4")
    sys.exit(1)

input_path = sys.argv[1]
output_path = sys.argv[2]

# 動画読み込み
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print("入力動画が開けません")
    sys.exit(1)

# 動画情報取得
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 出力設定
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
if not out.isOpened():
    print("出力ファイルの作成に失敗しました")
    cap.release()
    sys.exit(1)

# 最初のフレーム取得
ret, prev_frame = cap.read()
if not ret:
    print("最初のフレームが取得できません")
    cap.release()
    out.release()
    sys.exit(1)

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
grid_size = 75

# メインループ
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray, processed_frame = process_frame(prev_gray, frame, grid_size)
    prev_gray = gray

    # カラーであることを保証
    if len(processed_frame.shape) == 2 or processed_frame.shape[2] != 3:
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)

    # サイズを出力指定に合わせる
    if processed_frame.shape[1] != width or processed_frame.shape[0] != height:
        processed_frame = cv2.resize(processed_frame, (width, height))

    out.write(processed_frame)

cap.release()
out.release()
print("バッチ処理が完了しました。")
