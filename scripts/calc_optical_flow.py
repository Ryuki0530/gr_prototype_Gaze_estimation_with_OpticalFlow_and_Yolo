import cv2
import numpy as np

def calcOpticalFlow(prev_frame,curr_frame,plot_graph = True):
    #グレイスケール変換
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        0.5,   # ピラミッドスケール         分かりやすく言うと感度
        3,     # ピラミッドレイヤー数       大きな動きに対応し易くなる
        15,    # 窓サイズ                  でかいとノイズ耐性が上がる
        3,     # イテレーション回数         計算回数　上げると精度UP
        5,     # 多項式近似の拡大サイズ     上げると大きい動きに対応しやすくなり、小さい動きに弱くなる。
        1.2,   # 多項式近似の定数          上げるとノイズ耐性UP、小さい動きに鈍感に
        0      # オプションフラグ          
    )

    dx = np.mean(flow[..., 0])
    dy = np.mean(flow[..., 1])

    return dx,dy

