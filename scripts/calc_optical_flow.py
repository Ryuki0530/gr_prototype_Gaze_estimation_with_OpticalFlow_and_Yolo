import cv2
import numpy as np

def calcOpticalFlowAvg(prev_frame,curr_frame,plot_graph = True):
    #グレイスケール変換
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        0.3,   # ピラミッドスケール         分かりやすく言うと感度
        6,     # ピラミッドレイヤー数       大きな動きに対応し易くなる
        30,    # 窓サイズ                  でかいとノイズ耐性が上がる
        2,     # イテレーション回数         計算回数　上げると精度UP
        12,     # 多項式近似の拡大サイズ     上げると大きい動きに対応しやすくなり、小さい動きに弱くなる。
        2,   # 多項式近似の定数          上げるとノイズ耐性UP、小さい動きに鈍感に
        0      # オプションフラグ          
    )

    dx = np.mean(flow[..., 0])
    dy = np.mean(flow[..., 1])

    return dx,dy

def gridOpticalFlow(prev_gray_frame, curr_frame,grid_size = 20):

    curr_gray_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray_frame,
        curr_gray_frame,
        None,
        0.5,
        3,
        15,
        3,5,
        1.2,0
    )

    h,w = curr_gray_frame.shape

    for y in range(0,h,grid_size):
        for x in range(0,w,grid_size):
            fx, fy = flow[y:y+grid_size, x:x+grid_size].mean(axis=(0, 1))
