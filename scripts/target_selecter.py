import cv2
import numpy
import math

# ANGLE_THRESH = math.radians(45)


def select_target(detections, dx, dy, center, prev_idx=None ,FLOW_THRESH = 5.0):

    ANGLE_THRESH = math.radians(20)

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
    # --- フロー方向にある物体を探す ---
    # 基準となるインデックスと角度を初期化
    best_idx, best_angle = baseline_idx, ANGLE_THRESH

    # 候補リストをループして、フロー方向に最も近い物体を探す
    for idx, obj_cx, obj_cy, _ in cand:
        # 物体の中心座標から画面中央へのベクトルを計算
        vec_obj  = (obj_cx-center[0], obj_cy-center[1])
        
        # ベクトルの大きさ（ノルム）を計算
        norm_obj = math.hypot(*vec_obj)
        
        # ノルムがゼロの場合（物体が画面中央にある場合）はその物体を選択
        if norm_obj == 0:
            return idx
        
        # フロー方向（dx, dy）と物体方向（vec_obj）の角度を計算
        # 内積を利用して cosθ を求め、acos で角度に変換
        cosang = (dx*vec_obj[0] + dy*vec_obj[1]) / (math.hypot(dx,dy)*norm_obj)
        angle  = math.acos(max(min(cosang,1.0), -1.0))  # cosθ の範囲を [-1, 1] に制限
        
        # 角度が現在の最小角度より小さい場合、最適なインデックスと角度を更新
        if angle < best_angle:
            best_idx, best_angle = idx, angle

    # 最終的に選択されたインデックスを返す
    return best_idx


def select_target2(
    detections, dx, dy, center, prev_idx=None, 
    FLOW_THRESH=5.0, 
    shift_state=None, 
    shift_alpha=0.1, 
    shift_scale=1
):
    """
    簡略化版ターゲット選択アルゴリズム
    - 中心点の移動は常に処理
    - 対象変更は光フロー移動量が閾値を超えた場合のみ
    - shift_state: [shift_x, shift_y] を保持する外部リスト（呼び出し元で管理）
    """
    if len(detections) == 0:
        return None, shift_state, center

    # 初期化
    if shift_state is None:
        shift_state = [0.0, 0.0]

    # 光フロー移動量
    flow_mag = math.hypot(dx, dy)

    # フロー方向ベクトル（正規化）
    if flow_mag > 0:
        norm_dx = dx / flow_mag
        norm_dy = dy / flow_mag
    else:
        norm_dx = 0.0
        norm_dy = 0.0

    # 中心点の移動は常に処理
    shift_state[0] += norm_dx * flow_mag * shift_scale
    shift_state[1] += norm_dy * flow_mag * shift_scale

    # 反発的に中心点へ戻す
    shift_state[0] -= shift_alpha * shift_state[0]
    shift_state[1] -= shift_alpha * shift_state[1]

    # ずらした中心点
    shifted_center = (center[0] + shift_state[0], center[1] + shift_state[1])

    # 対象変更は光フロー移動量が閾値を超えた場合のみ
    if flow_mag > FLOW_THRESH or prev_idx is None or prev_idx >= len(detections):
        # 対象物体の中心座標を計算し、ずらした中心点に最も近いものを選択
        cand = []
        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = det.xyxy[0].cpu().numpy()
            obj_cx, obj_cy = (x1 + x2) / 2, (y1 + y2) / 2
            dist = math.hypot(obj_cx - shifted_center[0], obj_cy - shifted_center[1])
            cand.append((idx, dist))
        best_idx = min(cand, key=lambda c: c[1])[0]
        return best_idx, shift_state, shifted_center
    else:
        # 対象変更しない場合は前回のインデックスを維持
        return prev_idx, shift_state, shifted_center





