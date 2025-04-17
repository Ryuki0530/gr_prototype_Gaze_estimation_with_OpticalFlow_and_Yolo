import cv2
import numpy as np
import torch
import argparse
from ultralytics import YOLO

def calculate_optical_flow(prev_frame, current_frame):
    """光フローを計算する関数"""
    # グレースケールに変換
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    
    # Shi-Tomasiコーナー検出
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
    
    if prev_pts is None:
        return None, None
    
    # Lucas-Kanade法で光フローを計算
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, current_gray, prev_pts, None)
    
    # 有効な点のみを抽出
    good_prev = prev_pts[status == 1]
    good_next = next_pts[status == 1]
    
    return good_prev, good_next

def estimate_camera_movement(prev_pts, next_pts, frame_width, frame_height):
    """カメラの動きを推定する関数"""
    if prev_pts is None or next_pts is None or len(prev_pts) < 5:
        return "STABLE", (0, 0)
    
    # 動きベクトルを計算
    movements = next_pts - prev_pts
    avg_movement = np.mean(movements, axis=0)
    
    # 閾値を設定（フレームサイズの5%以上の動きを「大きな動き」と判断）
    threshold_x = frame_width * 0.05
    threshold_y = frame_height * 0.05
    
    # 動きの方向を判断
    if abs(avg_movement[0]) > threshold_x or abs(avg_movement[1]) > threshold_y:
        # 水平方向の動き
        if abs(avg_movement[0]) > abs(avg_movement[1]):
            if avg_movement[0] > 0:
                direction = "RIGHT"
            else:
                direction = "LEFT"
        # 垂直方向の動き
        else:
            if avg_movement[1] > 0:
                direction = "DOWN"
            else:
                direction = "UP"
        return direction, avg_movement
    else:
        return "STABLE", avg_movement

def select_initial_target(detections, frame_width, frame_height):
    """画像中心に最も近い物体を選択する関数"""
    center_x, center_y = frame_width / 2, frame_height / 2
    min_distance = float('inf')
    closest_object = None
    
    for detection in detections:
        x1, y1, x2, y2 = detection.xyxy[0].tolist()  # バウンディングボックスの座標
        obj_center_x = (x1 + x2) / 2
        obj_center_y = (y1 + y2) / 2
        
        # 中心点からの距離を計算
        distance = np.sqrt((center_x - obj_center_x)**2 + (center_y - obj_center_y)**2)
        
        if distance < min_distance:
            min_distance = distance
            closest_object = detection
    
    return closest_object

def select_next_target(detections, camera_direction, current_target_center, frame_width, frame_height):
    """カメラの動きに基づいて次のターゲットを選択する関数"""
    if not detections:
        return None
    
    # 中心座標における優先方向を決定
    center_x, center_y = frame_width / 2, frame_height / 2
    
    best_candidate = None
    best_score = float('-inf')
    
    for detection in detections:
        x1, y1, x2, y2 = detection.xyxy[0].tolist()
        obj_center_x = (x1 + x2) / 2
        obj_center_y = (y1 + y2) / 2
        
        # カメラの動きに基づくスコアリング
        score = 0
        
        if camera_direction == "RIGHT":
            # 中心より右側にある物体を優先
            score = obj_center_x - center_x
        elif camera_direction == "LEFT":
            # 中心より左側にある物体を優先
            score = center_x - obj_center_x
        elif camera_direction == "UP":
            # 中心より上側にある物体を優先
            score = center_y - obj_center_y
        elif camera_direction == "DOWN":
            # 中心より下側にある物体を優先
            score = obj_center_y - center_y
        
        # 現在のターゲットに近い場合はスコアを下げる（新しいターゲットを優先）
        if current_target_center is not None:
            curr_x, curr_y = current_target_center
            distance_to_current = np.sqrt((curr_x - obj_center_x)**2 + (curr_y - obj_center_y)**2)
            if distance_to_current < 50:  # 近すぎる場合は同じ物体の可能性がある
                score -= 1000
        
        if score > best_score:
            best_score = score
            best_candidate = detection
    
    return best_candidate

def is_target_following_camera(target_center, prev_target_center, camera_movement, threshold=0.7):
    """ターゲットがカメラの動きに追従しているか判定する関数"""
    if target_center is None or prev_target_center is None:
        return False
    
    target_movement = np.array(target_center) - np.array(prev_target_center)
    
    # カメラの動きとターゲットの動きの方向が似ているか確認
    camera_magnitude = np.linalg.norm(camera_movement)
    target_magnitude = np.linalg.norm(target_movement)
    
    if camera_magnitude == 0 or target_magnitude == 0:
        return False
    
    # コサイン類似度を計算
    cos_similarity = np.dot(camera_movement, target_movement) / (camera_magnitude * target_magnitude)
    
    # ターゲットの動きの大きさがカメラの動きの大きさに近いか確認
    magnitude_ratio = target_magnitude / camera_magnitude if camera_magnitude > 0 else 0
    
    # 方向が似ていて、大きさも近い場合は追従していると判断
    return cos_similarity > threshold and 0.5 < magnitude_ratio < 1.5

def main():
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description='カメラ動きと物体認識を用いた視線推定')
    parser.add_argument('--input', type=str, default='0', help='入力ビデオファイルパスまたはカメラインデックス')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='YOLOモデルのパス')
    args = parser.parse_args()
    
    # カメラまたはビデオファイルを開く
    if args.input.isdigit():
        cap = cv2.VideoCapture(int(args.input))
    else:
        cap = cv2.VideoCapture(args.input)
    
    # フレームサイズを取得
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # YOLOモデルの読み込み
    model = YOLO(args.model)
    
    # 初期フレームの読み込み
    ret, prev_frame = cap.read()
    if not ret:
        print("ビデオの読み込みに失敗しました")
        return
    
    # 現在のターゲット情報
    current_target = None
    current_target_center = None
    prev_target_center = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # YOLOで物体検出
        results = model(frame)
        detections = results[0].boxes
        
        # 最初のフレームまたはターゲットが消失した場合、初期ターゲットを選択
        if current_target is None and len(detections) > 0:
            current_target = select_initial_target(detections, frame_width, frame_height)
            if current_target is not None:
                x1, y1, x2, y2 = current_target.xyxy[0].tolist()
                current_target_center = ((x1 + x2) / 2, (y1 + y2) / 2)
        
        # 光フローを計算
        prev_pts, next_pts = calculate_optical_flow(prev_frame, frame)
        
        # カメラの動きを推定
        camera_direction, camera_movement = estimate_camera_movement(prev_pts, next_pts, frame_width, frame_height)
        
        # 現在のターゲットを追跡
        if current_target is not None:
            # ターゲットが検出されなくなった場合の処理
            target_found = False
            prev_target_center = current_target_center
            
            for detection in detections:
                x1, y1, x2, y2 = detection.xyxy[0].tolist()
                obj_center_x = (x1 + x2) / 2
                obj_center_y = (y1 + y2) / 2
                
                # 前フレームのターゲット位置に近い物体を探す
                if prev_target_center is not None:
                    prev_x, prev_y = prev_target_center
                    distance = np.sqrt((prev_x - obj_center_x)**2 + (prev_y - obj_center_y)**2)
                    
                    if distance < 50:  # 閾値は適宜調整
                        current_target = detection
                        current_target_center = (obj_center_x, obj_center_y)
                        target_found = True
                        break
            
            # ターゲットが見つからない場合
            if not target_found:
                current_target = None
                current_target_center = None
        
        # カメラが大きく動いた場合のターゲット切り替え判定
        if camera_direction != "STABLE" and current_target is not None and prev_target_center is not None:
            # ターゲットがカメラの動きに追従しているか確認
            if not is_target_following_camera(current_target_center, prev_target_center, camera_movement):
                # 追従していない場合は新しいターゲットを選択
                new_target = select_next_target(detections, camera_direction, current_target_center, frame_width, frame_height)
                if new_target is not None and new_target != current_target:
                    current_target = new_target
                    x1, y1, x2, y2 = current_target.xyxy[0].tolist()
                    current_target_center = ((x1 + x2) / 2, (y1 + y2) / 2)
        
        # 結果を描画
        for detection in detections:
            box = detection.xyxy[0].tolist()
            cls = int(detection.cls[0].item())
            conf = float(detection.conf[0].item())
            label = f"{model.names[cls]}: {conf:.2f}"
            
            # 現在のターゲットは赤、それ以外は緑で描画
            color = (0, 0, 255) if detection == current_target else (0, 255, 0)
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            cv2.putText(frame, label, (int(box[0]), int(box[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # カメラの動き方向を表示（英語表記に変更）
        cv2.putText(frame, f"Camera Movement: {camera_direction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # 光フローの矢印を描画
        if prev_pts is not None and next_pts is not None:
            for i, (prev, next) in enumerate(zip(prev_pts, next_pts)):
                p_x, p_y = prev.ravel()
                n_x, n_y = next.ravel()
                cv2.arrowedLine(frame, (int(p_x), int(p_y)), (int(n_x), int(n_y)), (0, 255, 255), 2)
        
        # 結果を表示
        cv2.imshow('Gaze Estimation', frame)
        
        # 次のイテレーションの準備
        prev_frame = frame.copy()
        
        # 'q'キーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 