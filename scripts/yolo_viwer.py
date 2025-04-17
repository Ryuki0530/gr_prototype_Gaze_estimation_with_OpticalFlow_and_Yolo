import cv2

def viewYoloResult(frame,model,detection):
    box = detection.xyxy[0].tolist()
    cls = int(detection.cls[0].item())
    conf = float(detection.conf[0].item())
    label = f"{model.names[cls]}: {conf:.2f}"

    color = (0,255,0)
    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
    cv2.putText(frame, label, (int(box[0]), int(box[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
