import cv2
from ultralytics import YOLO

# Cargar el modelo YOLO
model = YOLO('yolov8n.pt')

# Capturar video desde la cámara
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Ejecutar el modelo YOLO en el frame
    results = model(frame)
    
    # Mostrar resultados
    for obj in results[0].boxes.data:
        x1, y1, x2, y2, conf, class_id = obj
        if conf > 0.3 and int(class_id) == 0:  # Solo detectar personas (class_id 0)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    
    cv2.imshow("Detección YOLO", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
