import cv2

# Reemplaza con la URL RTSP de tu cámara Sricam
# Ejemplo si tu cámara es 192.168.1.4 y no requiere usuario/contraseña:
rtsp_url = "rtsp://192.168.1.4:554/onvif1" 

# Si tu cámara requiere usuario y contraseña (reemplaza 'tu_usuario' y 'tu_contraseña'):
# rtsp_url = "rtsp://tu_usuario:tu_contrasena@192.168.1.4:554/onvif1"

cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print(f"Error: No se pudo abrir el stream RTSP de la cámara en {rtsp_url}")
    exit()

print("Presiona 'q' para salir...")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: No se pudo leer el frame. El stream pudo haberse cerrado.")
        break

    # Aquí puedes realizar tu procesamiento de visión artificial en el 'frame'
    # Por ejemplo, convertir a escala de grises, detectar objetos, etc.
    # frame_procesado = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

    cv2.imshow('Sricam Feed', frame) # Muestra el frame original o procesado

    # Presiona 'q' para salir del loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()