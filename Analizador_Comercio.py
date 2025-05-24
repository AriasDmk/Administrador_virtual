import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
from PIL import Image, ImageTk
import threading
import time
import csv
from datetime import datetime
import os

class CameraHandler:
    def __init__(self):
        self.cap = None

    def select_camera(self):
        camera_index = simpledialog.askinteger("Seleccionar Cámara", "Ingrese el índice de la cámara:", initialvalue=0)
        if camera_index is not None:
            try:
                self.cap = cv2.VideoCapture(camera_index)
                if not self.cap.isOpened():
                    raise ValueError("No se pudo abrir la cámara seleccionada.")
                messagebox.showinfo("Éxito", f"Cámara {camera_index} seleccionada correctamente.")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo seleccionar la cámara: {e}")

    def release_camera(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()

class ZoneHandler:
    def __init__(self):
        self.zones = {"entrada": [], "servicio": [], "salida": [], "personal": []}
        self.current_zone = []
        self.current_zone_type = "entrada"
        self.zone_steps = [
            ("entrada", "Paso 1: Indique la zona de entrada."),
            ("servicio", "Paso 2: Indique la zona de servicio."),
            ("personal", "Paso 3: Indique la zona del personal."),
            ("salida", "Paso 4: Indique la zona de salida.")
        ]
        self.current_step = 0

    def define_zones(self, cap):
        if cap is None:
            messagebox.showerror("Error", "Por favor, seleccione una cámara primero.")
            return

        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Error", "No se pudo leer de la cámara.")
            return

        frozen_frame = frame.copy()

        def on_mouse(event, x, y, flags, param):
            nonlocal frozen_frame
            if event == cv2.EVENT_LBUTTONDOWN:
                self.current_zone = [(x, y)]
            elif event == cv2.EVENT_MOUSEMOVE and len(self.current_zone) == 1:
                temp_frame = frozen_frame.copy()
                x1, y1 = self.current_zone[0]
                cv2.rectangle(temp_frame, (x1, y1), (x, y), (255, 0, 0), 2)
                width = abs(x - x1)
                height = abs(y - y1)
                cv2.putText(temp_frame, f"{width}x{height}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.imshow("Definir Zonas", temp_frame)
            elif event == cv2.EVENT_LBUTTONUP:
                x1, y1 = self.current_zone[0]
                self.zones[self.current_zone_type].append(((x1, y1), (x, y)))
                self.current_zone = []
                cv2.rectangle(frozen_frame, (x1, y1), (x, y), (255, 0, 0), 2)
                width = abs(x - x1)
                height = abs(y - y1)
                cv2.putText(frozen_frame, f"{width}x{height}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.namedWindow("Definir Zonas")
        cv2.setMouseCallback("Definir Zonas", on_mouse)

        while True:
            cv2.putText(frozen_frame, self.zone_steps[self.current_step][1], (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frozen_frame, "Arrastre para definir zona, 'n' para siguiente, 'q' para terminar", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Definir Zonas", frozen_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('n'):
                self.current_step = (self.current_step + 1) % len(self.zone_steps)
                self.current_zone_type = self.zone_steps[self.current_step][0]
            elif key == ord('q'):
                break

        cv2.destroyWindow("Definir Zonas")
        messagebox.showinfo("Zonas Definidas", "Las zonas han sido definidas correctamente.")

class RetailAnalyticsSystem:
    def __init__(self):
        self.model = None
        self.camera_handler = CameraHandler()
        self.zone_handler = ZoneHandler()
        self.is_running = False
        self.counts = {"entrada": 0, "servicio": 0, "salida": 0, "personal": 0}
        self.times = {"entrada": 0, "servicio": 0, "salida": 0, "personal": 0}
        self.current_frame = None
        self.csv_filename = "retail_analytics.csv"
        self.csv_interval = 300  # 5 minutes
        self.last_save_time = time.time()
        self.frame_count = 0
        self.lock = threading.Lock()

        self.tracker = DeepSort(max_age=30, n_init=3, nn_budget=100)
        self.tracked_objects = {}

        self.setup_gui()
        self.load_model()

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Sistema de Análisis de Retail")
        self.root.geometry("800x600")

        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.video_frame = ttk.Frame(main_frame)
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack(padx=10, pady=10)

        control_frame = ttk.Frame(main_frame)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10)

        ttk.Button(control_frame, text="Seleccionar Cámara", command=self.camera_handler.select_camera).pack(fill=tk.X, pady=5)
        ttk.Button(control_frame, text="Definir Zonas", command=lambda: self.zone_handler.define_zones(self.camera_handler.cap)).pack(fill=tk.X, pady=5)
        self.start_button = ttk.Button(control_frame, text="Iniciar Análisis", command=self.toggle_analysis)
        self.start_button.pack(fill=tk.X, pady=5)

        self.stats_labels = {}
        for zone in ["entrada", "servicio", "salida", "personal"]:
            self.stats_labels[zone] = ttk.Label(control_frame, text=f"{zone.capitalize()}: 0 | Tiempo: 0.0 min")
            self.stats_labels[zone].pack(pady=5)

    def load_model(self):
        try:
            self.model = YOLO('yolov8n.pt')
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar el modelo: {e}")

    def toggle_analysis(self):
        if not self.is_running:
            self.start_analysis()
        else:
            self.stop_analysis()

    def start_analysis(self):
        if self.camera_handler.cap is None:
            messagebox.showerror("Error", "Por favor, seleccione una cámara primero.")
            return
        if not any(self.zone_handler.zones.values()):
            messagebox.showerror("Error", "Por favor, defina al menos una zona primero.")
            return

        self.is_running = True
        self.start_button.config(text="Detener Análisis")
        threading.Thread(target=self.analyze_video, daemon=True).start()

    def stop_analysis(self):
        self.is_running = False
        self.start_button.config(text="Iniciar Análisis")

    def analyze_video(self):
        while self.is_running:
            ret, frame = self.camera_handler.cap.read()
            if not ret:
                messagebox.showerror("Error", "No se pudo leer de la cámara.")
                break

            self.frame_count += 1
            if self.frame_count % 3 == 0:  # Procesar cada 3 fotogramas
                with self.lock:
                    self.current_frame = frame.copy()
                    try:
                        results = self.model(frame, size=320)
                        self.process_detections(results, frame)
                    except Exception as e:
                        messagebox.showerror("Error", f"Error al procesar el modelo: {e}")

            self.update_display(frame)
            self.save_to_csv_if_needed()
            self.update_stats()

        self.stop_analysis()

    def process_detections(self, results, frame):
        detections = results[0].boxes.data.cpu().numpy()

        if not isinstance(detections, np.ndarray) or len(detections) == 0:
            return

        bboxes = []
        for det in detections:
            x1, y1, x2, y2, conf, class_id = det
            if conf > 0.5 and int(class_id) == 0:  # Solo rastreamos personas (ID clase 0)
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                w, h = x2 - x1, y2 - y1
                bboxes.append([cx, cy, w, h, conf])

        if len(bboxes) > 0:
            tracks = self.tracker.update_tracks(bboxes, frame=frame)

            current_time = time.time()

            for track in tracks:
                if not track.is_confirmed():
                    continue

                bbox = track.to_tlbr()
                track_id = track.track_id
                x1, y1, x2, y2 = bbox
                center_point = ((x1 + x2) / 2, (y1 + y2) / 2)

                person_id = f"Persona_{int(track_id)}"

                if person_id not in self.tracked_objects:
                    self.tracked_objects[person_id] = {
                        "start_time": current_time,
                        "last_seen": current_time,
                        "zone": None
                    }
                else:
                    self.tracked_objects[person_id]["last_seen"] = current_time

                for zone_type in self.zone_handler.zones:
                    if self.is_in_zone(center_point, self.zone_handler.zones[zone_type]):
                        if self.tracked_objects[person_id]["zone"] != zone_type:
                            self.tracked_objects[person_id]["zone"] = zone_type
                            self.counts[zone_type] += 1
                        break

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, person_id, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        self.update_times()
        def update_times(self):
           current_time = time.time()
        for person_id, data in list(self.tracked_objects.items()):
            if current_time - data["last_seen"] > 5:
                if data["zone"]:
                    self.times[data["zone"]] += current_time - data["start_time"]
                del self.tracked_objects[person_id]
            elif data["zone"]:
                self.times[data["zone"]] += current_time - data["last_seen"]
                data["last_seen"] = current_time

    def is_in_zone(self, point, zones):
        for (p1, p2) in zones:
            if p1[0] <= point[0] <= p2[0] and p1[1] <= point[1] <= p2[1]:
                return True
        return False

    def update_stats(self):
        for zone in self.counts:
            avg_time = self.times[zone] / self.counts[zone] / 60 if self.counts[zone] > 0 else 0
            self.stats_labels[zone].config(text=f"{zone.capitalize()}: {self.counts[zone]} | Tiempo: {avg_time:.2f} min")

    def update_display(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        photo = ImageTk.PhotoImage(image=image)
        self.video_label.config(image=photo)
        self.video_label.image = photo

    def save_to_csv_if_needed(self):
        current_time = time.time()
        if current_time - self.last_save_time >= self.csv_interval:
            self.save_to_csv()
            self.last_save_time = current_time

    def save_to_csv(self):
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data = [current_datetime]
        for zone in ["entrada", "servicio", "salida", "personal"]:
            avg_time = self.times[zone] / self.counts[zone] / 60 if self.counts[zone] > 0 else 0
            data.extend([self.counts[zone], f"{avg_time:.2f}"])
        
        try:
            with open(self.csv_filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(data)
        except IOError as e:
            messagebox.showerror("Error", f"Error al guardar el archivo CSV: {e}")

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    def on_closing(self):
        if messagebox.askokcancel("Salir", "¿Está seguro que desea salir?"):
            self.is_running = False
            self.camera_handler.release_camera()
            self.root.destroy()

if __name__ == "__main__":
    app = RetailAnalyticsSystem()
    app.run()