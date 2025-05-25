import cv2
import numpy as np
from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox, filedialog
from PIL import Image, ImageTk
import threading
import time
import csv
from datetime import datetime
import os

class PeopleCounter:
    def __init__(self):
        self.model = None
        self.cap = None
        self.is_running = False
        self.person_tracker = {}  # Key: track_id, Value: dict with zone info
        self.detection_areas = []  # Multiple detection areas
        self.confidence_threshold = 0.5
        self.csv_filename = "people_counter.csv"
        self.csv_interval = 300  # Save every 5 minutes
        self.last_save_time = time.time()
        self.counts = {}  # Counts per zone
        self.times = {}   # Total time per zone
        self.counted_ids_per_zone = {}  # IDs counted per zone
        self.setup_gui()

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Contador de Personas con Seguimiento")
        self.root.geometry("1000x700")

        self.video_label = ttk.Label(self.root)
        self.video_label.pack(padx=10, pady=10)

        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(control_frame, text="Seleccionar Cámara", command=self.select_camera).pack(fill=tk.X, pady=5)
        ttk.Button(control_frame, text="Definir Áreas de Detección", command=self.select_detection_areas).pack(fill=tk.X, pady=5)
        ttk.Label(control_frame, text="Seleccionar Modelo YOLO:").pack(anchor=tk.W)
        self.model_combo = ttk.Combobox(control_frame, values=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"])
        self.model_combo.pack(fill=tk.X, pady=5)
        self.model_combo.set("yolov8n.pt")
        self.start_button = ttk.Button(control_frame, text="Iniciar Conteo", command=self.toggle_counting)
        self.start_button.pack(fill=tk.X, pady=5)

        self.stats_labels = {}

        # Agregar barra de estado con copyright
        self.status_bar = ttk.Label(self.root, text="Creado por AriasDmk © 2025", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X) # Posicionar abajo y expandir horizontalmente

    def select_camera(self):
        camera_index = simpledialog.askinteger("Seleccionar Cámara", "Ingrese el índice de la cámara:", initialvalue=0)
        if camera_index is not None:
            try:
                self.cap = cv2.VideoCapture(camera_index)
                if not self.cap.isOpened():
                    raise ValueError("No se pudo abrir la cámara seleccionada.")
                print(f"Cámara {camera_index} seleccionada correctamente.")
                messagebox.showinfo("Éxito", f"Cámara {camera_index} seleccionada correctamente.")
            except Exception as e:
                print(f"Error: No se pudo seleccionar la cámara: {e}")

    def select_detection_areas(self):
        if self.cap is None:
            print("Error: Por favor, seleccione una cámara primero.")
            return

        ret, frame = self.cap.read()
        if not ret:
            print("Error: No se pudo leer de la cámara.")
            return

        frame = cv2.flip(frame, 1)
        frozen_frame = frame.copy()

        self.detection_areas = []
        self.area_names = []
        self.stats_labels = {}
        self.counted_ids_per_zone = {}

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
                self.detection_areas.append((x1, y1, abs(x - x1), abs(y - y1)))
                self.current_zone = []
                cv2.rectangle(frozen_frame, (x1, y1), (x, y), (255, 0, 0), 2)
                width = abs(x - x1)
                height = abs(y - y1)
                cv2.putText(frozen_frame, f"{width}x{height}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                # Solicitar nombre de la zona
                zone_name = simpledialog.askstring("Nombre de la Zona", "Ingrese el nombre para esta zona:")
                if zone_name:
                    self.area_names.append(zone_name)
                else:
                    zone_name = f"Zona {len(self.area_names) + 1}"
                    self.area_names.append(zone_name)
                label = ttk.Label(self.root, text=f"{zone_name}: 0 | Tiempo: 0.0 min")
                label.pack(pady=5)
                self.stats_labels[zone_name] = label
                self.counts[zone_name] = 0
                self.times[zone_name] = 0.0
                self.counted_ids_per_zone[zone_name] = set()

        cv2.namedWindow("Definir Zonas")
        cv2.setMouseCallback("Definir Zonas", on_mouse)
        self.current_zone = []

        while True:
            cv2.putText(frozen_frame, "Arrastre para definir zona, 'n' para siguiente, 'q' para terminar", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Definir Zonas", frozen_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('n'):
                continue
            elif key == ord('q'):
                break

        cv2.destroyWindow("Definir Zonas")
        messagebox.showinfo("Zonas Definidas", "Las zonas han sido definidas correctamente.")

    def toggle_counting(self):
        if not self.is_running:
            self.start_counting()
        else:
            self.stop_counting()

    def start_counting(self):
        try:
            model_path = self.model_combo.get()
            if not model_path:
                model_path = "yolov8n.pt"
            self.model = YOLO(model_path)
            if self.cap is None:
                print("Error: Por favor, seleccione una cámara primero.")
                return

            self.is_running = True
            self.start_button.config(text="Detener Conteo")
            threading.Thread(target=self.process_video, daemon=True).start()
        except Exception as e:
            print(f"Error al iniciar el conteo: {e}")

    def stop_counting(self):
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
        self.start_button.config(text="Iniciar Conteo")

    def process_video(self):
        while self.is_running:
            success, frame = self.cap.read()
            if success:
                frame = cv2.flip(frame, 1)
                results = self.model.track(frame, persist=True, conf=self.confidence_threshold, classes=[0])
                annotated_frame = frame.copy()

                self.track_people(results[0].boxes, annotated_frame)
                self.draw_detection_areas(annotated_frame)
                self.update_video_display(annotated_frame)
                self.save_to_csv_if_needed()

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("Error al capturar el fotograma")
                break

        self.stop_counting()

    def track_people(self, boxes, frame):
        current_time = time.time()
        current_ids = set()

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            track_id = int(box.id[0]) if box.id is not None else None

            if cls == 0 and conf >= self.confidence_threshold and track_id is not None:
                current_ids.add(track_id)

                # Calcular punto central
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                if track_id not in self.person_tracker:
                    self.person_tracker[track_id] = {
                        'zones': {},  # key: zone_name, value: total_time_in_zone
                        'current_zone': None,
                        'zone_entry_time': None,
                        'captured_zones': set(),  # Zonas donde se ha capturado imagen
                    }

                person_data = self.person_tracker[track_id]

                # Verificar si está dentro de una zona
                inside_zone = False
                for zone_index, (zx, zy, zw, zh) in enumerate(self.detection_areas):
                    zone_name = self.area_names[zone_index]
                    if zx <= cx <= zx + zw and zy <= cy <= zy + zh:
                        inside_zone = True
                        if person_data['current_zone'] != zone_name:
                            # Ingresó a una nueva zona
                            person_data['current_zone'] = zone_name
                            person_data['zone_entry_time'] = current_time
                            if track_id not in self.counted_ids_per_zone[zone_name]:
                                self.counts[zone_name] += 1
                                self.counted_ids_per_zone[zone_name].add(track_id)
                        else:
                            # Sigue en la misma zona
                            time_in_zone = current_time - person_data['zone_entry_time']
                            total_time_in_zone = person_data['zones'].get(zone_name, 0) + time_in_zone
                            # Mostrar el tiempo en la etiqueta
                            cv2.putText(frame, f"ID: {track_id} | Tiempo: {int(total_time_in_zone)}s", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            # Si supera 60 segundos y no se ha capturado imagen
                            if total_time_in_zone >= 60 and zone_name not in person_data['captured_zones']:
                                self.capture_image(frame, track_id, zone_name)
                                person_data['captured_zones'].add(zone_name)
                        break
                if not inside_zone and person_data['current_zone'] is not None:
                    # Salió de la zona
                    time_spent = current_time - person_data['zone_entry_time']
                    zone_name = person_data['current_zone']
                    # Actualizar tiempo total en la zona
                    person_data['zones'][zone_name] = person_data['zones'].get(zone_name, 0) + time_spent
                    self.times[zone_name] += time_spent
                    person_data['current_zone'] = None
                    person_data['zone_entry_time'] = None

                # Dibujar bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                if person_data['current_zone'] is None:
                    # Mostrar solo el ID
                    cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        # Eliminar rastros antiguos
        old_ids = set(self.person_tracker.keys()) - current_ids
        for old_id in old_ids:
            person_data = self.person_tracker[old_id]
            if person_data['current_zone'] is not None:
                # Salió de la zona sin actualizar
                time_spent = current_time - person_data['zone_entry_time']
                zone_name = person_data['current_zone']
                # Actualizar tiempo total en la zona
                person_data['zones'][zone_name] = person_data['zones'].get(zone_name, 0) + time_spent
                self.times[zone_name] += time_spent
            del self.person_tracker[old_id]

        # Actualizar estadísticas
        self.update_stats()

    def capture_image(self, frame, track_id, zone_name):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"captura_{zone_name}_ID{track_id}_{timestamp}.jpg"
        folder_path = os.path.join("capturas", zone_name)
        os.makedirs(folder_path, exist_ok=True)
        cv2.imwrite(os.path.join(folder_path, filename), frame)
        print(f"Captura guardada: {filename}")

    def draw_detection_areas(self, frame):
        for idx, (x, y, w, h) in enumerate(self.detection_areas):
            zone_name = self.area_names[idx]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)
            cv2.putText(frame, f"{zone_name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def update_stats(self):
        for zone in self.counts:
            total_time = self.times[zone] / 60  # Convertir a minutos
            self.stats_labels[zone].config(text=f"{zone}: {self.counts[zone]} | Tiempo: {total_time:.2f} min")

    def update_video_display(self, frame):
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
        for zone in self.area_names:
            total_time = self.times[zone] / 60  # Convertir a minutos
            data.extend([self.counts[zone], f"{total_time:.2f}"])
        headers = ["Fecha y Hora"]
        for zone in self.area_names:
            headers.extend([f"Conteo {zone}", f"Tiempo {zone} (min)"])
        try:
            file_exists = os.path.isfile(self.csv_filename)
            with open(self.csv_filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow(headers)
                writer.writerow(data)
        except IOError as e:
            print(f"Error: Error al guardar el archivo CSV: {e}")

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    def on_closing(self):
        if messagebox.askokcancel("Salir", "¿Está seguro que desea salir?"):
            self.is_running = False
            if self.cap:
                self.cap.release()
            self.root.destroy()

if __name__ == "__main__":
    counter = PeopleCounter()
    counter.run()
