import cv2
import numpy as np
from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk
import threading
import time
import csv
from datetime import datetime
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading

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

    def define_zones(self, cap):
        if cap is None:
            messagebox.showerror("Error", "Por favor, seleccione una cámara primero.")
            return

        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.current_zone.append((x, y))
            elif event == cv2.EVENT_RBUTTONDOWN:
                if len(self.current_zone) > 2:
                    self.zones[self.current_zone_type].append(np.array(self.current_zone, np.int32))
                    self.current_zone = []

        cv2.namedWindow("Definir Zonas")
        cv2.setMouseCallback("Definir Zonas", on_mouse)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            for zone_type, zone_list in self.zones.items():
                for zone in zone_list:
                    cv2.polylines(frame, [zone], True, (0, 255, 0), 2)

            if len(self.current_zone) > 1:
                cv2.polylines(frame, [np.array(self.current_zone, np.int32)], False, (255, 0, 0), 2)

            cv2.putText(frame, f"Definiendo zona de {self.current_zone_type}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Clic izquierdo: añadir punto, Clic derecho: finalizar zona", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Presione 'n' para siguiente tipo de zona, 'q' para terminar", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Definir Zonas", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('n'):
                zone_types = list(self.zones.keys())
                current_index = zone_types.index(self.current_zone_type)
                self.current_zone_type = zone_types[(current_index + 1) % len(zone_types)]
                self.current_zone = []
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
        self.customer_count = 0
        self.staff_count = 0
        self.customer_time = {}
        self.staff_time = {}
        self.current_frame = None
        self.csv_filename = "retail_analytics.csv"
        self.csv_interval = 300  # 5 minutes
        self.last_save_time = time.time()
        self.fps = 0
        self.last_frame_time = 0
        self.frame_count = 0
        self.lock = threading.Lock()  # Lock for thread safety

        self.setup_gui()
        self.load_model()

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Sistema Avanzado de Análisis de Retail")
        self.root.geometry("1200x800")

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

        ttk.Label(control_frame, text="Intervalo de guardado (s):").pack(anchor=tk.W)
        self.interval_entry = ttk.Entry(control_frame)
        self.interval_entry.pack(fill=tk.X, pady=5)
        self.interval_entry.insert(0, str(self.csv_interval))

        ttk.Label(control_frame, text="Nombre archivo CSV:").pack(anchor=tk.W)
        self.csv_name_entry = ttk.Entry(control_frame)
        self.csv_name_entry.pack(fill=tk.X, pady=5)
        self.csv_name_entry.insert(0, self.csv_filename)

        self.stats_label = ttk.Label(control_frame, text="Clientes: 0 | Personal: 0")
        self.stats_label.pack(pady=10)

        self.fps_label = ttk.Label(control_frame, text="FPS: 0")
        self.fps_label.pack(pady=5)

        self.setup_graphs(control_frame)

    def setup_graphs(self, parent):
        fig, self.ax1 = plt.subplots(figsize=(4, 3))
        self.occupancy_graph = FigureCanvasTkAgg(fig, master=parent)
        self.occupancy_graph.get_tk_widget().pack(fill=tk.X, pady=10)
        self.ax1.set_title("Ocupación del Local")
        self.ax1.set_xlabel("Tiempo")
        self.ax1.set_ylabel("Número de Personas")

        fig, self.ax2 = plt.subplots(figsize=(4, 3))
        self.time_graph = FigureCanvasTkAgg(fig, master=parent)
        self.time_graph.get_tk_widget().pack(fill=tk.X, pady=10)
        self.ax2.set_title("Tiempo Promedio de Estancia")
        self.ax2.set_xlabel("Zona")
        self.ax2.set_ylabel("Tiempo (minutos)")

    def load_model(self):
        try:
            self.model = YOLO('yolov8n.pt')
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar el modelo: {e}")
            self.is_running = False

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
                break

            self.frame_count += 1
            if self.frame_count % 3 == 0:  # Procesar solo cada 3 fotogramas
                with self.lock:
                    self.current_frame = frame.copy()
                    results = self.model(frame)
                    self.process_detections(results, frame)

            self.update_display(frame)
            self.save_to_csv_if_needed()
            self.update_fps()

        self.stop_analysis()

    def process_detections(self, results, frame):
        detected_objects = results[0].boxes.data
        current_time = time.time()

        for obj in detected_objects:
            x1, y1, x2, y2, conf, class_id = obj
            if conf < 0.5:  # Umbral de confianza
                continue

            center_point = ((x1 + x2) / 2, (y1 + y2) / 2)
            is_staff = self.is_in_zone(center_point, self.zone_handler.zones["personal"])
            person_id = f"{'staff' if is_staff else 'customer'}_{len(self.staff_time if is_staff else self.customer_time)}"

            if is_staff:
                if person_id not in self.staff_time:
                    self.staff_time[person_id] = {"start": current_time, "zones": set()}
                self.staff_time[person_id]["last_seen"] = current_time
                self.staff_count = len(self.staff_time)
            else:
                if person_id not in self.customer_time:
                    self.customer_time[person_id] = {"start": current_time, "zones": set()}
                self.customer_time[person_id]["last_seen"] = current_time
                self.customer_count = len(self.customer_time)

            for zone_type in ["entrada", "servicio", "salida"]:
                if self.is_in_zone(center_point, self.zone_handler.zones[zone_type]):
                    if is_staff:
                        self.staff_time[person_id]["zones"].add(zone_type)
                    else:
                        self.customer_time[person_id]["zones"].add(zone_type)

            # Dibujar bounding box y etiqueta
            color = (0, 255, 0) if is_staff else (255, 0, 0)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"{'Staff' if is_staff else 'Customer'}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        self.update_stats()

    def is_in_zone(self, point, zones):
        point = np.array(point, dtype=np.float32)
        for zone in zones:
            zone = np.array(zone, dtype=np.int32)
            if cv2.pointPolygonTest(zone, point, False) >= 0:
                return True
        return False

    def update_stats(self):
        self.stats_label.config(text=f"Clientes: {self.customer_count} | Personal: {self.staff_count}")
        self.update_occupancy_graph()
        self.update_time_graph()

    def update_occupancy_graph(self):
        self.ax1.clear()
        self.ax1.plot([0, 1, 2, 3, 4], [0, self.customer_count, self.customer_count * 0.8, self.customer_count * 1.2, self.customer_count])
        self.ax1.set_title("Ocupación del Local")
        self.ax1.set_xlabel("Tiempo")
        self.ax1.set_ylabel("Número de Personas")
        self.occupancy_graph.draw()

    def update_time_graph(self):
        self.ax2.clear()
        zones = ["entrada", "servicio", "salida"]
        customer_avg_times = [self.calculate_average_time(zone, is_staff=False) for zone in zones]
        staff_avg_times = [self.calculate_average_time(zone, is_staff=True) for zone in zones]

        x = np.arange(len(zones))
        width = 0.35

        self.ax2.bar(x - width / 2, customer_avg_times, width, label='Clientes')
        self.ax2.bar(x + width / 2, staff_avg_times, width, label='Personal')

        self.ax2.set_title("Tiempo Promedio de Estancia")
        self.ax2.set_xlabel("Zona")
        self.ax2.set_ylabel("Tiempo (minutos)")
        self.ax2.set_xticks(x)
        self.ax2.set_xticklabels(zones)
        self.ax2.legend()

        self.time_graph.draw()

    def calculate_average_time(self, zone, is_staff=False):
        time_data = self.staff_time if is_staff else self.customer_time
        total_time = sum(time.time() - data["start"] for data in time_data.values() if zone in data["zones"])
        count = sum(1 for data in time_data.values() if zone in data["zones"])
        return total_time / count / 60 if count > 0 else 0  # Convertir a minutos

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
        data = [
            current_datetime,
            self.customer_count,
            self.staff_count,
            self.calculate_average_time("entrada", False),
            self.calculate_average_time("servicio", False),
            self.calculate_average_time("salida", False),
            self.calculate_average_time("entrada", True),
            self.calculate_average_time("servicio", True),
            self.calculate_average_time("salida", True)
        ]
        try:
            with open(self.csv_filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(data)
        except IOError as e:
            messagebox.showerror("Error", f"Error al guardar el archivo CSV: {e}")

    def update_fps(self):
        current_time = time.time()
        self.fps = 1 / (current_time - self.last_frame_time)
        self.last_frame_time = current_time
        self.fps_label.config(text=f"FPS: {self.fps:.2f}")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = RetailAnalyticsSystem()
    app.run()
