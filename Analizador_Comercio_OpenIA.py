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

# Nueva clase Tooltip para ventanas de información emergente
class Tooltip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event=None):
        if self.tooltip_window:
            return
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20

        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{x}+{y}")

        label = tk.Label(self.tooltip_window, text=self.text, justify=tk.LEFT,
                         background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                         font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)

    def hide_tooltip(self, event=None):
        if self.tooltip_window:
            self.tooltip_window.destroy()
        self.tooltip_window = None

class PeopleCounter:
    def __init__(self):
        # Variables para ajustes configurables
        self.model_name = "yolov8n.pt"  # Modelo YOLO por defecto
        self.confidence_threshold = 0.5  # Umbral de confianza mínimo
        self.classes_to_detect = [0]  # Clases a detectar (por defecto, personas)
        self.frame_skip = 1  # Procesar cada N frames (para mejorar rendimiento)

        self.model = None
        self.cap = None
        self.is_running = False
        self.person_tracker = {}  # Key: track_id, Value: dict with zone info
        self.detection_areas = []  # Multiple detection areas
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
        self.model_combo.set(self.model_name)
        self.start_button = ttk.Button(control_frame, text="Iniciar Conteo", command=self.toggle_counting)
        self.start_button.pack(fill=tk.X, pady=5)

        self.stats_labels = {}

        # Agregar barra de estado con copyright
        self.status_bar = ttk.Label(self.root, text="Creado por AriasDmk © 2025", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Botón para abrir la ventana de ajustes
        self.settings_button = ttk.Button(control_frame, text="Ajustes", command=self.open_settings_window)
        self.settings_button.pack(fill=tk.X, pady=5)

    def select_camera(self):
        # Usar askstring para permitir URL o índice numérico
        camera_input = simpledialog.askstring("Seleccionar Fuente de Video", "Ingrese el índice de la cámara local (ej: 0) o la URL de la cámara IP (ej: rtsp://user:pass@ip:port/stream)")

        if camera_input is not None:
            try:
                # Intentar convertir la entrada a un entero (índice de cámara local)
                camera_source = int(camera_input)
                print(f"Intentando abrir cámara local con índice: {camera_source}")
            except ValueError:
                # Si falla, tratar la entrada como una URL de cámara IP
                camera_source = camera_input
                print(f"Intentando abrir cámara IP con URL: {camera_source}")

            try: 
                self.cap = cv2.VideoCapture(camera_source)
                if not self.cap.isOpened():
                    raise ValueError(f"No se pudo abrir la fuente de video: {camera_input}.")
                print(f"Fuente de video {camera_input} seleccionada correctamente.")
                messagebox.showinfo("Éxito", f"Fuente de video {camera_input} seleccionada correctamente.")
            except Exception as e:
                print(f"Error: No se pudo seleccionar la fuente de video: {e}")
                messagebox.showerror("Error", f"Error al seleccionar la fuente de video: {e}")

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

    def open_settings_window(self):
        # Abre una nueva ventana para los ajustes
        self.settings_window = tk.Toplevel(self.root)
        self.settings_window.title("Ajustes")

        # Combobox para seleccionar el modelo YOLO
        model_label = ttk.Label(self.settings_window, text="Modelo YOLO:")
        model_label.grid(row=0, column=0, padx=5, pady=5)
        self.model_combo_setting = ttk.Combobox(self.settings_window, values=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"])
        self.model_combo_setting.grid(row=0, column=1, padx=5, pady=5)
        self.model_combo_setting.set(self.model_name)
        Tooltip(model_label, "Seleccione el modelo YOLO a utilizar. Modelos como 'n' (nano) son más rápidos pero menos precisos.\nModelos como 'x' (xlarge) son más precisos pero requieren más recursos.\nElija según el rendimiento de su hardware.")

        # Campo para ingresar el umbral de confianza mínimo
        confidence_label = ttk.Label(self.settings_window, text="Confianza mínima:")
        confidence_label.grid(row=1, column=0, padx=5, pady=5)
        self.confidence_entry_setting = ttk.Entry(self.settings_window)
        self.confidence_entry_setting.grid(row=1, column=1, padx=5, pady=5)
        self.confidence_entry_setting.insert(0, str(self.confidence_threshold))
        Tooltip(confidence_label, "Establece el umbral de confianza mínimo para considerar una detección válida.\nEl valor debe estar entre 0.0 y 1.0.\nValores más bajos (ej: 0.2) detectarán más objetos (posibles falsos positivos). Valores más altos (ej: 0.8) solo detectarán objetos con alta certeza.")

        # Combobox para seleccionar la clase a detectar y contar
        detect_label = ttk.Label(self.settings_window, text="Detectar y contar:")
        detect_label.grid(row=2, column=0, padx=5, pady=5)
        self.detect_combo_setting = ttk.Combobox(self.settings_window, values=["person", "car", "motorcycle"])
        self.detect_combo_setting.grid(row=2, column=1, padx=5, pady=5)
        # Diccionario inverso para obtener el nombre de la clase actual
        class_dict_inv = {0: 'person', 2: 'car', 3: 'motorcycle'}
        current_class_id = self.classes_to_detect[0] if self.classes_to_detect else 0
        self.detect_combo_setting.set(class_dict_inv.get(current_class_id, 'person'))
        Tooltip(detect_label, "Seleccione el tipo de objeto que desea detectar y contar.\nSolo se procesarán las detecciones de la clase seleccionada.")

        # Campo para ingresar cada cuántos frames procesar
        frame_skip_label = ttk.Label(self.settings_window, text="Procesar cada N frames:")
        frame_skip_label.grid(row=3, column=0, padx=5, pady=5)
        self.frame_skip_entry = ttk.Entry(self.settings_window)
        self.frame_skip_entry.grid(row=3, column=1, padx=5, pady=5)
        self.frame_skip_entry.insert(0, str(self.frame_skip))
        Tooltip(frame_skip_label, "Define la frecuencia de procesamiento de frames (número entero >= 1).\nUn valor de 1 procesa todos los frames (mayor precisión, menor rendimiento).\nValores mayores (ej: 5 o 10) procesan menos frames (menor precisión, mayor rendimiento) y pueden ser útiles en hardware limitado.")

        # Botones para guardar o cancelar los ajustes
        ttk.Button(self.settings_window, text="Guardar", command=self.save_settings).grid(row=4, column=0, padx=5, pady=5)
        ttk.Button(self.settings_window, text="Cancelar", command=self.settings_window.destroy).grid(row=4, column=1, padx=5, pady=5)

    def save_settings(self):
        # Guarda los ajustes ingresados por el usuario
        try:
            self.model_name = self.model_combo_setting.get()
            self.confidence_threshold = float(self.confidence_entry_setting.get())

            # Actualizamos la clase a detectar
            class_name = self.detect_combo_setting.get()
            class_dict = {'person': 0, 'car': 2, 'motorcycle': 3}
            self.classes_to_detect = [class_dict.get(class_name, 0)]

            self.frame_skip = int(self.frame_skip_entry.get())

            self.settings_window.destroy()  # Cerramos la ventana de ajustes
        except ValueError:
            messagebox.showerror("Error", "Por favor, ingrese valores válidos.")

    def toggle_counting(self):
        if not self.is_running:
            self.start_counting()
        else:
            self.stop_counting()

    def start_counting(self):
        try:
            model_path = self.model_combo.get() # Usar self.model_combo de la ventana principal
            if not model_path: # Fallback si no se seleccionó en la principal
                 model_path = self.model_name

            if not os.path.exists(model_path):
                 messagebox.showerror("Error", f"El archivo de modelo {model_path} no se encuentra. Asegúrese de descargarlo.")
                 return

            self.model = YOLO(model_path)
            if self.cap is None:
                print("Error: Por favor, seleccione una cámara primero.")
                return

            self.is_running = True
            self.start_button.config(text="Detener Conteo")
            threading.Thread(target=self.process_video, daemon=True).start()
        except Exception as e:
            print(f"Error al iniciar el conteo: {e}")
            messagebox.showerror("Error", f"Error al iniciar el conteo: {e}")

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

                # Aplicar preprocesamiento si se define un área de detección (opcional, podría ser configurable)
                # if self.detection_areas:
                #     frame = self.preprocess_frame(frame)

                # Realizar detección con ajustes configurados
                results = self.model.track(
                    frame, 
                    persist=True, 
                    conf=self.confidence_threshold, 
                    classes=self.classes_to_detect,
                    iou=0.5 # Ajustar IoU para NMS si es necesario
                    )

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

                if track_id not in self.person_tracker: # Initialize tracker data for new ID
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
                            if track_id not in self.counted_ids_per_zone.get(zone_name, set()): # Check if zone exists
                                self.counts[zone_name] = self.counts.get(zone_name, 0) + 1 # Ensure zone count exists
                                self.counted_ids_per_zone[zone_name] = self.counted_ids_per_zone.get(zone_name, set()) # Ensure zone set exists
                                self.counted_ids_per_zone[zone_name].add(track_id)
                        else:
                            # Sigue en la misma zona
                            time_in_zone = current_time - person_data['zone_entry_time']
                            total_time_in_zone = person_data['zones'].get(zone_name, 0) + time_in_zone
                            # Mostrar el tiempo en la etiqueta
                            cv2.putText(frame, f"ID: {track_id} | Tiempo: {int(time_in_zone)}s", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            # Si supera 60 segundos y no se ha capturado imagen
                            if time_in_zone >= 60 and zone_name not in person_data['captured_zones']:
                                self.capture_image(frame, track_id, zone_name)
                                person_data['captured_zones'].add(zone_name)
                        break
                if not inside_zone and person_data['current_zone'] is not None:
                    # Salió de la zona
                    time_spent = current_time - person_data['zone_entry_time']
                    zone_name = person_data['current_zone']
                    # Actualizar tiempo total en la zona
                    person_data['zones'][zone_name] = person_data['zones'].get(zone_name, 0) + time_spent
                    self.times[zone_name] = self.times.get(zone_name, 0) + time_spent # Ensure zone time exists
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
                self.times[zone_name] = self.times.get(zone_name, 0) + time_spent # Ensure zone time exists
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
            # Cambiar color a blanco (255, 255, 255) y grosor de línea a 1
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
            total_time = self.times.get(zone, 0) / 60  # Ensure zone time exists and convert to minutes
            data.extend([self.counts.get(zone, 0), f"{total_time:.2f}"]) # Ensure zone count exists
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
            if self.cap: # Check if cap is initialized before releasing
                self.cap.release()
            self.root.destroy()

if __name__ == "__main__":
    counter = PeopleCounter()
    counter.run()
