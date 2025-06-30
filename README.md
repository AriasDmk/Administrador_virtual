# 📊 Analizador de Comercio OpenIA

## Descripción

**Analizador de Comercio OpenIA** es una aplicación de análisis de tráfico de personas para comercios que utiliza inteligencia artificial para detectar, rastrear y analizar el comportamiento de los clientes en diferentes zonas de un establecimiento comercial.

## 🚀 Características Principales

### 🎯 Detección y Seguimiento
- **Detección de personas** usando YOLOv8 (You Only Look Once)
- **Seguimiento en tiempo real** con asignación de IDs únicos
- **Múltiples zonas de detección** configurables
- **Seguimiento de tiempo** que cada persona permanece en cada zona

### 📈 Análisis y Estadísticas
- **Conteo de personas** por zona
- **Tiempo promedio** de permanencia
- **Captura automática** de imágenes cuando alguien permanece más de 60 segundos
- **Exportación a CSV** con datos históricos
- **Interfaz gráfica** intuitiva con estadísticas en tiempo real

### ⚙️ Configuración Avanzada
- **Múltiples modelos YOLO** (nano, small, medium, large, xlarge)
- **Umbral de confianza** ajustable
- **Fuentes de video** flexibles (cámara local, IP, archivos)
- **Áreas de detección** personalizables con nombres

## 📋 Requisitos del Sistema

### Software
- Python 3.8 o superior
- OpenCV 4.x
- Ultralytics (YOLOv8)
- Tkinter (incluido con Python)
- PIL (Pillow)

### Hardware Recomendado
- **Mínimo**: CPU dual-core, 4GB RAM
- **Recomendado**: CPU quad-core, 8GB+ RAM, GPU compatible con CUDA
- **Cámara**: Webcam USB o cámara IP compatible con RTSP

## 🛠️ Instalación

### 1. Clonar el Repositorio
```bash
git clone <url-del-repositorio>
cd analizador_comercio_openIA
```

### 2. Crear Entorno Virtual
```bash
python -m venv .venv
```

### 3. Activar Entorno Virtual
```bash
# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### 4. Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 5. Descargar Modelos YOLO
Los modelos se descargan automáticamente en el primer uso, o puedes descargarlos manualmente:
```bash
# Modelo nano (recomendado para inicio)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

# Modelo small
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt

# Modelo medium
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt
```

## 🎮 Uso

### 1. Ejecutar la Aplicación
```bash
python Analizador_Comercio_OpenIA.py
```

### 2. Configurar la Cámara
1. Hacer clic en **"Seleccionar Cámara"**
2. Ingresar:
   - **Índice de cámara local**: `0`, `1`, `2`...
   - **URL de cámara IP**: `rtsp://usuario:contraseña@ip:puerto/stream`

### 3. Definir Zonas de Detección
1. Hacer clic en **"Definir Áreas de Detección"**
2. Arrastrar el mouse para crear rectángulos
3. Asignar nombres a cada zona (ej: "Entrada", "Vitrina 1", "Pasillo")
4. Presionar `q` para finalizar

### 4. Configurar Ajustes (Opcional)
1. Hacer clic en **"Ajustes"**
2. Seleccionar modelo YOLO según necesidades:
   - **yolov8n.pt**: Más rápido, menos preciso
   - **yolov8m.pt**: Balanceado (recomendado)
   - **yolov8x.pt**: Más preciso, más lento
3. Ajustar umbral de confianza (0.1 - 1.0)

### 5. Iniciar Análisis
1. Hacer clic en **"Iniciar Conteo"**
2. Observar estadísticas en tiempo real
3. Los datos se guardan automáticamente cada 5 minutos

## 📁 Estructura del Proyecto

```
analizador_comercio_openIA/
├── Analizador_Comercio_OpenIA.py    # Aplicación principal
├── Analizador_Comercio.py           # Versión anterior
├── escanerCamaras.py                # Escáner de cámaras IP
├── testyolo.py                      # Script de prueba YOLO
├── yolov8n.pt                       # Modelo YOLO nano
├── yolov8m.pt                       # Modelo YOLO medium
├── people_counter.csv               # Datos exportados
├── capturas/                        # Imágenes capturadas
│   ├── Entrada/
│   └── Vitrina 1/
├── deep_sort/                       # Algoritmo de seguimiento
├── sort/                           # Algoritmo de seguimiento
├── .venv/                          # Entorno virtual
└── README.md                       # Este archivo
```

## 📊 Interpretación de Datos

### Archivo CSV (`people_counter.csv`)
```csv
Fecha y Hora,Conteo Zona1,Tiempo Zona1 (min),Conteo Zona2,Tiempo Zona2 (min)
2024-12-08 14:25:05,1,0.08,0,0.00
```

**Columnas:**
- **Fecha y Hora**: Timestamp del registro
- **Conteo ZonaX**: Número de personas únicas que han pasado por la zona
- **Tiempo ZonaX (min)**: Tiempo total acumulado en minutos

### Carpetas de Capturas
- **Ubicación**: `capturas/[nombre_zona]/`
- **Formato**: `captura_[zona]_ID[track_id]_[timestamp].jpg`
- **Trigger**: Se captura cuando una persona permanece más de 60 segundos

## ⚙️ Configuración Avanzada

### Variables Configurables
```python
# En la clase PeopleCounter
self.confidence_threshold = 0.5    # Umbral de confianza (0.1-1.0)
self.classes_to_detect = [0]       # Clase 0 = personas
self.frame_skip = 1                # Procesar cada N frames
self.csv_interval = 300            # Guardar cada N segundos
```

### Modelos YOLO Disponibles
| Modelo | Tamaño | Velocidad | Precisión | Uso Recomendado |
|--------|--------|-----------|-----------|-----------------|
| yolov8n.pt | 6.2MB | ⚡⚡⚡⚡⚡ | ⭐⭐ | Desarrollo/Pruebas |
| yolov8s.pt | 22MB | ⚡⚡⚡⚡ | ⭐⭐⭐ | Uso general |
| yolov8m.pt | 50MB | ⚡⚡⚡ | ⭐⭐⭐⭐ | **Producción** |
| yolov8l.pt | 87MB | ⚡⚡ | ⭐⭐⭐⭐⭐ | Alta precisión |
| yolov8x.pt | 136MB | ⚡ | ⭐⭐⭐⭐⭐ | Máxima precisión |

## 🔧 Solución de Problemas

### Error: "No se pudo abrir la fuente de video"
- Verificar que la cámara esté conectada
- Comprobar permisos de acceso
- Para cámaras IP, verificar URL y credenciales

### Rendimiento Lento
- Cambiar a modelo YOLO más pequeño (nano o small)
- Reducir resolución de video
- Aumentar `frame_skip`
- Usar GPU si está disponible

### Detecciones Incorrectas
- Ajustar `confidence_threshold`
- Cambiar a modelo YOLO más grande
- Verificar iluminación del área

### Error de Memoria
- Reducir resolución de video
- Usar modelo YOLO más pequeño
- Cerrar otras aplicaciones

## 📈 Casos de Uso

### 🏪 Comercio Minorista
- **Análisis de tráfico** en diferentes secciones
- **Tiempo de permanencia** en vitrinas
- **Efectividad** de displays y promociones

### 🏢 Oficinas
- **Control de acceso** a áreas restringidas
- **Análisis de ocupación** de salas de reunión
- **Optimización** del uso de espacios

### 🏥 Hospitales
- **Monitoreo** de áreas de espera
- **Control de flujo** en pasillos
- **Análisis** de tiempos de espera

### 🏭 Industria
- **Seguridad** en áreas críticas
- **Control de acceso** a zonas restringidas
- **Análisis** de flujo de trabajadores

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crear una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abrir un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 👨‍💻 Autor

**AriasDmk** © 2025

## 🙏 Agradecimientos

- **Ultralytics** por YOLOv8
- **OpenCV** por el framework de visión por computadora
- **Comunidad de Python** por las librerías utilizadas

## 📞 Soporte

Para soporte técnico o consultas:
- Crear un issue en GitHub
- Contactar al desarrollador

---

**Versión**: 1.0.0  
**Última actualización**: Diciembre 2024  
**Compatibilidad**: Python 3.8+, Windows/Linux/Mac 