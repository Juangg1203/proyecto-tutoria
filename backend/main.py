import os

# Limitar hilos para reducir consumo de CPU
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import asyncio
import gc
import json
import logging
import time
import uuid
from pathlib import Path
from io import BytesIO

import cv2
import numpy as np
import onnxruntime as ort

from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
import psycopg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="API de ocupación de aulas")

# Permite consumir las rutas de lectura desde el dashboard web.
# Para restringirlo posteriormente, define CORS_ORIGINS en Render
# con los dominios separados por comas.
CORS_ORIGINS = [
    origen.strip()
    for origen in os.getenv("CORS_ORIGINS", "*").split(",")
    if origen.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def _leer_float(nombre, defecto):
    """Lee un número decimal de las variables de entorno de Render."""
    try:
        return float(os.getenv(nombre, str(defecto)))
    except ValueError:
        return float(defecto)


def _leer_int(nombre, defecto):
    """Lee un número entero de las variables de entorno de Render."""
    try:
        return int(os.getenv(nombre, str(defecto)))
    except ValueError:
        return int(defecto)


DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
APP_VERSION = "onnx-supabase-ram-v2"

MAX_IMAGE_BYTES = 8 * 1024 * 1024

# ============================================================
# AJUSTES (todos se pueden cambiar desde Render > Environment
# sin tocar el código; si no existen, se usa el valor por defecto)
# ============================================================

# --- Personas (YOLO) ---
# Confianza mínima para aceptar una persona. Más bajo = detecta más
# personas lejanas, pero también puede inventar alguna.
CONF_PERSONA = _leer_float("CONF_PERSONA", 0.25)
NMS_THRESHOLD = _leer_float("NMS_THRESHOLD", 0.45)
# Tamaño de entrada de YOLO. SOLO se usa si el modelo .onnx acepta
# tamaños variables; si el modelo es de tamaño fijo se usa el suyo.
INPUT_SIZE_SOLICITADO = _leer_int("INPUT_SIZE", 640)
ONNX_THREADS = max(1, _leer_int("ONNX_THREADS", 1))

# --- Sillas verdes (OpenCV HSV: H 0-179, S 0-255, V 0-255) ---
VERDE_MIN = np.array(
    [
        _leer_int("VERDE_H_MIN", 35),
        _leer_int("VERDE_S_MIN", 60),
        _leer_int("VERDE_V_MIN", 50),
    ],
    dtype=np.uint8,
)
VERDE_MAX = np.array(
    [_leer_int("VERDE_H_MAX", 85), 255, 255],
    dtype=np.uint8,
)
# Área mínima de una silla, como fracción del área de la imagen
SILLA_AREA_MIN_FRAC = _leer_float("SILLA_AREA_MIN_FRAC", 0.0010)
# Área máxima (una mancha verde enorme es una pared, no una silla)
SILLA_AREA_MAX_FRAC = _leer_float("SILLA_AREA_MAX_FRAC", 0.25)
# Lado mínimo (en píxeles) de la caja de una silla
SILLA_LADO_MIN = _leer_int("SILLA_LADO_MIN", 10)
# Si la caja es mucho más larga que ancha (o al revés) no es una silla
SILLA_ASPECTO_MAX = _leer_float("SILLA_ASPECTO_MAX", 3.5)
# Qué parte de la caja debe ser verde (descarta líneas finas y marcos)
SILLA_RELLENO_MIN = _leer_float("SILLA_RELLENO_MIN", 0.15)
# Si dos manchas verdes se solapan más de esta fracción de la menor,
# se consideran la misma silla partida en trozos
SILLA_FUSION_SOLAPE = _leer_float("SILLA_FUSION_SOLAPE", 0.50)

prediction_lock = asyncio.Lock()

MODEL_PATH = Path(__file__).resolve().parent / "yolo11n.onnx"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"No se encontró el modelo ONNX: {MODEL_PATH}"
    )


def leer_memoria_proceso_mb():
    """
    Lee la RAM residente actual (VmRSS) y el máximo
    residente desde el inicio del proceso (VmHWM).

    Está pensado para Linux, como el entorno de Render.
    Estas cifras corresponden al proceso de Python, no
    al consumo total del contenedor.
    """

    datos = {
        "rss_mb": None,
        "hwm_mb": None
    }

    try:
        with open("/proc/self/status", "r", encoding="utf-8") as archivo:
            for linea in archivo:
                if linea.startswith("VmRSS:"):
                    datos["rss_mb"] = round(
                        int(linea.split()[1]) / 1024, 2
                    )

                elif linea.startswith("VmHWM:"):
                    datos["hwm_mb"] = round(
                        int(linea.split()[1]) / 1024, 2
                    )

                if (
                    datos["rss_mb"] is not None
                    and datos["hwm_mb"] is not None
                ):
                    break

    except (OSError, ValueError, IndexError):
        pass

    return datos


async def muestrear_memoria(stop_event, tracker):
    """
    Muestrea la memoria residente aproximadamente cada
    100 ms mientras se ejecuta una petición.
    """

    while not stop_event.is_set():
        memoria = leer_memoria_proceso_mb()
        rss = memoria["rss_mb"]

        if rss is not None:
            actual = tracker.get("pico_rss_mb")

            if actual is None or rss > actual:
                tracker["pico_rss_mb"] = rss

        try:
            await asyncio.wait_for(
                stop_event.wait(),
                timeout=0.1
            )
        except asyncio.TimeoutError:
            pass


# Configurar ONNX Runtime con CPU y pocos hilos de ejecución
options = ort.SessionOptions()
options.intra_op_num_threads = ONNX_THREADS
options.inter_op_num_threads = 1
options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
options.graph_optimization_level = (
    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
)

inicio_carga = time.perf_counter()

session = ort.InferenceSession(
    str(MODEL_PATH),
    sess_options=options,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_names = [item.name for item in session.get_outputs()]

CARGA_MODELO_MS = round(
    (time.perf_counter() - inicio_carga) * 1000, 2
)


def determinar_tamano_entrada():
    """
    Averigua con qué tamaño trabaja el modelo.

    - Si el .onnx fue exportado con tamaño fijo (por ejemplo 320 o 640),
      es OBLIGATORIO usar ese tamaño: devuelve (tamaño, True).
    - Si el modelo acepta tamaños variables (dynamic), se usa el valor de
      INPUT_SIZE (múltiplo de 32): devuelve (tamaño, False).
    """
    forma = session.get_inputs()[0].shape

    try:
        alto, ancho = forma[2], forma[3]
    except (IndexError, TypeError):
        alto, ancho = None, None

    if isinstance(alto, int) and isinstance(ancho, int):
        if alto != ancho:
            raise RuntimeError(
                f"El modelo tiene una entrada fija no cuadrada ({alto}x{ancho}). "
                "Expórtalo de nuevo con un imgsz cuadrado, por ejemplo 640."
            )
        return alto, True

    solicitado = max(32, (INPUT_SIZE_SOLICITADO // 32) * 32)
    return solicitado, False


INPUT_SIZE, MODELO_TAMANO_FIJO = determinar_tamano_entrada()

# Lado máximo de la imagen con la que se trabaja (personas y sillas
# usan la MISMA imagen, así las coordenadas coinciden).
MAX_LADO_IMAGEN = max(640, INPUT_SIZE)

logger.info(json.dumps({
    "evento": "modelo_onnx_cargado",
    "version_api": APP_VERSION,
    "modelo": MODEL_PATH.name,
    "carga_modelo_ms": CARGA_MODELO_MS,
    "memoria_inicio_mb": leer_memoria_proceso_mb(),
    "input_shape": session.get_inputs()[0].shape,
    "tamano_entrada_usado": INPUT_SIZE,
    "tamano_fijo": MODELO_TAMANO_FIJO,
    "output_shapes": [
        item.shape for item in session.get_outputs()
    ]
}))

if INPUT_SIZE <= 416:
    logger.warning(json.dumps({
        "evento": "modelo_resolucion_baja",
        "tamano_entrada": INPUT_SIZE,
        "aviso": (
            "El modelo trabaja a baja resolución y no verá personas "
            "lejanas. Exporta yolo11n.onnx con imgsz=640."
        )
    }))


def calentar_modelo():
    """Una inferencia de prueba al arrancar, para que la primera foto real no sea lenta."""
    try:
        inicio = time.perf_counter()
        session.run(
            output_names,
            {
                input_name: np.zeros(
                    (1, 3, INPUT_SIZE, INPUT_SIZE), dtype=np.float32
                )
            }
        )
        logger.info(json.dumps({
            "evento": "modelo_calentado",
            "ms": round((time.perf_counter() - inicio) * 1000, 2)
        }))
    except Exception:
        logger.exception("No se pudo calentar el modelo (se continúa igual)")


calentar_modelo()


@app.get("/")
def inicio():
    return {
        "mensaje": "Servidor de tutoria funcionando",
        "modelo": "YOLO11n ONNX + OpenCV",
        "version_api": APP_VERSION,
        "carga_modelo_ms": CARGA_MODELO_MS,
        "tamano_entrada_modelo": INPUT_SIZE
    }


@app.get("/health")
def health():
    return {
        "estado": "ok",
        "modelo": "YOLO11n ONNX + OpenCV",
        "version_api": APP_VERSION,
        "base_datos_configurada": bool(DATABASE_URL),
        "tamano_entrada_modelo": INPUT_SIZE,
        "tamano_fijo": MODELO_TAMANO_FIJO,
        "conf_persona": CONF_PERSONA
    }


def abrir_conexion_bd():
    """Abre una conexión segura a PostgreSQL mediante DATABASE_URL."""
    if not DATABASE_URL:
        raise RuntimeError("La variable DATABASE_URL no está configurada en Render.")

    return psycopg.connect(
        DATABASE_URL,
        connect_timeout=8,
        sslmode="require",
        prepare_threshold=None
    )


def guardar_registro_bd(respuesta, origen="esp32-cam"):
    """Guarda únicamente los números del análisis, nunca la fotografía."""
    origen_limpio = (origen or "esp32-cam").strip()[:80] or "esp32-cam"

    with abrir_conexion_bd() as conexion:
        with conexion.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO public.registros_ocupacion
                    (personas, sillas, ocupadas, libres, estado, origen)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id, fecha_hora
                """,
                (
                    int(respuesta["personas"]),
                    int(respuesta["sillas"]),
                    int(respuesta["ocupadas"]),
                    int(respuesta["libres"]),
                    str(respuesta["estado"]),
                    origen_limpio,
                ),
            )
            fila = cursor.fetchone()

    return {
        "id": fila[0],
        "fecha_hora": fila[1].isoformat() if fila[1] else None,
        "origen": origen_limpio,
    }


def fila_a_registro(fila):
    return {
        "id": fila[0],
        "fecha_hora": fila[1].isoformat() if fila[1] else None,
        "personas": fila[2],
        "sillas": fila[3],
        "ocupadas": fila[4],
        "libres": fila[5],
        "estado": fila[6],
        "origen": fila[7],
    }


def consultar_registros_bd(limite):
    with abrir_conexion_bd() as conexion:
        with conexion.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, fecha_hora, personas, sillas, ocupadas,
                       libres, estado, origen
                FROM public.registros_ocupacion
                ORDER BY fecha_hora DESC, id DESC
                LIMIT %s
                """,
                (limite,),
            )
            filas = cursor.fetchall()

    return [fila_a_registro(fila) for fila in filas]


@app.get("/ocupacion/actual")
def ocupacion_actual():
    """Devuelve el registro más reciente para el dashboard."""
    try:
        registros = consultar_registros_bd(1)
    except Exception:
        logger.exception(json.dumps({
            "evento": "consulta_bd_error",
            "ruta": "/ocupacion/actual",
            "version_api": APP_VERSION,
        }))
        raise HTTPException(
            status_code=503,
            detail="No se pudo consultar la base de datos. Revisa la configuración y los logs."
        )

    if not registros:
        return {"encontrado": False, "registro": None}

    return {"encontrado": True, "registro": registros[0]}


@app.get("/ocupacion/historial")
def ocupacion_historial(limite: int = Query(default=100, ge=1, le=500)):
    """Devuelve hasta 500 registros recientes para gráficos e historial."""
    try:
        registros = consultar_registros_bd(limite)
    except Exception:
        logger.exception(json.dumps({
            "evento": "consulta_bd_error",
            "ruta": "/ocupacion/historial",
            "version_api": APP_VERSION,
        }))
        raise HTTPException(
            status_code=503,
            detail="No se pudo consultar la base de datos. Revisa la configuración y los logs."
        )

    return {"cantidad": len(registros), "registros": registros}


def decodificar_imagen(image_bytes):
    """
    Convierte los bytes recibidos en una imagen RGB y la reduce (si hace
    falta) a MAX_LADO_IMAGEN píxeles en su lado mayor, con un filtro de
    buena calidad para no perder detalle de las personas lejanas.
    """
    with Image.open(BytesIO(image_bytes)) as original:
        imagen = original.convert("RGB")

    imagen.thumbnail((MAX_LADO_IMAGEN, MAX_LADO_IMAGEN), Image.LANCZOS)
    return imagen


def preparar_entrada(imagen_rgb):
    """Redimensiona manteniendo proporciones y rellena hasta INPUT_SIZE x INPUT_SIZE."""

    inicio = time.perf_counter()

    alto, ancho = imagen_rgb.shape[:2]
    escala = min(INPUT_SIZE / ancho, INPUT_SIZE / alto)

    nuevo_ancho = max(1, int(round(ancho * escala)))
    nuevo_alto = max(1, int(round(alto * escala)))

    # INTER_AREA conserva mejor el detalle al reducir; INTER_LINEAR al ampliar.
    interpolacion = cv2.INTER_AREA if escala < 1 else cv2.INTER_LINEAR

    redimensionada = cv2.resize(
        imagen_rgb,
        (nuevo_ancho, nuevo_alto),
        interpolation=interpolacion
    )

    izquierda = (INPUT_SIZE - nuevo_ancho) // 2
    arriba = (INPUT_SIZE - nuevo_alto) // 2
    derecha = INPUT_SIZE - nuevo_ancho - izquierda
    abajo = INPUT_SIZE - nuevo_alto - arriba

    entrada = cv2.copyMakeBorder(
        redimensionada,
        arriba,
        abajo,
        izquierda,
        derecha,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114)
    )

    entrada = entrada.astype(np.float32) / 255.0
    entrada = np.transpose(entrada, (2, 0, 1))
    entrada = np.expand_dims(entrada, axis=0)
    entrada = np.ascontiguousarray(entrada)

    escala_x = nuevo_ancho / ancho
    escala_y = nuevo_alto / alto

    tiempo_ms = round(
        (time.perf_counter() - inicio) * 1000, 2
    )

    return (
        entrada,
        escala_x,
        escala_y,
        izquierda,
        arriba,
        tiempo_ms
    )


def detectar_personas(imagen):
    """Ejecuta ONNX Runtime y devuelve personas detectadas."""

    with imagen.convert("RGB") as rgb:
        imagen_rgb = np.array(rgb)

    alto_original, ancho_original = imagen_rgb.shape[:2]

    (
        entrada,
        escala_x,
        escala_y,
        izquierda,
        arriba,
        preparacion_ms
    ) = preparar_entrada(imagen_rgb)

    inicio_inferencia = time.perf_counter()

    outputs = session.run(
        output_names,
        {input_name: entrada}
    )

    inferencia_ms = round(
        (time.perf_counter() - inicio_inferencia) * 1000, 2
    )

    inicio_postproceso = time.perf_counter()

    predicciones = np.asarray(outputs[0])

    if predicciones.ndim == 3:
        predicciones = predicciones[0]

    # YOLO11n COCO debe devolver 4 coordenadas y 80 clases.
    if predicciones.shape[0] < predicciones.shape[1]:
        predicciones = predicciones.T

    if predicciones.shape[1] != 84:
        raise RuntimeError(
            f"Formato de salida ONNX inesperado: {predicciones.shape}"
        )

    # En este modelo exportado, la columna 4 contiene
    # la puntuación de la clase persona (clase COCO 0).
    puntuaciones = predicciones[:, 4]
    indices = np.flatnonzero(
        puntuaciones >= CONF_PERSONA
    )

    cajas_nms = []
    confianzas = []
    cajas_originales = []

    for indice in indices:
        centro_x, centro_y, ancho, alto = (
            predicciones[indice, :4].astype(float)
        )

        x1_modelo = centro_x - ancho / 2
        y1_modelo = centro_y - alto / 2
        x2_modelo = centro_x + ancho / 2
        y2_modelo = centro_y + alto / 2

        cajas_nms.append([
            int(round(x1_modelo)),
            int(round(y1_modelo)),
            max(1, int(round(ancho))),
            max(1, int(round(alto)))
        ])

        confianzas.append(float(puntuaciones[indice]))

        x1 = (x1_modelo - izquierda) / escala_x
        y1 = (y1_modelo - arriba) / escala_y
        x2 = (x2_modelo - izquierda) / escala_x
        y2 = (y2_modelo - arriba) / escala_y

        cajas_originales.append([
            float(np.clip(x1, 0, ancho_original)),
            float(np.clip(y1, 0, alto_original)),
            float(np.clip(x2, 0, ancho_original)),
            float(np.clip(y2, 0, alto_original))
        ])

    personas = []

    if cajas_nms:
        indices_nms = cv2.dnn.NMSBoxes(
            cajas_nms,
            confianzas,
            CONF_PERSONA,
            NMS_THRESHOLD
        )

        if indices_nms is not None and len(indices_nms) > 0:
            for indice in np.asarray(indices_nms).reshape(-1):
                x1, y1, x2, y2 = cajas_originales[int(indice)]

                if x2 <= x1 or y2 <= y1:
                    continue

                personas.append({
                    "id": len(personas) + 1,
                    "confianza": round(
                        confianzas[int(indice)], 4
                    ),
                    "box": [x1, y1, x2, y2]
                })

    postproceso_ms = round(
        (time.perf_counter() - inicio_postproceso) * 1000, 2
    )

    tiempos = {
        "preparacion_entrada_ms": preparacion_ms,
        "yolo_inferencia_ms": inferencia_ms,
        "yolo_postproceso_ms": postproceso_ms,
        "yolo_total_ms": round(
            preparacion_ms + inferencia_ms + postproceso_ms, 2
        )
    }

    del outputs, predicciones, entrada, imagen_rgb
    gc.collect()

    return personas, tiempos


def interseccion_cajas(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    return (
        max(0.0, x2 - x1)
        * max(0.0, y2 - y1)
    )


def area_caja(box):
    x1, y1, x2, y2 = box
    return (
        max(0.0, x2 - x1)
        * max(0.0, y2 - y1)
    )


def fusionar_fragmentos(piezas):
    """
    Une las manchas verdes que son trozos de una misma silla.

    Una silla puede quedar partida en varias manchas (respaldo y asiento
    separados por una pata, un reflejo o una persona sentada). Si la caja
    de una mancha se solapa en SILLA_FUSION_SOLAPE o más con la caja de
    la otra (medido contra la más pequeña), se funden en una sola silla.

    Dos sillas vecinas que solo se tocan por el borde NO se unen, porque
    casi no se solapan.
    """
    piezas = [dict(p, box=list(p["box"])) for p in piezas]

    hubo_cambio = True
    while hubo_cambio:
        hubo_cambio = False

        for i in range(len(piezas)):
            for j in range(i + 1, len(piezas)):
                a = piezas[i]
                b = piezas[j]

                interseccion = interseccion_cajas(a["box"], b["box"])
                menor = max(
                    min(area_caja(a["box"]), area_caja(b["box"])),
                    1.0
                )

                if interseccion / menor >= SILLA_FUSION_SOLAPE:
                    a["box"] = [
                        min(a["box"][0], b["box"][0]),
                        min(a["box"][1], b["box"][1]),
                        max(a["box"][2], b["box"][2]),
                        max(a["box"][3], b["box"][3])
                    ]
                    a["area_verde"] += b["area_verde"]
                    del piezas[j]
                    hubo_cambio = True
                    break

            if hubo_cambio:
                break

    return piezas


def detectar_sillas_verdes(imagen):
    """Detecta sillas verdes con OpenCV (color + filtros de forma)."""

    inicio = time.perf_counter()

    with imagen.convert("RGB") as rgb:
        imagen_rgb = np.array(rgb)

    alto, ancho = imagen_rgb.shape[:2]
    area_imagen = max(ancho * alto, 1)

    imagen_hsv = cv2.cvtColor(
        imagen_rgb,
        cv2.COLOR_RGB2HSV
    )

    mascara = cv2.inRange(
        imagen_hsv,
        VERDE_MIN,
        VERDE_MAX
    )

    # Quita puntos sueltos de ruido
    mascara = cv2.morphologyEx(
        mascara,
        cv2.MORPH_OPEN,
        np.ones((3, 3), dtype=np.uint8)
    )

    # Rellena huecos pequeños. Se mantiene en 5x5: un valor mayor pegaría
    # entre sí a dos sillas vecinas y las contaría como una sola.
    mascara = cv2.morphologyEx(
        mascara,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), dtype=np.uint8)
    )

    cantidad, etiquetas, estadisticas, centroides = (
        cv2.connectedComponentsWithStats(
            mascara,
            connectivity=8
        )
    )

    area_minima = max(
        120,
        int(area_imagen * SILLA_AREA_MIN_FRAC)
    )
    area_maxima = int(area_imagen * SILLA_AREA_MAX_FRAC)

    candidatas = []
    descartados = []
    conteo_descartes = {}

    def descartar(motivo, area, x, y, w, h):
        conteo_descartes[motivo] = conteo_descartes.get(motivo, 0) + 1

        # Solo se detallan los descartes con cierto tamaño (el ruido
        # diminuto no se lista para no llenar la respuesta).
        if area >= area_minima // 2:
            descartados.append({
                "area_pixeles": area,
                "caja": [x, y, x + w, y + h],
                "motivo": motivo
            })

    for indice in range(1, cantidad):
        x = int(estadisticas[indice, cv2.CC_STAT_LEFT])
        y = int(estadisticas[indice, cv2.CC_STAT_TOP])
        w = int(estadisticas[indice, cv2.CC_STAT_WIDTH])
        h = int(estadisticas[indice, cv2.CC_STAT_HEIGHT])
        area = int(estadisticas[indice, cv2.CC_STAT_AREA])

        if (
            area < area_minima
            or w < SILLA_LADO_MIN
            or h < SILLA_LADO_MIN
        ):
            descartar("componente_demasiado_pequeno", area, x, y, w, h)
            continue

        if area > area_maxima:
            descartar("componente_demasiado_grande", area, x, y, w, h)
            continue

        aspecto = max(w / h, h / w)
        if aspecto > SILLA_ASPECTO_MAX:
            descartar("forma_alargada", area, x, y, w, h)
            continue

        # Fracción de la caja que realmente es verde. El área del
        # componente ya es exactamente el número de píxeles verdes.
        relleno = area / max(w * h, 1)
        if relleno < SILLA_RELLENO_MIN:
            descartar("poco_relleno_verde", area, x, y, w, h)
            continue

        candidatas.append({
            "box": [float(x), float(y), float(x + w), float(y + h)],
            "area_verde": area
        })

    fusionadas = fusionar_fragmentos(candidatas)

    sillas = []
    for pieza in fusionadas:
        x1, y1, x2, y2 = pieza["box"]
        area_de_la_caja = max(area_caja(pieza["box"]), 1.0)

        sillas.append({
            "id": len(sillas) + 1,
            "box": [x1, y1, x2, y2],
            "area_verde": int(pieza["area_verde"]),
            "proporcion_verde": round(
                min(pieza["area_verde"] / area_de_la_caja, 1.0), 3
            ),
            "centro": [
                round((x1 + x2) / 2, 1),
                round((y1 + y2) / 2, 1)
            ]
        })

    descartados.sort(key=lambda d: d["area_pixeles"], reverse=True)

    diagnostico = {
        "componentes_verdes_totales": cantidad - 1,
        "area_minima_pixeles": area_minima,
        "area_maxima_pixeles": area_maxima,
        "fragmentos_fusionados": len(candidatas) - len(sillas),
        "descartes_por_motivo": conteo_descartes,
        "componentes_descartados": descartados[:30]
    }

    tiempo_ms = round(
        (time.perf_counter() - inicio) * 1000, 2
    )

    del imagen_rgb, imagen_hsv, mascara, etiquetas

    return sillas, diagnostico, tiempo_ms


def relacion_persona_silla(persona_box, silla_box):
    px1, py1, px2, py2 = persona_box
    sx1, sy1, sx2, sy2 = silla_box

    ancho_silla = max(sx2 - sx1, 1)
    alto_silla = max(sy2 - sy1, 1)

    silla_expandida = [
        sx1 - ancho_silla * 0.20,
        sy1 - alto_silla * 0.15,
        sx2 + ancho_silla * 0.20,
        sy2 + alto_silla * 0.15
    ]

    alto_persona = max(py2 - py1, 1)

    zona_inferior = [
        px1,
        py1 + alto_persona * 0.35,
        px2,
        py2
    ]

    inter_total = interseccion_cajas(
        persona_box,
        silla_expandida
    )

    inter_inferior = interseccion_cajas(
        zona_inferior,
        silla_expandida
    )

    area_persona = max(area_caja(persona_box), 1)
    area_silla = max(area_caja(silla_expandida), 1)

    return max(
        inter_total / min(area_persona, area_silla),
        inter_inferior / area_silla
    )


def asociar_personas_sillas(personas, sillas):
    candidatos = []

    for persona in personas:
        for silla in sillas:
            puntuacion = relacion_persona_silla(
                persona["box"],
                silla["box"]
            )

            if puntuacion >= 0.05:
                candidatos.append({
                    "persona": persona["id"],
                    "silla": silla["id"],
                    "puntuacion": round(puntuacion, 3)
                })

    candidatos.sort(
        key=lambda item: item["puntuacion"],
        reverse=True
    )

    personas_asignadas = set()
    sillas_asignadas = set()
    asignaciones = []

    for candidato in candidatos:
        persona_id = candidato["persona"]
        silla_id = candidato["silla"]

        asignada = (
            persona_id not in personas_asignadas
            and silla_id not in sillas_asignadas
        )

        candidato["asignada"] = asignada

        if asignada:
            personas_asignadas.add(persona_id)
            sillas_asignadas.add(silla_id)
            asignaciones.append(dict(candidato))

    return (
        len(sillas_asignadas),
        asignaciones,
        candidatos,
        [
            p["id"] for p in personas
            if p["id"] not in personas_asignadas
        ],
        [
            s["id"] for s in sillas
            if s["id"] not in sillas_asignadas
        ]
    )


def analizar_imagen(imagen):
    inicio_analisis = time.perf_counter()
    memoria_inicio = leer_memoria_proceso_mb()

    personas, tiempos_yolo = detectar_personas(imagen)
    memoria_despues_yolo = leer_memoria_proceso_mb()

    sillas, diagnostico_verde, opencv_ms = (
        detectar_sillas_verdes(imagen)
    )
    memoria_despues_opencv = leer_memoria_proceso_mb()

    inicio_asociacion = time.perf_counter()

    (
        ocupadas,
        asignaciones,
        coincidencias,
        personas_sin_silla,
        sillas_sin_persona
    ) = asociar_personas_sillas(personas, sillas)

    asociacion_ms = round(
        (time.perf_counter() - inicio_asociacion) * 1000,
        2
    )

    memoria_despues_asociacion = leer_memoria_proceso_mb()

    total_personas = len(personas)
    total_sillas = len(sillas)
    libres = max(total_sillas - ocupadas, 0)

    tiempos = dict(tiempos_yolo)
    tiempos["opencv_sillas_ms"] = opencv_ms
    tiempos["asociacion_ms"] = asociacion_ms

    # IMPORTANTE: las claves personas, sillas, ocupadas, libres y estado
    # deben seguir al principio. La ESP32-CAM solo lee los primeros bytes
    # de la respuesta para mostrarlas en el Monitor Serie.
    respuesta = {
        "version_api": APP_VERSION,
        "personas": total_personas,
        "sillas": total_sillas,
        "ocupadas": ocupadas,
        "libres": libres,
        "estado": "vacio" if total_personas == 0 else "ocupado",
        "diagnostico": {
            "metodo_personas": "YOLO11n ONNX Runtime",
            "metodo_sillas": "OpenCV HSV verde",
            "inferencias_yolo": 1,
            "tamano_entrada_modelo": INPUT_SIZE,
            "conf_persona": CONF_PERSONA,
            "tamano_imagen_analizada": [imagen.width, imagen.height],
            "personas_detectadas": [
                {
                    "id": p["id"],
                    "confianza": p["confianza"],
                    "caja": [round(v, 1) for v in p["box"]]
                }
                for p in personas
            ],
            "sillas_detectadas": [
                {
                    "id": s["id"],
                    "caja": [round(v, 1) for v in s["box"]],
                    "area_verde": s["area_verde"],
                    "proporcion_verde": s["proporcion_verde"]
                }
                for s in sillas
            ],
            "deteccion_color": diagnostico_verde,
            "asignaciones": asignaciones,
            "coincidencias_evaluadas": coincidencias,
            "personas_sin_silla": personas_sin_silla,
            "sillas_sin_persona": sillas_sin_persona
        },
        "tiempos_ms": tiempos,
        "memoria_mb": {
            "antes_yolo_rss": memoria_inicio["rss_mb"],
            "despues_yolo_rss": memoria_despues_yolo["rss_mb"],
            "despues_opencv_rss": memoria_despues_opencv["rss_mb"],
            "despues_asociacion_rss": memoria_despues_asociacion["rss_mb"]
        }
    }

    del personas, sillas
    gc.collect()

    memoria_final = leer_memoria_proceso_mb()
    respuesta["memoria_mb"]["despues_gc_rss"] = memoria_final["rss_mb"]
    respuesta["memoria_mb"]["pico_proceso_desde_inicio_hwm"] = (
        memoria_final["hwm_mb"]
    )

    tiempos["analisis_total_ms"] = round(
        (time.perf_counter() - inicio_analisis) * 1000, 2
    )

    return respuesta


ETIQUETAS_DESCARTE = {
    "componente_demasiado_pequeno": "pequeno",
    "componente_demasiado_grande": "grande",
    "forma_alargada": "alargado",
    "poco_relleno_verde": "poco verde",
}


def dibujar_resultado(imagen, respuesta):
    """
    Devuelve la foto en JPEG con lo que detectó el sistema:
      - ROJO    = personas (con su confianza)
      - VERDE   = sillas aceptadas
      - NARANJA = manchas verdes descartadas (con el motivo)
    Sirve para calibrar el sistema; no se guarda en ningún lado.
    """
    lienzo = cv2.cvtColor(np.array(imagen), cv2.COLOR_RGB2BGR)
    diagnostico = respuesta["diagnostico"]
    fuente = cv2.FONT_HERSHEY_SIMPLEX

    for d in diagnostico["deteccion_color"]["componentes_descartados"]:
        x1, y1, x2, y2 = [int(v) for v in d["caja"]]
        cv2.rectangle(lienzo, (x1, y1), (x2, y2), (0, 165, 255), 1)
        cv2.putText(
            lienzo,
            ETIQUETAS_DESCARTE.get(d["motivo"], d["motivo"]),
            (x1, max(y1 - 3, 12)),
            fuente, 0.4, (0, 165, 255), 1, cv2.LINE_AA
        )

    for s in diagnostico["sillas_detectadas"]:
        x1, y1, x2, y2 = [int(v) for v in s["caja"]]
        cv2.rectangle(lienzo, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            lienzo, f"S{s['id']}", (x1 + 3, y1 + 16),
            fuente, 0.5, (0, 255, 0), 1, cv2.LINE_AA
        )

    for p in diagnostico["personas_detectadas"]:
        x1, y1, x2, y2 = [int(v) for v in p["caja"]]
        cv2.rectangle(lienzo, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            lienzo, f"P{p['id']} {p['confianza']:.2f}", (x1 + 3, y2 - 6),
            fuente, 0.5, (0, 0, 255), 1, cv2.LINE_AA
        )

    texto = (
        f"Personas {respuesta['personas']} | Sillas {respuesta['sillas']} | "
        f"Ocupadas {respuesta['ocupadas']} | Libres {respuesta['libres']}"
    )
    cv2.rectangle(lienzo, (0, 0), (lienzo.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(
        lienzo, texto, (6, 16),
        fuente, 0.5, (255, 255, 255), 1, cv2.LINE_AA
    )

    correcto, buffer = cv2.imencode(
        ".jpg", lienzo, [cv2.IMWRITE_JPEG_QUALITY, 85]
    )
    if not correcto:
        raise RuntimeError("No se pudo codificar la imagen de depuración")

    return buffer.tobytes()


@app.post("/debug/imagen")
async def debug_imagen(file: UploadFile = File(...)):
    """
    Herramienta de calibración: recibe una foto y devuelve la misma foto
    con las personas y sillas dibujadas. NO guarda nada en la base de datos.
    """
    imagen = None

    try:
        image_bytes = await file.read(MAX_IMAGE_BYTES + 1)

        if not image_bytes:
            raise HTTPException(status_code=400, detail="La imagen está vacía")

        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise HTTPException(
                status_code=413,
                detail="La imagen supera el límite de 8 MB"
            )

        try:
            imagen = decodificar_imagen(image_bytes)
        except UnidentifiedImageError:
            raise HTTPException(
                status_code=400,
                detail="El archivo recibido no es una imagen válida"
            )

        async with prediction_lock:
            respuesta = await asyncio.to_thread(analizar_imagen, imagen)

        jpeg = await asyncio.to_thread(dibujar_resultado, imagen, respuesta)

        return Response(
            content=jpeg,
            media_type="image/jpeg",
            headers={
                "X-Personas": str(respuesta["personas"]),
                "X-Sillas": str(respuesta["sillas"]),
                "X-Ocupadas": str(respuesta["ocupadas"]),
                "X-Libres": str(respuesta["libres"]),
            },
        )

    finally:
        if imagen is not None:
            imagen.close()

        await file.close()
        gc.collect()


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    origen: str = Form(default="esp32-cam")
):
    inicio_api = time.perf_counter()
    request_id = uuid.uuid4().hex[:8]

    imagen = None
    image_bytes = b""
    stop_event = None
    sampler_task = None
    tracker = {"pico_rss_mb": None}

    try:
        inicio_lectura = time.perf_counter()
        image_bytes = await file.read(MAX_IMAGE_BYTES + 1)

        lectura_ms = round(
            (time.perf_counter() - inicio_lectura) * 1000, 2
        )

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="La imagen está vacía"
            )

        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise HTTPException(
                status_code=413,
                detail="La imagen supera el límite de 8 MB"
            )

        inicio_decodificacion = time.perf_counter()

        imagen = decodificar_imagen(image_bytes)

        decodificacion_ms = round(
            (time.perf_counter() - inicio_decodificacion) * 1000,
            2
        )

        inicio_espera = time.perf_counter()

        async with prediction_lock:
            espera_lock_ms = round(
                (time.perf_counter() - inicio_espera) * 1000, 2
            )

            memoria_antes = leer_memoria_proceso_mb()
            tracker["pico_rss_mb"] = memoria_antes["rss_mb"]

            stop_event = asyncio.Event()
            sampler_task = asyncio.create_task(
                muestrear_memoria(stop_event, tracker)
            )

            try:
                respuesta = await asyncio.to_thread(
                    analizar_imagen,
                    imagen
                )
            finally:
                stop_event.set()
                await sampler_task
                sampler_task = None

        memoria_despues = leer_memoria_proceso_mb()

        respuesta["memoria_mb"].update({
            "antes_analisis_rss": memoria_antes["rss_mb"],
            "despues_analisis_rss": memoria_despues["rss_mb"],
            "pico_muestreado_solicitud_rss": tracker["pico_rss_mb"],
            "pico_proceso_desde_inicio_hwm": memoria_despues["hwm_mb"]
        })

        tiempos = respuesta["tiempos_ms"]
        tiempos["lectura_bytes_ms"] = lectura_ms
        tiempos["decodificacion_imagen_ms"] = decodificacion_ms
        tiempos["espera_turno_ms"] = espera_lock_ms

        # La petición solo se considera exitosa si el registro quedó guardado.
        # Se almacena el conteo y metadatos mínimos, nunca los bytes de la foto.
        inicio_guardado_bd = time.perf_counter()
        try:
            registro_bd = await asyncio.to_thread(
                guardar_registro_bd,
                respuesta,
                origen,
            )
        except Exception:
            logger.exception(json.dumps({
                "evento": "guardar_bd_error",
                "version_api": APP_VERSION,
                "request_id": request_id,
                "personas": respuesta.get("personas"),
                "sillas": respuesta.get("sillas"),
            }))
            raise HTTPException(
                status_code=503,
                detail=(
                    "El análisis terminó, pero no se pudo guardar en Supabase. "
                    "Revisa DATABASE_URL, la conexión y los logs de Render."
                ),
            )

        tiempos["guardar_bd_ms"] = round(
            (time.perf_counter() - inicio_guardado_bd) * 1000, 2
        )
        tiempos["api_total_ms"] = round(
            (time.perf_counter() - inicio_api) * 1000, 2
        )
        respuesta["registro_bd"] = {
            "guardado": True,
            **registro_bd,
        }

        logger.info(json.dumps({
            "evento": "predict_completado",
            "version_api": APP_VERSION,
            "request_id": request_id,
            "tiempos_ms": tiempos,
            "memoria_mb": respuesta["memoria_mb"],
            "personas": respuesta["personas"],
            "sillas": respuesta["sillas"],
            "registro_bd": respuesta["registro_bd"],
        }))

        return respuesta

    except HTTPException as error:
        logger.warning(json.dumps({
            "evento": "predict_http_error",
            "version_api": APP_VERSION,
            "request_id": request_id,
            "status_code": error.status_code
        }))
        raise

    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="El archivo recibido no es una imagen válida"
        )

    except Exception:
        logger.exception(json.dumps({
            "evento": "predict_error",
            "version_api": APP_VERSION,
            "request_id": request_id
        }))

        raise HTTPException(
            status_code=500,
            detail="Error procesando la imagen. Revisa los logs."
        )

    finally:
        if stop_event is not None:
            stop_event.set()

        if sampler_task is not None:
            try:
                await sampler_task
            except Exception:
                pass

        if imagen is not None:
            imagen.close()

        image_bytes = b""
        await file.close()
        gc.collect()
