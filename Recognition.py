import time
import tensorflow as tf
# from PIL import Image
import cv2
import numpy as np
import Func as FC

# Limites de color en HSV (H-> elige color, S-> que tanto color de blanco a el color vivo, V-> que tanto color de negro al color)
# Los limites fueron elegidos para el filtro Gaussian Blur
Rojo = [(160,100,45),(180,255,255)]
Azul = [[90,100,45],[130,255,255]]
Colores = Rojo+ Azul
Colores = FC.ConvertirToNumpyArray(Colores)

# Cargar modelo pre-entrenado
# saved_model_loaded = tf.saved_model.load("yolov4-416")
saved_model_loaded = tf.saved_model.load("yolov4-SC-416")
infer = saved_model_loaded.signatures['serving_default']
file_names = "obj.names"
#****************** Preparando RealSense **********************************

# Preparar transmision de realsense a sistemas
FPS = 30 #Numero de fotogramas deseado  (maximo 30)
pipeline,profile = FC.ConfigurarRealSense(640,480,FPS)#introducir la resolucion a capturar 1280,720  640,480

# Configurar la limitacion de vision
metros = 1 #numero de metros maximo de viion
clipping_distance = FC.LimitarDistancia(profile,metros)

# Eliminar margen de error en imagen a color
align_to,align = FC.Alinear()

# ******************* MAIN ******************************
while True:
    # Obtencion de imagenes
    depth,coloreada = FC.Obtener_Imagenes(pipeline,align)

    # Obtenemos la matriz de imagen y profundidad en valores numpy
    profundidad, frame = FC.Obtener_Datos(depth,coloreada)# Convierte la informacion obtenida por la realsense en formato numpy

    # Eliminar fondo de la imagen a color
    frame = FC.EliminarFondo(frame,profundidad,clipping_distance,153)# Eliminamos fondo

    # # Eliminación de ruido
    # frame = cv2.GaussianBlur(frame,(45,45),0)

    # Iniciar a contar tiempo para contar FPS
    start_time = time.time()

    # Convertir a float32 para TF
    batch_data = tf.constant(FC.PreprocessToTF(frame))

    # Crear preicciones
    pred_bbox = infer(batch_data)

    # obtener configuracion de cuadros
    boxes,pred_conf=FC.ConfigurarCuadro(pred_bbox)

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
    )
    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]

    # Eliminación de ruido
    frame = cv2.GaussianBlur(frame,(45,45),0)

    # Dibujar cuadro de reconocimiento
    image = FC.draw_bbox(frame,pred_bbox,file_names,Colores,profundidad)

    # Calcular FPS
    fps = 1.0 / (time.time() - start_time)
    # MOstrar FPS
    # print("FPS: %.2f" % fps)
    # Acondicionar para mostrar imagen
    result = np.asarray(image)
    #Mostrar resultado
    cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("result", result)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
cv2.destroyAllWindows()
