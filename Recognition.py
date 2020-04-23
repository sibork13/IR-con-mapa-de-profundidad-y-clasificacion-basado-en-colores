#########################
#   By Oscar Herrera    #
#   april 28th, 2020    #
#########################

# ************  Importar librerias necesarias   ************************
import cv2
from darkflow.net.build import TFNet
import numpy as np
import time
import Funciones as FC

#*************** Definicion de parametros ***************************
options = {
    "threshold": 0.01, # Porcentaje umbral
    "pbLoad":"tiny-yolo-Nuevo.pb",
    "metaLoad":"tiny-yolo-Nuevo.meta"
}

tfnet = TFNet(options)
colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]

# Limites de color en HSV (H-> elige color, S-> que tanto color de blanco a el color vivo, V-> que tanto color de negro al color)
# Los limites fueron elegidos para el filtro Gaussian Blur
Rojo = [(160,100,45),(180,255,255)]
Azul = [[90,100,45],[130,255,255]]
Colores = Rojo+ Azul
Colores = FC.ConvertirToNumpyArray(Colores)


#****************** Preparando RealSense **********************************

# Preparar transmision de realsense a sistemas
FPS = 30 #Numero de fotogramas deseado  (maximo 30)
pipeline,profile = FC.ConfigurarRealSense(1280,720,FPS)#introducir la resolucion a capturar 1280,720  640,480

# Configurar la limitacion de vision
metros = 1 #numero de metros maximo de viion
clipping_distance = FC.LimitarDistancia(profile,metros)

# Eliminar margen de error en imagen a color
align_to,align = FC.Alinear()
# ******************** BLoque de procesamiento ****************************************
while True:
    # Iniciamos tiempo para contar FPS finales
    stime = time.time()

    # Obtencion de imagenes
    depth,coloreada = FC.Obtener_Imagenes(pipeline,align)

    # Obtenemos la matriz de imagen y profundidad en valores numpy
    profundidad, frame = FC.Obtener_Datos(depth,coloreada)# Convierte la informacion obtenida por la realsense en formato numpy

    # Eliminar fondo de la imagen a color
    frame = FC.EliminarFondo(frame,profundidad,clipping_distance,153)# Eliminamos fondo

    # Eliminación de EliminarFondo
    frame = cv2.GaussianBlur(frame,(45,45),0)

    # Buscar imagen con RCNN
    results = tfnet.return_predict(frame) #encuentra los resultados de la R-CNN

    for color, result in zip(colors, results):

        # Obtenemos coordenadas de cada objeto reconocido (solo la esquina superior izquierda y la esquina inferior derecha)
        tl = (result['topleft']['x'], result['topleft']['y'])# Coordenadas de la esquina superior izquierda
        br = (result['bottomright']['x'], result['bottomright']['y'])# Coordenadas de la esquina inferior derecha

        #Recortamos la imegen entera con solo la area del objeto reconocido
        Imagen_Auxiliar = frame[tl[1]:br[1],tl[0]:br[0]]

        # ************************ Reconocimiento de enemigo o amigo *********************

        # Conversion de colores
        hsv_image = cv2.cvtColor(Imagen_Auxiliar,cv2.COLOR_BGR2HSV)#Convertimos BGR  a HSV para deteccion de colores

        # Buscar areas de la imagen con el color definido
        mascaraR = cv2.inRange(hsv_image,Colores[0],Colores[1])#verifica cuales pixeles  estan dentro del rango, los que no esten los hace negros
        mascaraB = cv2.inRange(hsv_image,Colores[2],Colores[3])#verifica cuales pixeles  estan dentro del rango, los que no esten los hace negros
        # Encontrar pixeles de contornos de objetos
        ContornoRojo = cv2.findContours(mascaraR,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]
        ContornoAzul = cv2.findContours(mascaraB,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]
        # Dibujar contornos
        FC.DibujarContornos(Imagen_Auxiliar,ContornoRojo,(255,255,255),"Equipo")
        FC.DibujarContornos(Imagen_Auxiliar,ContornoAzul,(255,255,255),"Enemigo")


        label = result['label'] # Clase a la que pertenece el reconocimiento
        confidence = result['confidence'] # Porcentaje de seguridad de pertenencia a la clase
        text = '{}: {:.0f}%'.format(label, confidence * 100) # Texto a insertar
        frame = cv2.rectangle(frame, tl, br, color, 5) # Dibujar rectangulo

        # Sustituimos el area de la imagen original con la reconocida
        frame[tl[1]:br[1],tl[0]:br[0]] = Imagen_Auxiliar

        frame = cv2.putText(frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2) # Poner texto
    cv2.imshow('frame', frame)# MMostrar imagen resultante
    print('FPS {:.1f}'.format(1 / (time.time() - stime))) #Imprimir FPS
    if cv2.waitKey(1) & 0xFF == ord('q'): # Condicion para parar la transmision
        break

# capture.release()
pipeline.stop()
cv2.destroyAllWindows()
