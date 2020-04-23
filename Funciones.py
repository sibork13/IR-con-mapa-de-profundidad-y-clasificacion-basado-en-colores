#########################
#   By Oscar Herrera    #
#   april 28th, 2020    #
#########################

import pyrealsense2 as rs
import numpy as np
import cv2

def ConfigurarRealSense(x,y,FPS):
    pipeline = rs.pipeline() # Abrimos tuberias
    config = rs.config() # Inicializar parametros de configuracion
    config.enable_stream(rs.stream.depth, x, y, rs.format.z16, FPS)# Prametro de transmision para mapa de profundidad
    config.enable_stream(rs.stream.color, x, y, rs.format.bgr8, FPS)# Parametros de transmision para imagen a color
    profile = pipeline.start(config) # Inicializamos la transimsion
    return pipeline,profile

def LimitarDistancia(profile,DistanciaMaxima):
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    # print("Depth Scale is: " , depth_scale)
    clipping_distance = DistanciaMaxima / depth_scale #Distancia recortada
    return clipping_distance


def Alinear():
    align_to = rs.stream.color # Elegimos a que estream se debe alinear
    align = rs.align(align_to) # Utiliza la referencia para alinear los frame de profundidad
    return align_to,align


def Obtener_Imagenes(pipeline,align):
    frames = pipeline.wait_for_frames()#Obtiene frame
    aligned_frames = align.process(frames) # Alinea todos los frames recibidos a la referencia obtenida en align
    Imagen_Profundidad_Alineada = aligned_frames.get_depth_frame() # Obtenemos mapa de profundidad
    Imagen_Color = aligned_frames.get_color_frame() # Obtenemos imagen a color
    return Imagen_Profundidad_Alineada,Imagen_Color

def Obtener_Datos(Imagen_Profundidad,Imagen_color):
    Datos_Profundidad = np.asanyarray(Imagen_Profundidad.get_data()) #Convertimos el mapa de profundidad a valores numpy
    Datos_Color = np.asanyarray(Imagen_color.get_data()) # convertimos valores de imagen de color a valores numpy
    return Datos_Profundidad,Datos_Color

def EliminarFondo(Imagen_Color,Imagen_Profundidad,Distancia,Color_Fondo):
    depth_image_3d = np.dstack((Imagen_Profundidad,Imagen_Profundidad,Imagen_Profundidad)) # Creamos la matriz de profundidad en una de 3 dimensiones
    bg_removed = np.where((depth_image_3d > Distancia) | (depth_image_3d <= 0), Color_Fondo, Imagen_Color) # A todos los valores de todas las matrices mayores a la distancia maxima se cambia al valor de color insertado
    return bg_removed

def ConvertirToNumpyArray(Lista):
    Lista = np.array(Lista, np.uint8) #Convertimos la lista en una lista numpu
    return Lista


def DibujarContornos(imagen,contornos,color,Palabra):
    for c in contornos:
        M = cv2.moments(c) # Se encuentra el centroide de todos los momentos encontrados
        if (M["m00"]==0):M["m00"]=1 # En caso de que el denominador sea 0 se cambia a 1
        x = int(M["m10"] / M["m00"]) # Se saca el centro en y
        y = int(M["m01"] / M["m00"]) # Se saca el cntro en X
        cv2.drawContours(imagen,[c],0,color,2) # Dibuja el contorno
        epsilon = 0.1*cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c,epsilon,True)
        cv2.drawContours(imagen,approx,0,color,2)
        cv2.putText(imagen,str(Palabra),(x,y),1,2,(0,0,0),2) # Pone texto
