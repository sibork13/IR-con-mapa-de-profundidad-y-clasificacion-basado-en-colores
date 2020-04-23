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

def Obtener_Info_IR(Resultado):
    # Obtenemos coordenadas de cada objeto reconocido (solo la esquina superior izquierda y la esquina inferior derecha)
    IS = (Resultado['topleft']['x'], Resultado['topleft']['y'])# Coordenadas de la esquina superior izquierda
    DI = (Resultado['bottomright']['x'], Resultado['bottomright']['y'])# Coordenadas de la esquina inferior derecha
    etiqueta = Resultado['label'] # Clase a la que pertenece el reconocimiento
    certeza = Resultado['confidence'] # Porcentaje de seguridad de pertenencia a la clase
    return IS,DI,etiqueta,certeza


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


def Identificar_Compa_Ene(Imagen,IzquierdaSuperior,DerechaInferior,Colores):
    Imagen_Auxiliar = Imagen[IzquierdaSuperior[1]:DerechaInferior[1],IzquierdaSuperior[0]:DerechaInferior[0]]
    # Conversion de colores
    hsv_image = cv2.cvtColor(Imagen_Auxiliar,cv2.COLOR_BGR2HSV)#Convertimos BGR  a HSV para deteccion de colores
    # Buscar areas de la imagen con el color definido
    mascaraR = cv2.inRange(hsv_image,Colores[0],Colores[1])#verifica cuales pixeles  estan dentro del rango, los que no esten los hace negros
    mascaraB = cv2.inRange(hsv_image,Colores[2],Colores[3])#verifica cuales pixeles  estan dentro del rango, los que no esten los hace negros
    # Encontrar pixeles de contornos de objetos
    ContornoRojo = cv2.findContours(mascaraR,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]
    ContornoAzul = cv2.findContours(mascaraB,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]
    # Dibujar contornos
    DibujarContornos(Imagen_Auxiliar,ContornoRojo,(255,255,255),"Equipo")
    DibujarContornos(Imagen_Auxiliar,ContornoAzul,(255,255,255),"Enemigo")
    # Sustituimos el area de la imagen original con la reconocida
    Imagen[IzquierdaSuperior[1]:DerechaInferior[1],IzquierdaSuperior[0]:DerechaInferior[0]] = Imagen_Auxiliar
    return Imagen


def Dibujar_Info(Imagen,IzquierdaSuperior,DerechaInferior,color,etiqueta,certeza):
    text = '{}: {:.0f}%'.format(etiqueta, certeza * 100) # Texto a insertar
    Imagen = cv2.rectangle(Imagen, IzquierdaSuperior, DerechaInferior, color, 5) # Dibujar rectangulo
    Imagen = cv2.putText(Imagen, text, IzquierdaSuperior, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2) # Poner texto
    return Imagen
