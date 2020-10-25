import colorsys
import random
import cv2
import numpy as np
import pyrealsense2 as rs

def ConvertirToNumpyArray(Lista):
    Lista = np.array(Lista, np.uint8) #Convertimos la lista en una lista numpu
    return Lista

#************* FUNCIONES REALSENSE********************
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

def PreprocessToTF(frame):
    image_data = cv2.resize(frame, (416, 416))
    image_data = image_data /255.
    image_data = image_data[np.newaxis, ...].astype(np.float32)
    return image_data

def ConfigurarCuadro(pred_bbox):
    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]
    return boxes,pred_conf



# Funcion que lee el archivo .names
def read_class_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

def DibujarContornos(imagen,contornos,color,Palabra,Imagen_Profundidad):
    for c in contornos:
        M = cv2.moments(c) # Se encuentra el centroide de todos los momentos encontrados
        if (M["m00"]==0):M["m00"]=1 # En caso de que el denominador sea 0 se cambia a 1
        x = int(M["m10"] / M["m00"]) # Se saca el centro en y
        y = int(M["m01"] / M["m00"]) # Se saca el cntro en X
        Dist = Imagen_Profundidad[y,x] /10
        cv2.drawContours(imagen,[c],0,color,2) # Dibuja el contorno
        epsilon = 0.1*cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c,epsilon,True)
        cv2.drawContours(imagen,approx,0,color,2)
        cv2.putText(imagen,str(Palabra)+"D: "+str(Dist),(x,y),1,2,(0,0,0),2) # Pone texto

def Identificar_Compa_Ene(Imagen,IzquierdaSuperior,DerechaInferior,Colores,Imagen_Profundidad):
    Imagen_Auxiliar = Imagen[int(IzquierdaSuperior[1]):int(DerechaInferior[1]),int(IzquierdaSuperior[0]):int(DerechaInferior[0])]
    # Conversion de colores
    hsv_image = cv2.cvtColor(Imagen_Auxiliar,cv2.COLOR_BGR2HSV)#Convertimos BGR  a HSV para deteccion de colores
    # Buscar areas de la imagen con el color definido
    mascaraR = cv2.inRange(hsv_image,Colores[0],Colores[1])#verifica cuales pixeles  estan dentro del rango, los que no esten los hace negros
    mascaraB = cv2.inRange(hsv_image,Colores[2],Colores[3])#verifica cuales pixeles  estan dentro del rango, los que no esten los hace negros
    # Encontrar pixeles de contornos de objetos
    ContornoRojo = cv2.findContours(mascaraR,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]
    ContornoAzul = cv2.findContours(mascaraB,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]
    # Dibujar contornos
    DibujarContornos(Imagen_Auxiliar,ContornoRojo,(255,255,255),"Equipo",Imagen_Profundidad)
    DibujarContornos(Imagen_Auxiliar,ContornoAzul,(255,255,255),"Enemigo",Imagen_Profundidad)
    # Sustituimos el area de la imagen original con la reconocida
    Imagen[int(IzquierdaSuperior[1]):int(DerechaInferior[1]),int(IzquierdaSuperior[0]):int(DerechaInferior[0])] = Imagen_Auxiliar
    return Imagen


# Funcion que regresa la imagen con los cuadros dibujados
def draw_bbox(image, bboxes, class_dir, colores,Imagen_Profundidad,show_label=True):
    classes=read_class_names(class_dir)
    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)
    out_boxes, out_scores, out_classes, num_boxes = bboxes
    for i in range(num_boxes[0]):
        if int(out_classes[0][i]) < 0 or int(out_classes[0][i]) > num_classes: continue
        coor = out_boxes[0][i]
        coor[0] = int(coor[0] * image_h)
        coor[2] = int(coor[2] * image_h)
        coor[1] = int(coor[1] * image_w)
        coor[3] = int(coor[3] * image_w)

        fontScale = 0.5
        score = out_scores[0][i]
        class_ind = int(out_classes[0][i])
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (coor[1], coor[0]), (coor[3], coor[2])
        image =Identificar_Compa_Ene(image,c1,c2,colores,Imagen_Profundidad)
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            bbox_mess = '%s: %.2f' % (classes[class_ind], score)
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
            c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
            cv2.rectangle(image, c1, (np.float32(c3[0]), np.float32(c3[1])), bbox_color, -1) #filled

            cv2.putText(image, bbox_mess, (c1[0], np.float32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
    return image
