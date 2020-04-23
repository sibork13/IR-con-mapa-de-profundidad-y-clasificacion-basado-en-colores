# IR-con-mapa-de-profundidad-y-clasificacion-basado-en-colores

El sistema captura la escena y, con uso de la real sense
se obtiene la imagen y un mapa de profundidad.
Se define una distancia maxima de vision para que, con 
el mapa de profundidad y la imagen a color, se pueda eliminar
todo aquello que en la imagen este a una distancia mayor que la definida.
Despues se utiliza reconocimiento de imagenes para identificar a los
robots humanoides dentro de la escena.
Una vez identificados los robots, se aplica identificacion de colores para
definir la categoria del robot segun un distintivo que estos porten.

Requerimientos para ejecutar el programa.
1. Cython
2. numpy
3. darkflow
4. OpenCV
5. TENSORFLOW
6. pyrealsense2
