# Importaciones
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import numpy as np
import csv
import time

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# Importaciones de detección de objetos
# lamamos los scrips de tensorflow 
from utils import label_map_util # cargar las etiquetas
from utils import visualization_utils as vis_util # procesar las imagenes
#from utils.image_utils import image_saver

# inicializamos .csv
with open('traffic_measurement.csv', 'w') as f:
    writer = csv.writer(f)
    csv_line = \
        'Vehicle Type/Size, Vehicle Color, Vehicle Movement Direction, Vehicle Speed (km/h)'
    writer.writerows([csv_line.split(',')])
# el modelo de ssd requiere tener tensorflow la version 1.4 o mayor
if tf.__version__ < '1.4.0':
    raise ImportError('Actualice su instalación de tensorflow a v1.4. * O posterior!'
                      )

# entrada del video
cap = cv2.VideoCapture('sub-1504614469486.mp4')
#cap = cv2.VideoCapture('Video02.mp4')
# Variables
limite_velocidad = '40'
total_passed_vehicle = 0  # contar de vehículos
# Estamos utilizando el modelo de "SSD con Mobilenet".
# modelo descargado
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
#DOWNLOAD_BASE = \
#    'http://download.tensorflow.org/models/object_detection/'


# Ruta al gráfico de detección de congelados. Este es el modelo real que se utiliza para la detección de objetos.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# Lista de las cadenas que se utilizan para agregar la etiqueta correcta para cada caja.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90#cantidad de clases que estan dentro de este label

# Descargar el modelo 
# descomentar si aún no ha descargado el modelo
# Cargue un modelo de Tensorflow (congelado) en la memoria.
#almacenar en la memoria el modelo de inception
detection_graph = tf.Graph()# crear el objeto de tensorgraph
with detection_graph.as_default():# levantar el modelo frozen
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


# Cargando mapa de etiquetas
# La etiqueta asigna los índices a los nombres de las categorías, 
#de modo que cuando nuestra red de convolución predice 5, 
#sabemos que esto corresponde al avión. Aquí utilizo funciones de utilidad internas, 
#pero cualquier cosa que devuelva los enteros de asignación de un diccionario a las etiquetas de cadena apropiadas estaría bien
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map,
        max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Codigo de ayuda
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width,
            3)).astype(np.uint8)


# Deteccion
def object_detection_function():
    total_passed_vehicle = 0
    speed = 'waiting...'
    direction = 'waiting...'
    size = 'waiting...'
    color = 'waiting...'
    #deteccion de objetos 
    #correr la sesion de tensorflow as sess
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:

            # Tensores de entrada y salida definidos para detección_gráfico
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            
            # Cada cuadro representa una parte de la imagen donde se detectó un objeto en particular.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Cada puntuación representa cómo el nivel de confianza para cada uno de los objetos.
            # La puntuación se muestra en la imagen del resultado, junto con la etiqueta de la clase.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
           
            # para todos los cuadros que se extraen del video de entrada
            while True:
                (ret, frame) = cap.read()

                if not ret:
                    print ('final del archivo de video...')
                    break

                input_frame = frame


          
                # Expandir dimensiones ya que el modelo espera que las imágenes tengan forma: [1, Ninguna, Ninguna, 3]
                image_np_expanded = np.expand_dims(input_frame, axis=0)

                # Actual deteccion.
                #introducimos los datos a la sesion "sess"
                (boxes, scores, classes, num) = \
                    sess.run([detection_boxes, detection_scores,
                             detection_classes, num_detections],
                             feed_dict={image_tensor: image_np_expanded})

                #Visualización de los resultados de una detección
                #transformamos los datos a una funcion array ya que los datos estan en 
                #formato tensor
                (counter, csv_line) = \
                    vis_util.visualize_boxes_and_labels_on_image_array(
                    cap.get(1),
                    input_frame,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=4,
                    )

                #Conteo de los vehiculos 
                total_passed_vehicle = total_passed_vehicle + counter

                # insertar texto de información en el cuadro de video
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(
                    input_frame,
                    'Vehiculos Detectados: ' + str(total_passed_vehicle),
                    (10, 35),
                    font,
                    0.8,
                    (0, 0xFF, 0xFF),
                    2,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    )
            
              
                #cuando el vehículo haya pasado la línea y se haya contado, haga que el color de la línea de control sea verde
                if counter == 1:
                    cv2.line(input_frame, (0, 200), (640, 200), (0, 0xFF, 0), 5)
                    # Si la velocidad calculada excede el límite de velocidad, guarde una imagen de la velocidad del automóvil
                    if speed > limite_velocidad:
                        print ('Sobre paso la velocidad!')                      
                        cv2.imwrite("vehiculo"+str(total_passed_vehicle) + '.jpg',frame)                         
                        print ('captura guardada!')
                     
                else:
                    cv2.line(input_frame, (0, 200), (640, 200), (0, 0, 0xFF), 5)           
               
                # insertar texto de información en el cuadro de video
                cv2.rectangle(input_frame, (10, 275), (230, 337), (180, 132, 109), -1)
                cv2.putText(
                    input_frame,
                    'Control',
                    (15, 190),
                    font,
                    0.6,
                    (0, 0, 0xFF),
                    2,
                    cv2.LINE_AA,
                    )
                cv2.putText(
                    input_frame,
                    'INFORMACION DEL VEHICULO',
                    (11, 290),
                    font,
                    0.5,
                    (0xFF, 0xFF, 0xFF),
                    1,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    )
                cv2.putText(
                    input_frame,
                    '-Direccion de movimiento: ' + direction,
                    (14, 302),
                    font,
                    0.4,
                    (0xFF, 0xFF, 0xFF),
                    1,
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    )

                cv2.putText(
                    input_frame,
                    '-Velocidad(km/h): ' + speed,
                    (14, 312),
                    font,
                    0.4,
                    (0xFF, 0xFF, 0xFF),
                    1,
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    )
                cv2.putText(
                    input_frame,
                    '-Color: ' + color,
                    (14, 322),
                    font,
                    0.4,
                    (0xFF, 0xFF, 0xFF),
                    1,
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    )
                cv2.putText(
                    input_frame,
                    '-Vehiculo tamanio/tipo: ' + size,
                    (14, 332),
                    font,
                    0.4,
                    (0xFF, 0xFF, 0xFF),
                    1,
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    )

                cv2.imshow('vehiculos detectados', input_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                if csv_line != 'not_available':
                    with open('traffic_measurement.csv', 'a') as f:
                        writer = csv.writer(f)
                        (size, color, direction, speed) = \
                            csv_line.split(',')
                        writer.writerows([csv_line.split(',')])

            cap.release()
            cv2.destroyAllWindows()

object_detection_function()		
