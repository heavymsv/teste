from flask import Flask, Response, jsonify, request, send_file
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import csv
import cv2
import numpy as np
import imutils
from datetime import datetime
from collections import defaultdict
from datetime import datetime
from threading import Thread
from ultralytics import YOLO
import math
import ast
import os
import shutil
import subprocess
from datetime import datetime

import serial
import serial.tools.list_ports



app = Flask(__name__)
CORS(app)
app.config["JWT_SECRET_KEY"] = "NewtecPI"  # Change this!
jwt = JWTManager(app)


socketio = SocketIO(app)
socketio.init_app(app, cors_allowed_origins="*")
ativar = False
model = YOLO("models/yolov8s.pt")
points_shape = [[(0,0),(0,0),(0,0),(0,0)]]
detection_limit1 = [(0,0),(1,1)]

lower_red = np.array([0, 0, 150], dtype = "uint8")
upper_red= np.array([50, 50, 255], dtype = "uint8")

lower_green = np.array([110, 110, 0], dtype = "uint8")
upper_green= np.array([255, 255, 90], dtype = "uint8")

LABELS = {0: u'__background__', 1: u'person', 2: u'bicycle',3: u'car', 4: u'motorcycle', 5: u'airplane', 6: u'bus', 7: u'train', 8: u'truck', 9: u'boat', 10: u'traffic light', 11: u'fire hydrant', 12: u'stop sign', 13: u'parking meter', 14: u'bench', 15: u'bird', 16: u'cat', 17: u'dog', 18: u'horse', 19: u'sheep', 20: u'cow', 21: u'elephant', 22: u'bear', 23: u'zebra', 24: u'giraffe', 25: u'backpack', 26: u'umbrella', 27: u'handbag', 28: u'tie', 29: u'suitcase', 30: u'frisbee', 31: u'skis', 32: u'snowboard', 33: u'sports ball', 34: u'kite', 35: u'baseball bat', 36: u'baseball glove', 37: u'skateboard', 38: u'surfboard', 39: u'tennis racket', 40: u'bottle', 41: u'wine glass', 42: u'cup', 43: u'fork', 44: u'knife', 45: u'spoon', 46: u'bowl', 47: u'banana', 48: u'apple', 49: u'sandwich', 50: u'orange', 51: u'broccoli', 52: u'carrot', 53: u'hot dog', 54: u'pizza', 55: u'donut', 56: u'cake', 57: u'chair', 58: u'couch', 59: u'potted plant', 60: u'bed', 61: u'dining table', 62: u'toilet', 63: u'tv', 64: u'laptop', 65: u'mouse', 66: u'remote', 67: u'keyboard', 68: u'cell phone', 69: u'microwave', 70: u'oven', 71: u'toaster', 72: u'sink', 73: u'refrigerator', 74: u'book', 75: u'clock', 76: u'vase', 77: u'scissors', 78: u'teddy bear', 79: u'hair drier', 80: u'toothbrush'}
COLORS = [(89, 161, 197),(67, 161, 255),(19, 222, 24),(186, 55, 2),(167, 146, 11),(190, 76, 98),(130, 172, 179),(115, 209, 128),(204, 79, 135),(136, 126, 185),(209, 213, 45),(44, 52, 10),(101, 158, 121),(179, 124, 12),(25, 33, 189),(45, 115, 11),(73, 197, 184),(62, 225, 221),(32, 46, 52),(20, 165, 16),(54, 15, 57),(12, 150, 9),(10, 46, 99),(94, 89, 46),(48, 37, 106),(42, 10, 96),(7, 164, 128),(98, 213, 120),(40, 5, 219),(54, 25, 150),(251, 74, 172),(0, 236, 196),(21, 104, 190),(226, 74, 232),(120, 67, 25),(191, 106, 197),(8, 15, 134),(21, 2, 1),(142, 63, 109),(133, 148, 146),(187, 77, 253),(155, 22, 122),(218, 130, 77),(164, 102, 79),(43, 152, 125),(185, 124, 151),(95, 159, 238),(128, 89, 85),(228, 6, 60),(6, 41, 210),(11, 1, 133),(30, 96, 58),(230, 136, 109),(126, 45, 174),(164, 63, 165),(32, 111, 29),(232, 40, 70),(55, 31, 198),(148, 211, 129),(10, 186, 211),(181, 201, 94),(55, 35, 92),(129, 140, 233),(70, 250, 116),(61, 209, 152),(216, 21, 138),(100, 0, 176),(3, 42, 70),(151, 13, 44),(216, 102, 88),(125, 216, 93),(171, 236, 47),(253, 127, 103),(205, 137, 244),(193, 137, 224),(36, 152, 214),(17, 50, 238),(154, 165, 67),(114, 129, 60),(119, 24, 48),(73, 8, 110)]
CONF_TRESHOLD = 0.50

track_history = defaultdict(lambda: [])

# Define the number of rectangles and their properties
num_rectangles = 4
rectangle_color = (192, 192, 192)  # Light gray color in BGR format
border_color = (0, 0, 0)  # Black color for borders in BGR format
border_thickness = 1


font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.6
font_thickness = 1
font_color = (0, 0, 0) 

# Calculate the height at which the rectangles will be placed (top of the image)
top_y = 0

button_on = True
match = -1

totalFrame = []
novoFrame = False
startAnalisys = True


botoes_names = [ f'Detection', f'Lane', f'Crosswalk', f'Traffic lights' ]
botoes_areas_top = [ (0,0), (0,0), (0,0), (0,0) ]
botoes_areas_bottom = [ (0,0), (0,0), (0,0), (0,0) ]

button_press = False

zoom = 60

b_shape_detection = False
shape_detection = []

b_lanes_points = False
lanes_points = []
b_select_lane = False
direction = True
b_direction = False
index_lanes_points = -1

b_crosswalk_1 = False
b_crosswalk_2 = False
crosswalk_1 = []
crosswalk_2 = []
b_select_crosswalk_1 = False
b_select_crosswalk_2 = False
index_crosswalk = -1

circle_1 = []
circle_2 = []
b_select_circle_1 = False
b_select_circle_2 = False

carros_fx1 = []
carros_fx2 = []
infra_fx1 = []
infra_fx2 = []

bool_fx1 = False
bool_fx2 = False
bool_if1 = False
bool_if2 = False

delta_x=0
delta_y=0

LOCAL_FILES_DIR = 'imgs'
DOWNLOADS_DIR = 'downloads'
PROCESSED_DIR = 'history'

delta = 0.8
past_red = 0
is_red = 0
initialyze = True
quant = 0

init_hora = ""
fin_hora = ""

local = ""
numero = ""
portaria = ""
dataPort = ""
dataAfer = ""
orgao = ""

tracker = cv2.TrackerCSRT_create()


def mass_of_rect(p1,p2):
    #print(abs((p1[0]-p2[0])*(p1[1]-p2[1])))
    #print(cv2)
    pass

def cross_line(track):
    global button_on, match, b_shape_detection, shape_detection, b_lanes_points, lanes_points, b_select_lane
    global button_press, direction, b_direction, index_lanes_points, b_crosswalk_1, b_crosswalk_2, crosswalk_1
    global crosswalk_2, b_select_crosswalk_1, b_select_crosswalk_2,index_crosswalk, circle_1, circle_2
    global b_select_circle_1, b_select_circle_2,zoom
    if(len(track)<=1):
        print("track muito curta")
        return False

    detection_limit1 = crosswalk_1
    

    if (not direction):
        print("not direction")
        a = (detection_limit1[1][1] -detection_limit1[0][1] )/(detection_limit1[1][0] -detection_limit1[0][0] )
        b = ((detection_limit1[0][1]*detection_limit1[1][0] - detection_limit1[1][1]*detection_limit1[0][0])/(zoom/100) )/(detection_limit1[1][0]-detection_limit1[0][0])
        res1 = a*track[-2][0]+b
        res2 = a*track[-1][0]+b
        if((res1<track[-2][1] and res2>track[-1][1]) or (res1>track[-2][1] and res2<track[-1][1])):
            print('Cruzou')
            return True
        return False
    else:
        print("direction")
        a = (detection_limit1[1][0]-detection_limit1[0][0])/(detection_limit1[1][1]-detection_limit1[0][1])
        b = (detection_limit1[0][0]*detection_limit1[1][1] - detection_limit1[1][0]*detection_limit1[0][1])/(detection_limit1[1][1]-detection_limit1[0][1])
        res1 = a*track[-2][1]+b
        res2 = a*track[-1][1]+b
        if((res1<track[-2][0] and res2>track[-1][0]) or (res1>track[-2][0] and res2<track[-1][0])):
            print('Cruzou')
            return True
        return False


def box_label(image, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
    lw = max(round(sum(image.shape) / 2 * 0.003), 2)
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    obj = cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    mass_of_rect(p1,p2)
    if label:
        tf = max(lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image,
                    label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0,
                    lw / 3,
                    txt_color,
                    thickness=tf,
                    lineType=cv2.LINE_AA)

def adjust_line(lanes_points):
    return lanes_points


def between(x,a,b):
    return (x<a and x>b) or (x>a and x<b)

def d_radius(p1,p2):
    dist = (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
    return dist**(1/2)

def next_to(point, a, b):
    return [b if (d_radius(point,a)>d_radius(point,b)) else a][0]

#verifica se estar dentro da lane 1 ou 2

def det_faixa(point, points):

    print("entrada: " + str(point))
    print("espaço: " + str(points))
    
    x, y = point
    n = len(points)
    if n != 4:
        raise ValueError("This algorithm works for four points only.")

    A, B, C, D = points

    AB = (B[0] - A[0], B[1] - A[1])
    BC = (C[0] - B[0], C[1] - B[1])
    CD = (D[0] - C[0], D[1] - C[1])
    DA = (A[0] - D[0], A[1] - D[1])

    AP = (x - A[0], y - A[1])
    BP = (x - B[0], y - B[1])
    CP = (x - C[0], y - C[1])
    DP = (x - D[0], y - D[1])

    CrossProductAB_AP = AB[0] * AP[1] - AB[1] * AP[0]
    CrossProductBC_BP = BC[0] * BP[1] - BC[1] * BP[0]
    CrossProductCD_CP = CD[0] * CP[1] - CD[1] * CP[0]
    CrossProductDA_DP = DA[0] * DP[1] - DA[1] * DP[0]

    if (CrossProductAB_AP > 0 and CrossProductBC_BP > 0 and CrossProductCD_CP > 0 and CrossProductDA_DP > 0) or \
       (CrossProductAB_AP < 0 and CrossProductBC_BP < 0 and CrossProductCD_CP < 0 and CrossProductDA_DP < 0):
        return 1
    else:
        return 2



#ajuste de faixa de pedestre

def adjust_crosswalk(index, point):
    global direction, shape_detection

    if(not direction):
        if index == 0:
            if(between(point[0], shape_detection[0][0], shape_detection[3][0])):
                x1 = point[0]
                a = (shape_detection[0][1] - shape_detection[3][1])/(shape_detection[0][0] - shape_detection[3][0])
                b = (shape_detection[0][1]*shape_detection[3][0] - shape_detection[3][1]*shape_detection[0][0])/(shape_detection[0][0] - shape_detection[3][0])
                y1 = abs(a*x1 - b)

                return (x1, y1)
                    
            elif (between(point[1], shape_detection[0][1], shape_detection[3][1])):
                y1 = point[1]

                a = (shape_detection[0][0] - shape_detection[3][0])/(shape_detection[0][1] - shape_detection[3][1])
                b = (shape_detection[0][0]*shape_detection[3][1] - shape_detection[3][0]*shape_detection[0][1])/(shape_detection[0][1] - shape_detection[3][1])
                x1 = abs(a*y1 - b)

                return (x1, y1)
            else:
                return next_to(point, shape_detection[0], shape_detection[3])
            
        else:
            if(between(point[0], shape_detection[1][0], shape_detection[2][0])):
                x1 = point[0]
                a = (shape_detection[1][1] - shape_detection[2][1])/(shape_detection[1][0] - shape_detection[2][0])
                b = (shape_detection[1][1]*shape_detection[2][0] - shape_detection[2][1]*shape_detection[1][0])/(shape_detection[1][0] - shape_detection[2][0])
                y1 = abs(a*x1 - b)

                return (x1, y1)
                    
            elif (between(point[1], shape_detection[1][1], shape_detection[2][1])):
                y1 = point[1]

                a = (shape_detection[1][0] - shape_detection[2][0])/(shape_detection[1][1] - shape_detection[2][1])
                b = (shape_detection[1][0]*shape_detection[2][1] - shape_detection[2][0]*shape_detection[1][1])/(shape_detection[1][1] - shape_detection[2][1])
                x1 = abs(a*y1 - b)

                return (x1, y1)
            else:
                return next_to(point, shape_detection[1], shape_detection[2])
    else:
        if index == 0:
            if(between(point[0], shape_detection[1][0], shape_detection[0][0])):
                x1 = point[0]
                a = (shape_detection[1][1] - shape_detection[0][1])/(shape_detection[1][0] - shape_detection[0][0])
                b = (shape_detection[1][1]*shape_detection[0][0] - shape_detection[0][1]*shape_detection[1][0])/(shape_detection[1][0] - shape_detection[0][0])
                y1 = abs(a*x1 - b)

                return (x1, y1)
                    
            elif (between(point[1], shape_detection[1][1], shape_detection[0][1])):
                y1 = point[1]

                a = (shape_detection[1][0] - shape_detection[0][0])/(shape_detection[1][1] - shape_detection[0][1])
                b = (shape_detection[1][0]*shape_detection[0][1] - shape_detection[0][0]*shape_detection[1][1])/(shape_detection[1][1] - shape_detection[0][1])
                x1 = abs(a*y1 - b)

                return (x1, y1)
            else:
                return next_to(point, shape_detection[0], shape_detection[1])
            
        else:
            if(between(point[0], shape_detection[2][0], shape_detection[3][0])):
                x1 = point[0]
                a = (shape_detection[2][1] - shape_detection[3][1])/(shape_detection[2][0] - shape_detection[3][0])
                b = (shape_detection[2][1]*shape_detection[3][0] - shape_detection[3][1]*shape_detection[2][0])/(shape_detection[2][0] - shape_detection[3][0])
                y1 = abs(a*x1 - b)

                return (x1, y1)
                    
            elif (between(point[1], shape_detection[2][1], shape_detection[3][1])):
                y1 = point[1]

                a = (shape_detection[2][0] - shape_detection[3][0])/(shape_detection[2][1] - shape_detection[3][1])
                b = (shape_detection[2][0]*shape_detection[3][1] - shape_detection[3][0]*shape_detection[2][1])/(shape_detection[2][1] - shape_detection[3][1])
                x1 = abs(a*y1 - b)

                return (x1, y1)
            else:
                return next_to(point, shape_detection[2], shape_detection[3])

#pega areas previamente salvas
def write_area():
    global ativar, shape_detection, lanes_points, crosswalk_1, crosswalk_2, circle_1, circle_2, direction  
    try:
        # Get data from the request
        print('direction: '+str(direction))
        data = { 'shape_detection':shape_detection,
                 'lanes_points':lanes_points,
                 'crosswalk_1':crosswalk_1,
                 'crosswalk_2':crosswalk_2,
                 'circle_1':circle_1,
                 'circle_2':circle_2,
                 'direction': direction}
        
        with open('areas.csv', 'w', newline='') as file:
            # Create a CSV writer object
            print(data.keys())
            csv_writer = csv.DictWriter(file, fieldnames=data.keys())

            # Write the header if the file is empty
            csv_writer.writeheader()
            print(data.keys())
            # Write the data
            print(type(data))
            csv_writer.writerow(data)

        return jsonify(data)
    except Exception as e:
        print(e)
        return str(e)

def read_area():
    global ativar, shape_detection, lanes_points, crosswalk_1, crosswalk_2, circle_1, circle_2, direction
    try:
        with open('areas.csv', 'r') as file:
            # Assuming the first row contains headers
            data = []
            csv_reader = csv.DictReader(file)
            print('Carga')
            data = [row for row in csv_reader]
            if(data!=[]):
                
                #data = jsonify(data)
                print('Carga Leve')
                print(data)
                if(data[0]!=[]):
                    print('Carga solta')
                    keys = data[0].keys()
                    if(data[0]['shape_detection']!=[]):
                        print(type(ast.literal_eval(data[0]['shape_detection'])))
                        shape_detection = ast.literal_eval(data[0]['shape_detection'])
                    if(data[0]['lanes_points']!=[]):
                        lanes_points = ast.literal_eval(data[0]['lanes_points'])
                    if(data[0]['crosswalk_1']!=[]):
                        crosswalk_1 = ast.literal_eval(data[0]['crosswalk_1'])
                    if(data[0]['crosswalk_2']!=[]):
                        crosswalk_2 = ast.literal_eval(data[0]['crosswalk_2'])
                    if(data[0]['circle_1']!=[]):
                        circle_1 = ast.literal_eval(data[0]['circle_1'])
                    if(data[0]['circle_2']!=[]):
                        circle_2 = ast.literal_eval(data[0]['circle_2'])
                    if(data[0]['direction']!=[]):
                        direction = ast.literal_eval(data[0]['direction'])
            
            print(shape_detection)
            print('direction: '+str(direction))
            return data
    except FileNotFoundError:
        return "CSV file not found", 404

#ajuste de faixas
def adjust_points(index, point):
    global direction, shape_detection

    if(direction):
        if index == 0:
            if(between(point[0], shape_detection[0][0], shape_detection[3][0])):
                x1 = point[0]
                a = (shape_detection[0][1] - shape_detection[3][1])/(shape_detection[0][0] - shape_detection[3][0])
                b = (shape_detection[0][1]*shape_detection[3][0] - shape_detection[3][1]*shape_detection[0][0])/(shape_detection[0][0] - shape_detection[3][0])
                y1 = abs(a*x1 - b)

                return (x1, y1)
                    
            elif (between(point[1], shape_detection[0][1], shape_detection[3][1])):
                y1 = point[1]

                a = (shape_detection[0][0] - shape_detection[3][0])/(shape_detection[0][1] - shape_detection[3][1])
                b = (shape_detection[0][0]*shape_detection[3][1] - shape_detection[3][0]*shape_detection[0][1])/(shape_detection[0][1] - shape_detection[3][1])
                x1 = abs(a*y1 - b)

                return (x1, y1)
            else:
                return next_to(point, shape_detection[0], shape_detection[3])
            
        else:
            if(between(point[0], shape_detection[1][0], shape_detection[2][0])):
                x1 = point[0]
                a = (shape_detection[1][1] - shape_detection[2][1])/(shape_detection[1][0] - shape_detection[2][0])
                b = (shape_detection[1][1]*shape_detection[2][0] - shape_detection[2][1]*shape_detection[1][0])/(shape_detection[1][0] - shape_detection[2][0])
                y1 = abs(a*x1 - b)

                return (x1, y1)
                    
            elif (between(point[1], shape_detection[1][1], shape_detection[2][1])):
                y1 = point[1]

                a = (shape_detection[1][0] - shape_detection[2][0])/(shape_detection[1][1] - shape_detection[2][1])
                b = (shape_detection[1][0]*shape_detection[2][1] - shape_detection[2][0]*shape_detection[1][1])/(shape_detection[1][1] - shape_detection[2][1])
                x1 = abs(a*y1 - b)

                return (x1, y1)
            else:
                return next_to(point, shape_detection[1], shape_detection[2])
    else:
        if index == 0:
            if(between(point[0], shape_detection[1][0], shape_detection[0][0])):
                x1 = point[0]
                a = (shape_detection[1][1] - shape_detection[0][1])/(shape_detection[1][0] - shape_detection[0][0])
                b = (shape_detection[1][1]*shape_detection[0][0] - shape_detection[0][1]*shape_detection[1][0])/(shape_detection[1][0] - shape_detection[0][0])
                y1 = abs(a*x1 - b)

                return (x1, y1)
                    
            elif (between(point[1], shape_detection[1][1], shape_detection[0][1])):
                y1 = point[1]

                a = (shape_detection[1][0] - shape_detection[0][0])/(shape_detection[1][1] - shape_detection[0][1])
                b = (shape_detection[1][0]*shape_detection[0][1] - shape_detection[0][0]*shape_detection[1][1])/(shape_detection[1][1] - shape_detection[0][1])
                x1 = abs(a*y1 - b)

                return (x1, y1)
            else:
                return next_to(point, shape_detection[0], shape_detection[1])
            
        else:
            if(between(point[0], shape_detection[2][0], shape_detection[3][0])):
                x1 = point[0]
                a = (shape_detection[2][1] - shape_detection[3][1])/(shape_detection[2][0] - shape_detection[3][0])
                b = (shape_detection[2][1]*shape_detection[3][0] - shape_detection[3][1]*shape_detection[2][0])/(shape_detection[2][0] - shape_detection[3][0])
                y1 = abs(a*x1 - b)

                return (x1, y1)
                    
            elif (between(point[1], shape_detection[2][1], shape_detection[3][1])):
                y1 = point[1]

                a = (shape_detection[2][0] - shape_detection[3][0])/(shape_detection[2][1] - shape_detection[3][1])
                b = (shape_detection[2][0]*shape_detection[3][1] - shape_detection[3][0]*shape_detection[2][1])/(shape_detection[2][1] - shape_detection[3][1])
                x1 = abs(a*y1 - b)

                return (x1, y1)
            else:
                return next_to(point, shape_detection[2], shape_detection[3])



#verifica evento de clique na tela no ajuste de parametros

def click_event(event, x, y, flags, params):
    global button_on, match, b_shape_detection, shape_detection, b_lanes_points, lanes_points, b_select_lane
    global button_press, direction, b_direction, index_lanes_points, b_crosswalk_1, b_crosswalk_2, crosswalk_1
    global crosswalk_2, b_select_crosswalk_1, b_select_crosswalk_2,index_crosswalk, circle_1, circle_2
    global b_select_circle_1, b_select_circle_2, delta_x, delta_y

    distance_f_t = 50

  
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
  
        # displaying the coordinates
        # on the Shell
        if(button_on):
            for i in range(num_rectangles):
                if((x > botoes_areas_top[i][0]) and (x < botoes_areas_bottom[i][0])
                   and
                   (y > botoes_areas_top[i][1]) and (y < botoes_areas_bottom[i][1])):
                    match = i
                    print(match)
                    button_on = False
                    if(i == 0):
                        b_shape_detection = False
                        shape_detection = []
                        b_lanes_points = False
                        lanes_points = []

                        b_crosswalk_1 = False

                        crosswalk_1 = []
                        crosswalk_2 = []

                    if(i == 1):
                        if len(shape_detection) == 4:
                            b_lanes_points = True
                            b_select_lane = True
                            b_direction = True
                            lanes_points = []
                        else:
                            print('yolo')
                            button_on = True
                    if(i == 2):
                        if len(lanes_points) == 2:
        
                            b_crosswalk_1 = False
                            b_select_crosswalk_1 = True

                            crosswalk_1 = []
                            crosswalk_2 = []

                        else:
                            print('yolo')
                            button_on = True
                    if(i == 3):
                        circle_1 = [(x,y),30]
                        circle_2 = []
                        b_select_circle_1 = True
                        
        else:
            if (not b_shape_detection) and match == 0:
                shape_detection.append((x,y))
                if(len(shape_detection) == 4):
                    button_on = True
                    b_shape_detection = True
                    
            if (b_lanes_points) and match == 1 and not button_press:
                print('haleluia')
                for point in lanes_points:
                    if(d_radius(point,(x,y))<=25):
                        index_lanes_points = lanes_points.index(point)
                        #print(index_lanes_points)
                        
                        button_press = True

            if (b_select_crosswalk_1) and match == 2 and not button_press:
                print('haleluia')
                for point in crosswalk_1:
                    if(d_radius(point,(x,y))<=25):
                        index_crosswalk = crosswalk_1.index(point)
                        #print(index_lanes_points)
                        
                        button_press = True

            if (b_select_crosswalk_2) and match == 2 and not button_press:
                print('haleluia')
                for point in crosswalk_2:
                    if(d_radius(point,(x,y))<=25):
                        index_crosswalk = crosswalk_2.index(point)
                        #print(index_lanes_points)
                        
                        button_press = True
                        
         #   button_on = True
            
        print(x, ' ', y)
    
    if event == cv2.EVENT_LBUTTONUP and button_press:
        button_press = False

    if button_press and event == cv2.EVENT_MOUSEMOVE:
        #print(index_lanes_points)
        if(b_select_lane):
            points_l = adjust_points(index_lanes_points,(x,y))
            lanes_points[index_lanes_points] = (int(points_l[0]), int(points_l[1]))

        if(b_select_crosswalk_1):
            points_l = adjust_crosswalk(index_crosswalk,(x,y))
            crosswalk_1[index_crosswalk] = (int(points_l[0]), int(points_l[1]))

        if(b_select_crosswalk_2):
            points_l = adjust_crosswalk(index_crosswalk,(x,y))
            crosswalk_2[index_crosswalk] = (int(points_l[0]), int(points_l[1]))

    if event == cv2.EVENT_MOUSEMOVE and b_select_circle_1:
        circle_1[0] = (x,y)

    if event == cv2.EVENT_MOUSEMOVE and b_select_circle_2:
        circle_2[0] = (x,y)

    if event == cv2.EVENT_MOUSEWHEEL:
        print(flags)
        sum_w = 0
        if(flags>0):
            sum_w = 10
        if(flags<0):
            sum_w = -10

        if(b_select_circle_1):
            circle_1[1] = circle_1[1]+[0 if (circle_1[1]+sum_w)<0 else sum_w][0]
        if(b_select_circle_2):
            circle_2[1] = circle_2[1]+[0 if (circle_2[1]+sum_w)<0 else sum_w][0]
            

    if event == cv2.EVENT_RBUTTONDOWN:
        
        if(b_select_lane):
            direction = not direction
            b_direction = True

        if(b_select_circle_2):
            b_select_circle_2 = False
            button_on = True

        if(b_select_circle_1):
            b_select_circle_1 = False
            b_select_circle_2 = True
            circle_2 = [(x,y),30]
            

    if event == cv2.EVENT_LBUTTONDBLCLK:
        if(b_select_lane):
            print('saiu')
            b_select_lane = False
            button_on = True
            b_lanes_points = False

            mid_point = ((float(lanes_points[0][0]+lanes_points[1][0])/2),(float(lanes_points[0][1]+lanes_points[1][1])/2))

            slope = (float(lanes_points[1][1]-lanes_points[0][1])/float(lanes_points[1][0]-lanes_points[0][0]))

            if(slope!=0):
                a_slope = (-1)/slope
                delta_x = distance_f_t/math.sqrt(1+a_slope**2)
                delta_y = a_slope*delta_x
            else:
                delta_x = 0
                delta_y = distance_f_t
            

        if(b_select_crosswalk_2):
            print('2')
            b_select_crosswalk_2 = False
            button_on = True

        if(b_select_crosswalk_1):
            print('1')
            b_select_crosswalk_1 = False
            b_select_crosswalk_2 = True

def compare_hours(hour1, hour2):
    # Parse the input strings into datetime objects with a common date (e.g., "2000-01-01")
    date = "2000-01-01"
    datetime1 = datetime.strptime(f"{date} {hour1}", "%Y-%m-%d %H:%M")
    datetime2 = datetime.strptime(f"{date} {hour2}", "%Y-%m-%d %H:%M")

    # Compare the datetime objects
    return datetime1 > datetime2
    


def analyze_frame(frame, frame1):
    global startAnalisys, zoom, delta, past_red, is_red, initialyze, quant, socketio
    global button_on, match, b_shape_detection, shape_detection, b_lanes_points, lanes_points, b_select_lane
    global button_press, direction, b_direction, index_lanes_points, b_crosswalk_1, b_crosswalk_2, crosswalk_1
    global crosswalk_2, b_select_crosswalk_1, b_select_crosswalk_2,index_crosswalk, circle_1, circle_2
    global b_select_circle_1, b_select_circle_2, init_hora, fin_hora, carros_fx1, carros_fx2, infra_fx1, infra_fx2
    global bbox, ret, tracker, track_history, zoom, local, numero, portaria, dataPort, dataAfer, orgao 

    print('state 1')
    
    startAnalisys = False
    mask = np.zeros(frame.shape, dtype=np.uint8)
    #roi_corners = np.array(points_shape, dtype=np.int32)
    shape_detection_adj = [[math.floor(x/(zoom/100)) for x in pair ] for pair in shape_detection]
  
    roi_corners = np.array([shape_detection_adj], dtype=np.int32)

    channel_count = frame.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,)*channel_count
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)

    masked_frame = cv2.bitwise_and(frame, mask)

    # predict bounding boxes on frame
    results = model.track(masked_frame, verbose=False, classes=[2,3,5,7,9], persist=True)

    #boxes = results[0].boxes.xywh.cpu()

    boxes = results[0].boxes.xywh.cuda()

    #print(results[0].boxes.cls.int().cpu().tolist())

    masked_frame = results[0].plot()
    crosswalk_1_adj = [[math.floor(x/(zoom/100)) for x in pair ] for pair in crosswalk_1]
    limiter = np.hstack(crosswalk_1_adj).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(masked_frame, [limiter], isClosed=False, color=(1, 56, 106), thickness=10)

    print('state 2')

    if(True):
        x1_p = (circle_1[0][0]-circle_1[1])
        x2_p = (circle_1[0][0]+circle_1[1])
        y1_p = (circle_1[0][1]-circle_1[1])
        y2_p = (circle_1[0][1]+circle_1[1])
        
        x1 = int(x1_p/(zoom/100))
        x2 = int(x2_p/(zoom/100))
        y1 = int(y1_p/(zoom/100))
        y2 = int(y2_p/(zoom/100))

        cropped_light = frame[y1:y2,x1:x2]

    print('state 4')
    if(results[0].boxes.id != None):
        #track_ids = results[0].boxes.id.int().cpu().tolist()
        track_ids = results[0].boxes.id.int().cuda().tolist()
        #objects = results[0].boxes.cls.int().cpu().tolist()
        objects = results[0].boxes.cls.int().cuda().tolist()
        #print("detectou")
        #print(objects)
        

        for box, track_id, obj in zip(boxes, track_ids, objects):
            x, y, w, h = box
            track = track_history[track_id]
            
            track.append((float(x), float(y)))  # x, y center point

            #print(track)
            #local, numero, portaria, dataPort, dataAfer, orgao 

            track_history[track_id] = track
            if (cross_line(track)):
                detection_limit1 = crosswalk_1

                print("detectou passagem")

                now = datetime.now()
                s = now.strftime("%d%m%y%H%M%S%f")
                
                #print([0.6*x for x in detection_limit1[0]])
                #cv2.line(frame, [int(x/(zoom/100)) for x in detection_limit1[0]], [int(x/(zoom/100)) for x in detection_limit1[1]], ( 255, 0, 0), 5)
                #cv2.imwrite(filename, frame)

                fx_aval = det_faixa(track[-1], [(shape_detection_adj[0] , shape_detection_adj[1], [math.floor(x/(zoom/100)) for x in lanes_points[0]], [math.floor(x/(zoom/100)) for x in lanes_points[1]]) if direction else (shape_detection_adj[0] , shape_detection_adj[3], [math.floor(x/(zoom/100)) for x in lanes_points[0]], [math.floor(x/(zoom/100)) for x in lanes_points[1]])][0])

                print("é vermelho " + str(is_red))
                print("faixa " +
                      str(fx_aval))

                height, width, channels = frame.shape

                font                   = cv2.FONT_HERSHEY_SIMPLEX
                x, y0 = (10,int(height*0.8))
                fontScale              = 1.5
                fontColor              = (255,0,0)
                thickness              = 2
                lineType               = 3

                faixa = fx_aval
                
                now = datetime.now()
                data = now.strftime("%d/%m/%Y")
                hora = now.strftime("%Hhrs %Mmin %Sseg")
                nome = "NTAG"+numero+"-"+now.strftime("%y%m%d%H%M%S")+str(faixa)

                text = 'Faixa: ' + str(faixa) + ' Local: ' + str(local) + f"\nArt. 208 CTB - Avancar o sinal Vermelho\n" + "Data: " + str(data) + " Hora: " + str(hora) + f"\n\nLegalizacao - N Equipamento: " + str(numero) + " Portaria: " + portaria + " Data Portaria: " + dataPort + f"\nData Certificacao: " + dataAfer + f"\nOrgao Autuador: " + orgao

                print(text)

                frame_cpy = frame

                cv2.rectangle(frame_cpy, (int(box[0]-box[2]/2), int(box[1]-box[3]/2)), (int(box[0]+box[2]/2), int(box[1]+box[3]/2)), (0, 255, 0), 2)

                #cookie_cutter = frame_cpy[int(box[1]-box[3]/2):int(box[1]+box[3]/2),int(box[0]-box[2]/2):int(box[0]+box[2]/2)]

                #cookie_cutter = [frame1 if faixa == 1 else frame2][0]

                cookie_cutter = frame1
                
                car_exp = 100
                
                height, width = cookie_cutter.shape[:2]
                new_height = int(height * (car_exp / 100))
                new_width = int(width * (car_exp / 100))

                
        
                cookie_cutter = cv2.resize(cookie_cutter, (new_width, new_height))

                height, width = frame_cpy.shape[:2]
                orig_exp = new_height/height
                new_height = int(height * (orig_exp))
                new_width = int(width * (orig_exp))

                frame_cpy = cv2.resize(frame_cpy, (new_width, new_height))

            
                frame_cpy = cv2.hconcat([frame_cpy, cookie_cutter])

                height, width = frame_cpy.shape[:2]

                white_rectangle = np.full((150, width, 3), 255, dtype=np.uint8)
            
                frame_cpy = cv2.vconcat([frame_cpy, white_rectangle])
                
                text_size, _ = cv2.getTextSize(text, font, fontScale, thickness)
                line_height = text_size[1] + 5
                for i, line in enumerate(text.split("\n")):
                    y = y0 + i * line_height
                    cv2.putText(frame_cpy,
                                line,
                                (x, y),
                                font,
                                fontScale,
                                fontColor,
                                thickness,
                                lineType)


                
                filename_r = "imgs\\red\\" + nome+".ntgs.jpg"
                filename_nr = "imgs\\nonred\\" + nome+".ntgs.jpg"
                

                if(is_red):

                    cv2.imwrite(filename_r, frame_cpy)

                    if(fx_aval==1):

                        infra_fx1.append(( track_id, now.strftime("-> %d/%m/%Y %H:%M") ))
                        socketio.emit('if1', { 'data': infra_fx1 })

                    else:

                        infra_fx2.append(( track_id, now.strftime("-> %d/%m/%Y %H:%M") ))
                        socketio.emit('if2', { 'data': infra_fx2 })
                    
                else:
                    cv2.imwrite(filename_nr, frame_cpy)

                    if(fx_aval==1):

                        carros_fx1.append(( track_id, now.strftime("-> %d/%m/%Y %H:%M") ))
                        socketio.emit('fx1', { 'data': carros_fx1 })

                    else:

                        carros_fx2.append(( track_id, now.strftime("-> %d/%m/%Y %H:%M") ))
                        socketio.emit('fx2', { 'data': carros_fx2 })
                    
                
            if len(track) > 100:  # retain 90 tracks for 90 frames
                track.pop(0)

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(masked_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

    if quant>10:
        #print("quadro Analisado")
        quant = 0
    else:
        #print(quant)
        quant += 1
        
    startAnalisys = True


# Replace this with your RTSP URL
rtsp_url = "rtsp://admin:@192.168.1.10:554"
rtsp_url = "rtsp://admin:siemens1@192.168.0.7:554/cam/realmonitor?channel=1&subtype=0"
rtsp_url2 = "rtsp://admin:siemens1@192.168.0.8:554/cam/realmonitor?channel=1&subtype=0"
rtsp_url3 = "rtsp://admin:siemens1@192.168.0.9:554/cam/realmonitor?channel=1&subtype=0"

cap = cv2.VideoCapture(rtsp_url)
cap_aux1 = cv2.VideoCapture(rtsp_url2)
#cap_aux2 = cv2.VideoCapture(rtsp_url3)

#cap = cv2.VideoCapture(0)

def read_frames():
    global totalFrame, novoFrame, totalFrame1, totalFrame2, novoFrame1, novoFrame2, ativar, startAnalisys, cap, bbox, ret, tracker, track_history, crosswalk_1, local, numero, portaria, dataPort, dataAfer, orgao
    global rtsp_url, rtsp_url2, rtsp_url3
    success, frame = cap.read()
    des_sucess1, frame1 = cap_aux1.read()
    #des_sucess2, frame2 = cap_aux2.read()
    
    
    while True:
        success, frame = cap.read()  # Read a frame from the video source
        des_sucess1, frame1 = cap_aux1.read()
        #des_sucess2, frame2 = cap_aux2.read()

        if des_sucess1:
            novoFrame1 = True
            totalFrame1 = frame1
        else:
            totalFrame1=[]
 
        if success:
            novoFrame = True
            totalFrame = frame

            print('ready for')
            if ativar and startAnalisys:
                #print('till here')
                if shape_detection: 
                    print('thread')
                    t = Thread(target = analyze_frame, args =(frame, frame1,))
                    t.start()
                
        
        if not success:
            cap = cv2.VideoCapture(rtsp_url)
            print('not ready for')
            continue

def verify_red_serial():
    global is_red
    ports = serial.tools.list_ports.comports()

    if(len(ports)>0):
        try:
            verifier = serial.Serial(ports[0].name,19200)
            while(1):
                k=verifier.read(1)
                print(k)
                if(k=='N'):
                    is_red=0
                    socketio.emit('red_light', {'data': 'nonred'})
                else:
                    is_red=1
                    socketio.emit('red_light', {'data': 'red'})
        except:
            pass
    else:
        print('Dispositivo não conectado')
        
        

def generate_frames():
    global totalFrame, novoFrame, ativar, startAnalisys

    tserial = Thread(target = verify_red_serial)
    tserial.start()

    t1 = Thread(target = read_frames)
    t1.start()

    #print('GayBriels start')

    while len(totalFrame)==0:
        pass

    #print('GayBriels')
    
    while True:        

        # Convert the frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', totalFrame)
        if not ret:
            break
        frame_bytes = buffer.tobytes()

        # Yield the frame in a response to the client
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/adjust_cam')
def adjust_cam():
    global totalFrame, novoFrame, zoom, startAnalisys
    global button_on, match, b_shape_detection, shape_detection, b_lanes_points, lanes_points, b_select_lane
    global button_press, direction, b_direction, index_lanes_points, b_crosswalk_1, b_crosswalk_2, crosswalk_1
    global crosswalk_2, b_select_crosswalk_1, b_select_crosswalk_2,index_crosswalk, circle_1, circle_2
    global b_select_circle_1, b_select_circle_2, delta_x, delta_y

    cv2.namedWindow('Result')
    cv2.setMouseCallback('Result', click_event)

    startAnalisys = False


    while True:

        while not novoFrame:
            pass

        img = totalFrame

        novoFrame = False

        scale_percent = zoom # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)

        
          
        # resize image
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

        height, width, channels = img.shape
        
        rectangle_width = width // num_rectangles  # Calculate the width of each rectangle

        rectangle_height = height // 10  # Calculate the height of each rectangle (half of the original height)


        # Create a black background for the buttons
        button_area = np.zeros((rectangle_height, width, 3), dtype=np.uint8)
        button_area[:] = rectangle_color

        # Calculate the width at which the rectangles will be placed
        button_width = width // num_rectangles

        top_y = 0

        if(shape_detection!=[]):
            shape = img.copy()
            roi_corners = np.array([shape_detection], dtype=np.int32)
            cv2.fillPoly(shape, roi_corners, (255, 255, 51))
            alpha = 0.7
            mask = shape.astype(bool)
            
            img[mask] = cv2.addWeighted(img, alpha, shape, 1 - alpha, 0)[mask]

        if(button_on):

            for i in range(num_rectangles):
                # # Calculate the coordinates for the rectangle
                top_left = (i * button_width, 0)
                bottom_right = ((i + 1) * button_width, rectangle_height)

                botoes_areas_top[i] = top_left
                botoes_areas_bottom[i] = bottom_right

                # Draw the rectangle on the image with a border
                cv2.rectangle(img, top_left, bottom_right, rectangle_color, -1)  # -1 fills the rectangle
                cv2.rectangle(img, top_left, bottom_right, border_color, border_thickness)

                # Add text to the button
                #text = f"Button {i+1}"
                text_size, _ = cv2.getTextSize(botoes_names[i], font, font_scale, font_thickness)
                text_x = top_left[0] + (button_width - text_size[0]) // 2
                text_y = top_left[1] + (rectangle_height + text_size[1]) // 2
                cv2.putText(img, botoes_names[i], (text_x, text_y), font, font_scale, font_color, font_thickness)

        if b_shape_detection:
            shape = img.copy()
            roi_corners = np.array([shape_detection], dtype=np.int32)
            cv2.fillPoly(shape, roi_corners, (255, 255, 51))
            alpha = 0.7
            mask = shape.astype(bool)
            
            img[mask] = cv2.addWeighted(img, alpha, shape, 1 - alpha, 0)[mask]

        else:
            shape = img.copy()
            for point in shape_detection:
                cv2.circle(shape, point, 25, (255, 255, 255), cv2.FILLED)
            alpha = 0.5
            mask = shape.astype(bool)
            
            img[mask] = cv2.addWeighted(img, alpha, shape, 1 - alpha, 0)[mask]


        if b_lanes_points:
            if(b_direction):
                if(direction):
                    lanes_points = shape_detection[0:2]
                else:
                    lanes_points = shape_detection[1:3]
                b_direction = False
            shape = img.copy()
            #print(lanes_points)
            for point in lanes_points:
                cv2.circle(shape, point, 25, (51, 51, 255), cv2.FILLED)

            if(len(lanes_points)==2):
                cv2.line(img, lanes_points[0], lanes_points[1], ( 255, 255, 255), 5)
                mid_point = ((float(lanes_points[0][0]+lanes_points[1][0])/2),(float(lanes_points[0][1]+lanes_points[1][1])/2))
                point1 = (int(float(mid_point[0])+delta_x),int(float(mid_point[1])+delta_y))
                point2 = (int(float(mid_point[0])-delta_x),int(float(mid_point[1])-delta_y))
                cv2.putText(img, '1', point1, cv2.FONT_HERSHEY_SIMPLEX, 1, ( 255, 255, 255), 2, cv2.LINE_AA )
                cv2.putText(img, '2', point2, cv2.FONT_HERSHEY_SIMPLEX, 1, ( 255, 255, 255), 2, cv2.LINE_AA )
                
            
            alpha = 0.5
            mask = shape.astype(bool)
            
            img[mask] = cv2.addWeighted(img, alpha, shape, 1 - alpha, 0)[mask]
        else:

            if(len(lanes_points)==2):
                cv2.line(img, lanes_points[0], lanes_points[1], ( 255, 255, 255), 5)


        if b_select_crosswalk_1:

            if(len(crosswalk_1) == 0):
                if(not direction):
                    crosswalk_1 = shape_detection[0:2]
                else:
                    crosswalk_1 = shape_detection[1:3]
            
            shape = img.copy()
            #print(lanes_points)
            for point in crosswalk_1:
                cv2.circle(shape, point, 25, (51, 255, 51), cv2.FILLED)

            if(len(crosswalk_1)==2):
                cv2.line(img, crosswalk_1[0], crosswalk_1[1], ( 255, 100, 100), 5)
            
            alpha = 0.5
            mask = shape.astype(bool)
            
            img[mask] = cv2.addWeighted(img, alpha, shape, 1 - alpha, 0)[mask]
            
        else:

            if(len(crosswalk_1)==2):
                cv2.line(img, crosswalk_1[0], crosswalk_1[1], ( 255, 100, 100), 5)

        if b_select_crosswalk_2:

            if(len(crosswalk_2) == 0):
                if(not direction):
                    crosswalk_2 = shape_detection[0:2]
                else:
                    crosswalk_2 = shape_detection[1:3]
            
            shape = img.copy()
            #print(lanes_points)
            for point in crosswalk_2:
                cv2.circle(shape, point, 25, (51, 255, 51), cv2.FILLED)

            if(len(crosswalk_2)==2):
                cv2.line(img, crosswalk_2[0], crosswalk_2[1], ( 100, 255, 100), 5)
            
            alpha = 0.5
            mask = shape.astype(bool)
            
            img[mask] = cv2.addWeighted(img, alpha, shape, 1 - alpha, 0)[mask]
            
        else:

            if(len(crosswalk_2)==2):
                shape = img.copy()
                cv2.line(img, crosswalk_2[0], crosswalk_2[1], ( 100, 255, 100), 5)
            
        if(len(circle_1)==2):
            shape = img.copy()

            cv2.circle(shape, circle_1[0], circle_1[1], (0, 0, 255), cv2.FILLED)

            alpha = 0.5
            mask = shape.astype(bool)
            
            img[mask] = cv2.addWeighted(shape, alpha, shape, 1 - alpha, 0)[mask]

        if(len(circle_2)==2):
            shape = img.copy()

            cv2.circle(shape, circle_2[0], circle_2[1], (0, 0, 255), cv2.FILLED)

            alpha = 0.5
            mask = shape.astype(bool)
            
            img[mask] = cv2.addWeighted(shape, alpha, shape, 1 - alpha, 0)[mask]
         
        cv2.imshow("Result", img)
    
        if (cv2.waitKey(1) & 0xFF) == ord('q'): 
            break

    cv2.destroyAllWindows()

    startAnalisys = True

    write_area()

    return "configurado", 200



@app.route('/read_v')
def read_csv():
    global ativar, init_hora, fin_hora, cap, local, numero, portaria, dataPort, dataAfer, orgao, cap_aux1
    global rtsp_url, rtsp_url2, rtsp_url3
    try:
        with open('data.csv', 'r') as file:
            # Assuming the first row contains headers
            csv_reader = csv.DictReader(file)
            data = [row for row in csv_reader][0]
            data = jsonify(data)
            print(data.json)
            if(data.json!=[]):
                print(data.json['ativar'])
                if(data.json['equipamento']):
                    numero = data.json['equipamento']
                if(data.json['local'] and data.json['sentido']):
                    local = data.json['local'] + " " + data.json['sentido']
                if(data.json['dataCertificacao']):
                    dataAfer = data.json['dataCertificacao'][8:10] + "/" + data.json['dataCertificacao'][5:7] + "/" + data.json['dataCertificacao'][0:4]
                if(data.json['inmetro']):
                    portaria = data.json['inmetro']
                if(data.json['portaria']):
                    dataPort = data.json['portaria'][8:10] + "/" + data.json['portaria'][5:7] + "/" + data.json['portaria'][0:4]
                if(data.json['orgao']):
                    orgao = data.json['orgao']
                if(data.json['ativar']):
                    ativar = True
                if(data.json['rtsp']!=""):
                    rtsp_url = data.json['rtsp']
                    cap = cv2.VideoCapture(rtsp_url)
                    print("Go camera")
                if(data.json['rtsp1']!=""):
                    rtsp_url2 = data.json['rtsp1']
                    cap_aux1 = cv2.VideoCapture(rtsp_url2)
                    print("Go camera")
##                if(data.json['rtsp2']!=""):
##                    rtsp_url3 = data.json['rtsp2']
##                    cap_aux2 = cv2.VideoCapture(rtsp_url3)
##                    print("Go camera")
                if(data.json['tempoInicio']!="" and data.json['tempoFim']!=""):
                    init_hora = data.json['tempoInicio']
                    fin_hora = data.json['tempoFim']
            
            #print(data.ativar)
            return data
    except FileNotFoundError:
        return "CSV file not found", 404
    
@app.route('/download_zip')
def download_zip():
    zip_filename = 'jpg_files.zip'
    zip_path = os.path.join(DOWNLOADS_DIR, zip_filename)

    # Create a zip file containing .jpg files from the local directory
    with open(zip_path, 'wb') as zf:
        import zipfile
        with zipfile.ZipFile(zf, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(LOCAL_FILES_DIR):
                for file in files:
                    if file.lower().endswith('.jpg'):
                        file_path = os.path.join(root, file)
                        zipf.write(file_path, os.path.relpath(file_path, LOCAL_FILES_DIR))

    for root, dirs, files in os.walk(LOCAL_FILES_DIR):
        for file in files:
            if file.lower().endswith('.jpg'):
                file_path = os.path.join(root, file)
                dest_path = os.path.join(PROCESSED_DIR, file)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.move(file_path, dest_path)


    return send_file(zip_path, as_attachment=True)

@app.route('/all_tickets')
def all_tickets():
    relative_folder_path = 'history'

    # Get the absolute path by joining the relative path with the current working directory
    folder_path = os.path.join(os.getcwd(), relative_folder_path)

    # Open the file explorer using the absolute path
    subprocess.Popen(['explorer', folder_path], shell=True)


    return Response(status=204)

@app.route('/write_v', methods=['POST'])
def write_csv():
    global ativar
    try:
        # Get data from the request
        data = request.json

        print(data)

        if not data:
            return "No data provided", 400

        # Assuming the CSV file already exists


        print("ativar = ")
        print(data['ativar'])

        if(data['ativar'] and not ativar):
            ativar = True
        if(not data['ativar'] and ativar):
            ativar = False

        print(ativar)
        
        with open('data.csv', 'w', newline='') as file:
            # Create a CSV writer object
            print(data.keys())
            csv_writer = csv.DictWriter(file, fieldnames=data.keys())

            # Write the header if the file is empty
            csv_writer.writeheader()
            print(data.keys())
            # Write the data
            print(type(data))
            csv_writer.writerow(data)

        return jsonify(data), 201
    except Exception as e:
        print(e)
        return str(e), 500
    

@app.route('/')
def index():
    return """
    <!doctype html>
    <html>
    <head>
        <title>Video Streaming</title>
    </head>
    <body>
        <h1>Live Video Streaming</h1>
        <img src="/video_feed" width="640" height="480" />
    </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def video1_exp():
    
    global cap_aux1, novoFrame1, totalFrame1, zoom
    
    
    if(data.json['rtsp1']!=""):
        cap_aux1 = cv2.VideoCapture(data.json['rtsp1'])
    cv2.namedWindow('Camera 1')
    while True:

        while not novoFrame1:
            pass

        img = totalFrame1

        novoFrame1 = False

        scale_percent = zoom # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        cv2.imshow('Camera 1', img)
    

        if (cv2.waitKey(1) & 0xFF) == ord('q'): 
            break
        
    cv2.destroyAllWindows()


@app.route('/video1')
def video_feed_1():
    t_c1 = Thread(target = video1_exp)
    t_c1.start()
    return "video 1", 200

@app.route("/login", methods=["POST"])
def login():
    username = request.json.get("username", None)
    password = request.json.get("password", None)
    # Validate username and password
    user = authenticate_user(username, password)
    if user:
        access_token = create_access_token(identity=user["id"], expires_delta=False)
        print(access_token)
        return jsonify(access_token=access_token)
    else:
        return jsonify({"msg": "Bad username or password"}), 401

@app.route("/get_user", methods=["POST"])
def user_id():
    username = request.json.get("username", None)
    password = request.json.get("password", None)
    # Validate username and password
    user = authenticate_user(username, password)
    if user:
        return jsonify({"nivel": user["id"]}), 200
    else:
        return jsonify({"msg": "Bad username or password"}), 401

    

def authenticate_user(username, password):
    with open("cred.kik", 'r') as file:
        creds = csv.DictReader(file)
        for cred in creds:
            if(cred['user'] == username):
                if(cred['password'] == password):
                    return cred

        return None

##@socketio.on('/request_data')
##def send_data():
##    emit('data_update', data)
##
##@socketio.on('/connect')
##def handle_connect():
##    emit('connected', 'Connected to the server')
   

if __name__ == '__main__':
    read_area()
    socketio.run(app, debug=True, port=80)
