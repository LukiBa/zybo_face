# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 18:09:27 2021

@author: lukas
"""

import queue
import cv2
import dlib
import numpy as np
import timeit
import utils
from queue import Queue

predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()
facerec = dlib.face_recognition_model_v1("./dlib_face_recognition_resnet_model_v1.dat")

url = "http://10.0.0.241/zm/cgi-bin/nph-zms?mode=jpeg&monitor=2&maxfps=5&scale=100&user=admin&pass=admin"

img2detct_queue = Queue(maxsize=3)
detect2desc_queue = Queue(maxsize=3)
desc2plot_queue = Queue(maxsize=3)

print("Create Workers")

w_image_loaader = utils.Image_loader(img2detct_queue,url,384,5)
w_detector = utils.Detector(img2detct_queue,detect2desc_queue,detector,predictor)
#w_plotter = utils.Plotter(detect2desc_queue,"Face Detector")

print("Start Workers")

w_image_loaader()
w_detector()       

time = timeit.default_timer()       
while(True):
    
    rects, shapes, img = detect2desc_queue.get()
    face_descriptors = []
    for rect in rects:
        (x, y, w, h) = utils.rect_to_bb(rect)
        cv2.rectangle(img, (int(x), int(y)), 
                      (int(x + w), int(y + h)), 
                      (0, 255, 0), 2)

    for shape in shapes:
        shape_np = utils.shape_to_np(shape)
        for (px, py) in shape_np:
            cv2.circle(img, (int(px), int(py)), 1, (255, 0, 0), -1)
        cv2.line(img, tuple(shape_np[27,:]), tuple(shape_np[30,:]), [255, 0, 0], 2)      
        cv2.line(img, tuple(shape_np[27,:]), tuple(shape_np[33,:]), [255, 0, 0], 2)    
        cv2.line(img, tuple(shape_np[36,:]), tuple(shape_np[45,:]), [127, 127, 0], 2)    
        rot_angle = utils.get_angle(shape_np[27,:]-shape_np[30,:],shape_np[27,:]-shape_np[33,:])
        tilt_angle = utils.get_angle(shape_np[45,:]-shape_np[36,:],np.array([1,0]))
        outtext = "rot: " + str(rot_angle) + " tilt: " + str(tilt_angle)
        img = cv2.putText(img, outtext, (0, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)   
        
        
    new_time = timeit.default_timer()
    outtext = "fps: " + str(1/(new_time-time))
    time = new_time
    (tw, th), _ = cv2.getTextSize(outtext, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    img = cv2.rectangle(img, (0, 384 - 20), (0 + tw, 384), (0, 255, 0), -1)
    img = cv2.putText(img, outtext, (0, 384 - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)   
          
    cv2.imshow("Face Detector", img)    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Destroy all the windows
cv2.destroyAllWindows()

