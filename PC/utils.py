# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 19:04:48 2021

@author: lukas
"""

import numpy as np
import cv2
import dlib

from typing import Union
import pathlib

from queue import Queue
import threading
import multiprocessing
import time
import timeit


def letterbox(
        img, new_shape=(416, 416),
        color=(114, 114, 114),
        auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
        new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / \
            shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def write_text_bottom(img, outtext: str, color=(0, 0, 255)):
    (tw, th), _ = cv2.getTextSize(outtext, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
    img = cv2.rectangle(img, (0, img.shape[0] - 20), (0 + tw, img.shape[0]), color, -1)
    textColor = (0, 0, 0) if max(color) > 127 else (255, 255, 255)
    img = cv2.putText(img, outtext, (0, img.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX,
                      0.3, textColor, 1)
    return


def write_text_top(img, outtext: str, color=(0, 0, 255)):
    (tw, th), _ = cv2.getTextSize(outtext, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
    img = cv2.rectangle(img, (0, 20), (0 + tw, img.shape[0]), color, -1)
    textColor = (0, 0, 0) if max(color) > 127 else (255, 255, 255)
    img = cv2.putText(img, outtext, (0, 5), cv2.FONT_HERSHEY_SIMPLEX,
                      0.3, textColor, 1)
    return


def rect_to_np(rect, dtpye=np.int32):
    x1 = rect.left()
    y1 = rect.top()
    x2 = rect.right()
    y2 = rect.bottom()
    return np.array([x1, y1, x2, y2], dtype=dtpye)


def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def shape_to_np(shape, dtype=np.int32):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return coords


def get_angle(a, b):
    #a = a / sum(a**2)
    #b = b / sum(b**2)
    return np.arccos(np.dot(a,b)/np.sqrt(sum(a**2)*sum(b**2)))*180/np.pi*np.sign(a[0])


def drawBoxAndName(image, bbox, name, score):
    fontScale = 0.3
    bbox_color = (0, 255, 0)
    c1, c2 = (bbox[0], bbox[1]), (bbox[2], bbox[3])
    cv2.rectangle(image, c1, c2, bbox_color, 2)
    if score > 0.0:
        bbox_mess = '%s: %.2f' % (name, score)
    else:
        bbox_mess = '%s' % (name)
    t_size = cv2.getTextSize(bbox_mess, cv2.FONT_HERSHEY_SIMPLEX, fontScale, 1)[0]
    c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
    cv2.rectangle(image, c1, (np.float32(c3[0]), np.float32(c3[1])), bbox_color, -1)  # filled
    cv2.putText(image, bbox_mess, (c1[0], np.float32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                fontScale, (0, 0, 0),1, lineType=cv2.LINE_AA)
    return image


class Descriptor_FileHandler():
    def __init__(self, file_path, threshold=0.6) -> None:
        self.threshold = threshold
        self.__knownPersons = {}
        self.file_path = pathlib.Path(file_path)
        if self.file_path.exists():
            with open(self.file_path, 'r') as descriptor_file:
                # Cache labels
                for num, line in enumerate(descriptor_file):
                    desc = line.strip().split(',')
                    self.__knownPersons[desc[0]] = np.array(desc[1:]).astype(np.float32)

    def exists(self, descriptor) -> Union[bool, str]:
        for key, value in self.__knownPersons.items():
            euclidean_distance = np.linalg.norm(value-descriptor)
            if euclidean_distance < self.threshold:
                return True, key
        return False, "unknown"

    def add(self, new_name: str, descriptor: np.ndarray) -> Union[bool, str]:
        exists, name = self.exists(descriptor)
        if exists:
            if name != new_name:
                return False, "[Error]: Person already registered as {}. New name: {}.".format(
                    name, new_name)
            else:
                return False, "[Info]: Person {} already registered.".format(new_name)

        if new_name in self.__knownPersons:
            return False, "[Error]: Person {} already registered but descriptor does not match.".format(new_name)

        self.__knownPersons[name] = descriptor
        with open(self.file_path, 'a+') as descriptor_file:
            new_line = [new_name]
            new_line = new_line + list(descriptor.astype(str))
            new_line = ','.join(new_line) + "\n"
            descriptor_file.write(new_line)
        return True, "[Info]: Person successfully registered."


class Worker():
    def __init__(self):
        self.thread = None

    def __call__(self):
        self.thread = threading.Thread(target=self._process,
                                       daemon=True)
        self.thread.start()
        print("Started {}".format(self.__class__.__name__))

    def _process(self):
        raise NotImplemented("implement a process method")


class Image_loader(Worker):
    def __init__(self, out_Queue: Queue, video_in, image_size, frame_rate):
        super().__init__()
        self.out_Queue = out_Queue
        self.camera = cv2.VideoCapture(video_in)
        self.image_size = image_size
        self.time = int(1000/frame_rate)
        print("Init done")

    def __del__(self):
        self.camera.release()

    def _process(self):
        while(True):
            #print(self.__class__.__name__ + ": fetching image.")
            start = timeit.default_timer()
            ret, frame = self.camera.read()
            image_data, ratio, pad = letterbox(frame, (self.image_size, self.image_size),
                                               auto=False, scaleup=True)
            self.out_Queue.put(image_data)
            sleep_time = self.time - (timeit.default_timer() - start)
            sleep_time = np.maximum(0, sleep_time)
            time.sleep(sleep_time/1000)
            #print(self.__class__.__name__ + ": done.")


class Detector(Worker):
    def __init__(self, in_Queue: Queue, out_Queue: Queue, detector, predictor):
        super().__init__()
        self.in_Queue = in_Queue
        self.out_Queue = out_Queue
        self.detector = detector
        self.predictor = predictor

    def _process(self):
        while(True):
            #print(self.__class__.__name__ + ": detecting faces.")
            img = self.in_Queue.get()
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            rects = self.detector(gray, 1)
            shapes = []
            for rect in rects:
                shapes.append(self.predictor(gray, rect))

            faces = (rects, shapes, img)
            self.out_Queue.put(faces)
            #print(self.__class__.__name__ + ": done.")



def computeFaceDescriptors(queueIn: multiprocessing.Queue, queueOut: multiprocessing.Queue, facerecPath):
    facerec = dlib.face_recognition_model_v1(facerecPath)
    while(True):
        print("wait for queue entry")
        shapes, img = queueIn.get()
        print("start processing face descriptor")
        face_descriptors = []
        for shape in shapes:
            face_descriptors.append(
                facerec.compute_face_descriptor(img, shape))
        print("Face decriptor processing done.")
        time.sleep(5)
        print("Put face decriptor in output queue.")
        queueOut.put(face_descriptors)

class FaceDecriptorProcess():
    def __init__(self, queueIn: multiprocessing.Queue, queueOut: multiprocessing.Queue, facerecPath):
        self.process = None
        self.queueIn = queueIn
        self.queueOut = queueOut
        self.facerecPath = facerecPath

    def __call__(self):
        self.process = multiprocessing.Process(target=computeFaceDescriptors,
                                               args=(self.queueIn, self.queueOut, self.facerecPath),
                                               daemon=True)
        self.process.start()

    def kill(self):
        self.process.terminate()