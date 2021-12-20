# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 18:09:27 2021

@author: lukas
"""

import cv2
import dlib
import numpy as np
import timeit
import utils
import queue
import multiprocessing
import pathlib
import argparse
import time


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--descriptor_file', type=str,
                        default='./saved_descriptors', help='path to descriptor file')
    parser.add_argument('--threshold', type=float, default=0.6,
                        help='Threshold of euclidean distance to distinguish persons.')
    parser.add_argument('--max_angle', type=float, default=4.0, help='maximum rotation angle of the face.')
    parser.add_argument('--max_fps', type=float, default=5.0, help='maximum frame rate of the application.')
    parser.add_argument(
        '--cam_url', type=str,
        default="http://10.0.0.241/zm/cgi-bin/nph-zms?mode=jpeg&monitor=2&maxfps=5&scale=100&user=admin&pass=admin",
        help="IP camera url including username and password")
    parser.add_argument('--landmarkPredictor', type=str,
                        default="../dlib_models/shape_predictor_68_face_landmarks.dat",
                        help="Path to dlib 68 face landmark predictor: shape_predictor_68_face_landmarks.dat")
    parser.add_argument('--faceDescriptor', type=str,
                        default="../dlib_models/dlib_face_recognition_resnet_model_v1.dat",
                        help="Path to dlibs face recognition model: dlib_face_recognition_resnet_model_v1.dat")
    return parser.parse_args()


class StateMachine():
    def __init__(self, url, predictorPath, facerecPath, descriptorFilePath,
                 threshold=0.6, maxFps: float = 5.0, imgSize: int = 384,
                 maxAngle: float = 4.0, MaxMovement=50.0, showLandmarks: bool = False) -> None:

        predictor = dlib.shape_predictor(predictorPath)
        detector = dlib.get_frontal_face_detector()

        self.__decriptorHandler = utils.Descriptor_FileHandler(descriptorFilePath, threshold)

        self.__minLatency = 1000.0/maxFps
        self.__imgSize = imgSize
        self.__maxAngle = maxAngle
        self.__MaxMovement = MaxMovement
        self.__showLandmarks = showLandmarks

        self.__state = self.___waitForFace
        self.__imgPos = np.zeros((4), dtype=np.int32)
        self.__name = "processing.."
        self.__faceDetected = False
        self.__score = 0.0
        self.__maxMissDetection = 2
        self.__missDetections = 0
        self.__ReqHeadRot = 8.0

        self.__imgQueue = queue.Queue(maxsize=3)
        self.__detectQueue = queue.Queue(maxsize=3)
        self.__faceRecQueueIn = multiprocessing.Queue(maxsize=3)
        self.__faceRecQueueOut = multiprocessing.Queue(maxsize=3)

        self.__ImageWorker = utils.Image_loader(self.__imgQueue, url, imgSize,
                                                maxFps)
        self.__DetectionWorker = utils.Detector(self.__imgQueue, self.__detectQueue,
                                                detector, predictor)

        self.__FaceRecWorker = utils.FaceDecriptorProcess(self.__faceRecQueueIn, self.__faceRecQueueOut, 
                                                          facerecPath)

        self.__ImageWorker()
        self.__DetectionWorker()
        self.__FaceRecWorker()

    def __del__(self) -> None:
        return self.__FaceRecWorker.kill()

    def __call__(self, key) -> np.ndarray:
        # execute current state
        return self.__state(key)

    def __discardCurrentDescriptor(self) -> None:
        # If Output Queue is empty -> Face descriptor computations are not done yet --> kill the process and restart it
        self.__name == "processing.."
        self.__faceDetected = False
        self.__score = 0.0
        
        if self.__faceRecQueueOut.empty():
            self.__FaceRecWorker.kill()
            self.__FaceRecWorker()
            return
        # If Output Queue is not empty -> Face descriptor computations are done --> discard the face descriptor in the Queue
        _ = self.__faceRecQueueOut.get()
        return

    def ___waitForFace(self, key) -> np.ndarray:
        rects, shapes, img = self.__detectQueue.get()

        # Multiple persons
        if len(rects) > 1:
            outtext = "Error: Multiple faces detected."
            utils.write_text_bottom(img, outtext, (0, 0, 255))
            return img

        # No Person
        if len(rects) < 1:
            outtext = "Error: No faces detected."
            utils.write_text_bottom(img, outtext, (0, 0, 255))
            return img

        shape_np = utils.shape_to_np(shapes[0])

        # Check face alignment
        rot_angle = utils.get_angle(shape_np[27, :]-shape_np[30, :],
                                    shape_np[27, :]-shape_np[33, :])
        tilt_angle = utils.get_angle(shape_np[45, :]-shape_np[36, :],
                                     np.array([1, 0]))

        if np.abs(rot_angle) > self.__maxAngle or np.abs(tilt_angle) > self.__maxAngle:
            outtext = "Look straight into the camera. Current rot angle: " + \
                str(rot_angle) + " tilt angle: " + str(tilt_angle)

            utils.write_text_bottom(img, outtext, (0, 127, 255))
            return img

        # start Computation of face descriptor
        self.__faceRecQueueIn.put((shapes, img))

        # draw rectangle
        rect_np = utils.rect_to_np(rects[0], dtpye=np.int32)  # convert dlib rectangle to numpy
        img = utils.drawBoxAndName(img, rect_np, self.__name, self.__score)

        # draw landmarks
        if self.__showLandmarks:
            for (px, py) in shape_np:
                cv2.circle(img, (int(px), int(py)), 1, (255, 0, 0), -1)

        # store face position for tracking
        self.__imgPos = rect_np

        # next state --> Do life check look left
        self.__state = self.__lifeCheckLookLeft
        return img

    def __lifeCheckLookLeft(self, key) -> np.ndarray:
        return self.__lifeCheckLookLeftRight(False, self.__lifeCheckLookRight)

    def __lifeCheckLookRight(self, key) -> np.ndarray:
        return self.__lifeCheckLookLeftRight(True, self.__tracking)

    def __lifeCheckLookLeftRight(self, nLeftRight, nextState) -> np.ndarray:
        rects, shapes, img = self.__detectQueue.get()
        # Multiple persons
        if len(rects) > 1:
            outtext = "Error: Multiple faces detected."
            utils.write_text_bottom(img, outtext, (0, 0, 255))
            self.__missDetections += 1
            if self.__missDetections > self.__maxMissDetection:
                self.__discardCurrentDescriptor()
                self.__state = self.___waitForFace
            return img

        # No Person
        if len(rects) < 1:
            outtext = "Error: No faces detected."
            utils.write_text_bottom(img, outtext, (0, 0, 255))
            self.__missDetections += 1
            if self.__missDetections > self.__maxMissDetection:
                self.__discardCurrentDescriptor()
                self.__state = self.___waitForFace
            return img

        # Check for feasible movement -> If face jumps around most properly it is no real person
        rect_np = utils.rect_to_np(rects[0], dtpye=np.int32)  # convert dlib rectangle to numpy
        movement = np.linalg.norm(rect_np-self.__imgPos)
        if movement > self.__MaxMovement:
            self.__discardCurrentDescriptor()
            self.__state = self.___waitForFace
            return img

        shape_np = utils.shape_to_np(shapes[0])

        # Check face alignment
        rot_angle = utils.get_angle(shape_np[27, :]-shape_np[30, :],
                                    shape_np[27, :]-shape_np[33, :])

        if not nLeftRight and (rot_angle < self.__ReqHeadRot):
            outtext = "Rotate your head LEFT. Current rot angle: " + str(rot_angle)
            utils.write_text_bottom(img, outtext, (100, 255, 255))
            cv2.arrowedLine(img, (30, int(self.__imgSize/2)), (5, int(self.__imgSize/2)),
                            (100, 255, 255), 3)
        elif nLeftRight and (rot_angle > (-self.__ReqHeadRot)):
            outtext = "Rotate your head RIGHT. Current rot angle: " + str(rot_angle)
            utils.write_text_bottom(img, outtext, (255, 255, 100))
            cv2.arrowedLine(img, (self.__imgSize-30, int(self.__imgSize/2)-5),
                            (self.__imgSize, int(self.__imgSize/2)),
                            (255, 255, 100), 3)
        else:
            outtext = "Good"
            utils.write_text_bottom(img, outtext, (0, 255, 0))
            # next state --> Do life check look left
            self.__state = nextState

        img = utils.drawBoxAndName(img, rect_np, self.__name, self.__score)
        if self.__showLandmarks:
            for (px, py) in shape_np:
                cv2.circle(img, (int(px), int(py)), 1, (255, 0, 0), -1)
        self.__imgPos = rect_np
        return img

    def __tracking(self, key):
        rects, shapes, img = self.__detectQueue.get()
        # Multiple persons
        if len(rects) > 1:
            outtext = "Error: Multiple faces detected."
            utils.write_text_bottom(img, outtext, (0, 0, 255))
            self.__missDetections += 1
            if self.__missDetections > self.__maxMissDetection:
                self.__discardCurrentDescriptor()
                self.__state = self.___waitForFace
            return img

        # No Person
        if len(rects) < 1:
            outtext = "Error: No faces detected."
            utils.write_text_bottom(img, outtext, (0, 0, 255))
            self.__missDetections += 1
            if self.__missDetections > self.__maxMissDetection:
                self.__discardCurrentDescriptor()
                self.__state = self.___waitForFace
            return img

        # Check for feasible movement -> If face jumps around most properly it is no real person
        rect_np = utils.rect_to_np(rects[0], dtpye=np.int32)  # convert dlib rectangle to numpy
        movement = np.linalg.norm(rect_np-self.__imgPos)
        if movement > self.__MaxMovement:
            self.__discardCurrentDescriptor()
            self.__state = self.___waitForFace
            return img

        shape_np = utils.shape_to_np(shapes[0])

        # Check face alignment
        rot_angle = utils.get_angle(shape_np[27, :]-shape_np[30, :],
                                    shape_np[27, :]-shape_np[33, :])
        tilt_angle = utils.get_angle(shape_np[45, :]-shape_np[36, :],
                                     np.array([1, 0]))

        if self.__faceDetected:
            outtext = self.__name + " detected with {}\% confidence.".format(self.__score)
            utils.write_text_bottom(img, outtext, (0, 255, 0))
        else:
            if self.__faceRecQueueOut.empty():
                outtext = self.__name
                utils.write_text_bottom(img, outtext, (255, 0, 0))
            else:
                faceDescriptor = self.__faceRecQueueOut.get()
                self.__faceDetected, self.__name = self.__decriptorHandler.exists(faceDescriptor)
                self.__score = 99.38  # dlib face recognition accuracy

        img = utils.drawBoxAndName(img, rect_np, self.__name, self.__score)
        if self.__showLandmarks:
            for (px, py) in shape_np:
                cv2.circle(img, (int(px), int(py)), 1, (255, 0, 0), -1)
        self.__imgPos = rect_np
        return img


def main(opt):

    stm = StateMachine(opt.cam_url, opt.landmarkPredictor, opt.faceDescriptor,
                       opt.descriptor_file, opt.threshold,
                       maxFps=opt.max_fps, imgSize=384, showLandmarks=True)

    key = 0
    while(True):
        img = stm(key)
        cv2.imshow("Face Detector", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Destroy all the windows
    cv2.destroyAllWindows()
    print("Leave Face Detection")


if __name__ == '__main__':
    opt = _create_parser()
    print(opt)
    main(opt)
