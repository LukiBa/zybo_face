import cv2
import dlib
import numpy as np
import timeit
import core.utils as utils
import core.faceDetect as faceDetect
import queue
import multiprocessing
import pathlib
import os
import argparse
from elevate import elevate
from intuitus_nn import Framebuffer

imageSize = 384


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=0.6,
                        help='Threshold of euclidean distance to distinguish persons.')
    parser.add_argument('--max_angle', type=float, default=4.0,
                        help='maximum rotation angle of the face.')
    parser.add_argument('--max_fps', type=float, default=5.0,
                        help='maximum rotation angle of the face.')
    parser.add_argument(
        '--cam_url', type=str,
        default="http://10.0.0.241/zm/cgi-bin/nph-zms?mode=jpeg&monitor=2&scale=100&user=admin&pass=admin")
    parser.add_argument('--descriptor_file', type=str,
                        default='./descriptors', help='path to descriptor file')
    parser.add_argument('--intuitusCommandPath', type=str,
                        default="./zybo/face-commands",
                        help="path to intuitus commands")
    parser.add_argument('--landmarkPredictor', type=str,
                        default="./dlib_models/shape_predictor_68_face_landmarks.dat",
                        help="path to dlib 68 face landmark model")
    parser.add_argument('--faceDescriptor', type=str,
                        default="./dlib_models/dlib_face_recognition_resnet_model_v1.dat",
                        help="path to dlib face recognition resnet model v1")

    return parser.parse_args()


class StateMachine():
    def __init__(self, url, DetectorCommandPath, predictorPath, facerecPath, descriptorFilePath,
                 threshold=0.6, maxFps: float = 5.0, imgSize: int = 384,
                 maxAngle: float = 4.0, MaxMovement=50.0, showLandmarks: bool = False) -> None:

        predictor = dlib.shape_predictor(predictorPath)
        detector = faceDetect.Detector(DetectorCommandPath, imgSize, 0.03, 0.5, 0.08)

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
        self.__processingDesc = False
        self.__score = 0.0
        self.__maxMissDetection = 2
        self.__missDetections = 0
        self.__ReqHeadRot = 8.0

        self.__imgQueue = queue.Queue(maxsize=2)
        self.__detectQueue = queue.Queue(maxsize=2)
        self.__faceRecQueueIn = multiprocessing.Queue(maxsize=2)
        self.__faceRecQueueOut = multiprocessing.Queue(maxsize=2)

        self.__ImageWorker = utils.Image_loader(self.__imgQueue, url, imgSize)
        self.__DetectionWorker = utils.Detector(self.__imgQueue, self.__detectQueue,
                                                detector, predictor)

        self.__FaceRecWorker = utils.FaceDecriptorProcess(
            self.__faceRecQueueIn, self.__faceRecQueueOut, facerecPath)

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
        self.__name = "processing.."
        self.__faceDetected = False
        self.__score = 0.0
        if self.__processingDesc:
            if self.__faceRecQueueOut.empty():
                self.__FaceRecWorker.kill()
                self.__FaceRecWorker()
                return
            # If Output Queue is not empty -> Face descriptor computations are done --> discard the face descriptor in the Queue
            _ = self.__faceRecQueueOut.get()
            self.__processingDesc = False
        return

    def ___waitForFace(self, key) -> np.ndarray:
        bboxes, shapes, img = self.__detectQueue.get()

        # Multiple persons
        if bboxes.shape[0] > 1:
            outtext = "Error: Multiple faces detected."
            utils.write_text_bottom(img, outtext, (0, 0, 255))
            return img

        # No Person
        if bboxes.shape[0] < 1:
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
        self.__processingDesc = True
        self.__faceRecQueueIn.put((shapes, img))

        # draw rectangle
        img = utils.drawBoxAndName(img, bboxes[0, :], self.__name, self.__score)

        # draw landmarks
        if self.__showLandmarks:
            for (px, py) in shape_np:
                cv2.circle(img, (int(px), int(py)), 1, (255, 0, 0), -1)

        # store face position for tracking
        self.__imgPos = bboxes[0, :]

        # next state --> Do life check look left
        self.__state = self.__lifeCheckLookLeft
        return img

    def __lifeCheckLookLeft(self, key) -> np.ndarray:
        return self.__lifeCheckLookLeftRight(False, self.__lifeCheckLookRight)

    def __lifeCheckLookRight(self, key) -> np.ndarray:
        return self.__lifeCheckLookLeftRight(True, self.__tracking)

    def __lifeCheckLookLeftRight(self, nLeftRight, nextState) -> np.ndarray:
        bboxes, shapes, img = self.__detectQueue.get()
        # Multiple persons
        if bboxes.shape[0] > 1:
            outtext = "Error: Multiple faces detected."
            utils.write_text_bottom(img, outtext, (0, 0, 255))
            self.__missDetections += 1
            if self.__missDetections > self.__maxMissDetection:
                self.__discardCurrentDescriptor()
                self.__state = self.___waitForFace
            return img

        # No Person
        if bboxes.shape[0] < 1:
            outtext = "Error: No faces detected."
            utils.write_text_bottom(img, outtext, (0, 0, 255))
            self.__missDetections += 1
            if self.__missDetections > self.__maxMissDetection:
                self.__discardCurrentDescriptor()
                self.__state = self.___waitForFace
            return img

        # Check for feasible movement -> If face jumps around most properly it is no real person
        movement = np.linalg.norm(bboxes[0, :]-self.__imgPos)
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
            cv2.arrowedLine(img, (self.__imgSize-30, int(self.__imgSize/2)),
                            (self.__imgSize-5, int(self.__imgSize/2)),
                            (255, 255, 100), 3)
        else:
            outtext = "Good"
            utils.write_text_bottom(img, outtext, (0, 255, 0))
            # next state --> Do life check look left
            self.__state = nextState

        img = utils.drawBoxAndName(img, bboxes[0, :], self.__name, self.__score)
        if self.__showLandmarks:
            for (px, py) in shape_np:
                cv2.circle(img, (int(px), int(py)), 1, (255, 0, 0), -1)
        self.__imgPos = bboxes[0, :]
        return img

    def __tracking(self, key):
        bboxes, shapes, img = self.__detectQueue.get()
        # Multiple persons
        if bboxes.shape[0] > 1:
            outtext = "Error: Multiple faces detected."
            utils.write_text_bottom(img, outtext, (0, 0, 255))
            self.__missDetections += 1
            if self.__missDetections > self.__maxMissDetection:
                self.__discardCurrentDescriptor()
                self.__state = self.___waitForFace
            return img

        # No Person
        if bboxes.shape[0] < 1:
            outtext = "Error: No faces detected."
            utils.write_text_bottom(img, outtext, (0, 0, 255))
            self.__missDetections += 1
            if self.__missDetections > self.__maxMissDetection:
                self.__discardCurrentDescriptor()
                self.__state = self.___waitForFace
            return img

        # Check for feasible movement -> If face jumps around most properly it is no real person
        movement = np.linalg.norm(bboxes[0, :]-self.__imgPos)
        if movement > self.__MaxMovement:
            self.__discardCurrentDescriptor()
            self.__state = self.___waitForFace
            return img

        if self.__faceDetected:
            outtext = self.__name + " detected with {}% confidence.".format(self.__score)
            utils.write_text_bottom(img, outtext, (0, 255, 0))
        else:
            if self.__faceRecQueueOut.empty():
                outtext = self.__name
                utils.write_text_bottom(img, outtext, (255, 0, 0))
            else:
                faceDescriptor = self.__faceRecQueueOut.get()
                self.__processingDesc = False
                self.__faceDetected, self.__name = self.__decriptorHandler.exists(faceDescriptor)
                self.__score = 99.38  # dlib face recognition accuracy

        img = utils.drawBoxAndName(img, bboxes[0, :], self.__name, self.__score)
        if self.__showLandmarks:
            shape_np = utils.shape_to_np(shapes[0])
            for (px, py) in shape_np:
                cv2.circle(img, (int(px), int(py)), 1, (255, 0, 0), -1)
        self.__imgPos = bboxes[0, :]
        return img


def main(opt):
    # get root permissions (Required to access device drivers)
    if os.getuid() != 0:
        try: elevate() # get root privileges. Required to open kernel driver 
        except: raise Exception("Root permission denied")        
    # Initialize framebuffer
    fb = Framebuffer('/dev/fb0')
    screen_size = fb.get_screensize()
    print(screen_size)
    black_screen = np.zeros([screen_size[0], screen_size[1], 3], dtype=np.uint8)
    fb.show(black_screen, 0)
    fbOffsetCenter = int(
        (screen_size[1] * (screen_size[0]-imageSize)/2 + (screen_size[1]-imageSize)/2)*3)

    # Start state machine
    stm = StateMachine(opt.cam_url, opt.intuitusCommandPath, opt.landmarkPredictor, opt.faceDescriptor,
                       opt.descriptor_file, opt.threshold,
                       maxFps=opt.max_fps, imgSize=imageSize, showLandmarks=False)

    # Setup keyboard thread
    keyboardInput = utils.KeyInput()
    keyboardInput()
    key = keyboardInput.getKeyboardInput()
    time = timeit.default_timer()
    meanFps = 0
    while(True):
        img = stm(key)
        new_time = timeit.default_timer()
        meanFps = (meanFps + 1/(new_time-time))/2  
        time = new_time
        utils.write_text_top(img, str(meanFps), (0, 127, 0))
        fb.show(img, fbOffsetCenter)
        print("Plot: {}".format((timeit.default_timer()-time)*1000))
        
        key = keyboardInput.getKeyboardInput()
        if key == 'q':
            break

    # Destroy all the windows
    print("Leave Face Detection")


if __name__ == '__main__':
    opt = _create_parser()
    print(opt)
    main(opt)
