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
from queue import Queue
import pathlib
import argparse
import time


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--descriptor_file', type=str,
                        default='./saved_descriptors', help='path to descriptor file. If not descriptor file exists yet it creates a file.')
    parser.add_argument('--threshold', type=float, default=0.6,
                        help='Threshold of euclidean distance to distinguish persons. 0.6 default value for dlib.')
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


def main(opt):
    predictor = dlib.shape_predictor(opt.landmarkPredictor)
    detector = dlib.get_frontal_face_detector()
    facerec = dlib.face_recognition_model_v1(opt.faceDescriptor)

    DecriptorHandler = utils.Descriptor_FileHandler(opt.descriptor_file, opt.threshold)

    img2detect_queue = Queue(maxsize=3)
    detect2desc_queue = Queue(maxsize=3)
    desc2plot_queue = Queue(maxsize=3)

    w_image_loader = utils.Image_loader(img2detect_queue, opt.cam_url, 384, 5)
    w_detector = utils.Detector(img2detect_queue, detect2desc_queue, detector, predictor)

    w_image_loader()
    w_detector()

    while(True):

        rects, shapes, img = detect2desc_queue.get()
        if len(rects) > 1:
            outtext = "Error: Multiple faces dectected."
            (tw, th), _ = cv2.getTextSize(outtext, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            img = cv2.rectangle(img, (0, 384 - 20), (0 + tw, 384), (255, 0, 0), -1)
            img = cv2.putText(
                img, outtext, (0, 384 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0),
                1)
            cv2.imshow("Face Detector", img)
            continue
        if len(rects) < 1:
            outtext = "Error: No faces detected."
            (tw, th), _ = cv2.getTextSize(outtext, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            img = cv2.rectangle(img, (0, 384 - 20), (0 + tw, 384), (255, 0, 0), -1)
            img = cv2.putText(
                img, outtext, (0, 384 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0),
                1)
            cv2.imshow("Face Detector", img)
            continue

        (x, y, w, h) = utils.rect_to_bb(rects[0])
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

        shape_np = utils.shape_to_np(shapes[0])
        for (px, py) in shape_np:
            cv2.circle(img, (int(px), int(py)), 1, (255, 0, 0), -1)

        rot_angle = utils.get_angle(
            shape_np[27, :] - shape_np[30, :],
            shape_np[27, :] - shape_np[33, :])
        tilt_angle = utils.get_angle(shape_np[45, :]-shape_np[36, :], np.array([1, 0]))
        if np.abs(rot_angle) > opt.max_angle or np.abs(tilt_angle) > opt.max_angle:
            outtext = "Look straight into the camera. Current rot angle: " + \
                str(rot_angle) + " tilt angle: " + str(tilt_angle)
            (tw, th), _ = cv2.getTextSize(outtext, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            img = cv2.rectangle(img, (0, 384 - 20), (0 + tw, 384), (255, 0, 0), -1)
            img = cv2.putText(
                img, outtext, (0, 384 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0),
                1)
            key = cv2.waitKey(1) & 0xFF
        else:
            outtext = "Save face? [s]: "
            (tw, th), _ = cv2.getTextSize(outtext, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            img = cv2.rectangle(img, (0, 384 - 20), (0 + tw, 384), (0, 0, 255), -1)
            img = cv2.putText(
                img, outtext, (0, 384 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0),
                1)

            cv2.imshow("Face Detector", img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                desc = np.array(facerec.compute_face_descriptor(img, shapes[0])).astype(np.float32)
                name = input("Type new user name: ").strip()
                ret, msg = DecriptorHandler.add(name, desc)
                if ret:
                    (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    img = cv2.rectangle(img, (0, 384 - 20), (0 + tw, 384), (0, 255, 0), -1)
                    img = cv2.putText(
                        img, outtext, (0, 384 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0),
                        1)
                    cv2.imshow("Face Detector", img)
                    time.sleep(3)

        cv2.imshow("Face Detector", img)
        if key == ord('q'):
            break

    # Destroy all the windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
    opt = _create_parser()
    print(opt)
    main(opt)
