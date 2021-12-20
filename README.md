# Zybo face
![GitHub](https://img.shields.io/github/license/LukiBa/zybo_yolo)  
Face verification based on intuitus hardware accelerator and dlib using a ZYBO-Z7-20 FPGA board. 

## Features
- [x] Create face descriptor file using a python PC application
- [x] Capture video stream from IP camera
- [x] Face detection using YOLOv3 tiny and Intuitus hardware accelerator
- [x] Landmark prediction using dlib
- [x] Facial pose estimation. 
- [x] Face recognition using dlibs face recognition resnet model v1
- [x] Live verification 
- [x] Plot results to hdmi screen
- [x] frame rate up to 5 fps
- [x] Power consumption < 3.5W 

## Create and manage face descriptor files using your PC:
1. install python packages: numpy, opencv, dlib, pathlib
2. Clone repository and cd into it
3. download shape_predictor_68_face_landmarks.dat and dlib_face_recognition_resnet_model_v1.dat from dlib webpage.
4. Run manage face descriptor files application using:
````sh
python PC/manage_descriptors.py
````
For information about usage run ``python PC/manage_descriptors.py --help``
To run an emulation of the face verification application on your PC use:
````sh
python PC/application_pc.py
````

## Run test application on your Zynq-7000 board:
1. install python packages: numpy, opencv, dlib, pathlib
2. install Intuitus driver interface package: <https://github.com/LukiBa/Intuitus-intf.git>

Run application using:
````sh
python zybo/application.py
````

## Hardware Requirements
- Zybo-Z7-20 Board or similar (e.g. Zedboard)
- microSD card >=8GB 
- Optional:
    - HDMI Display 

## Setup Hardware
1. SD card: Format first Partition (300-500MB) to FAT and the second partition to EXT4
2. Copy BOOT.BIN and image.ub to the first partition 
3. Download linaro-developer and extract it to the second partition (tested version: stretch)
4. Copy intuitus.ko to extracted rootfs.
5. Connect ethernet, display, keyboard and power to Zybo and insert the SD card
6. Boot up the system
7. Update the OS using ``sudo apt-get upgrade && apt-get update``
8. Install build-essentials, cmake, opencv and swig
9. Install PyEnv
10. Install python 3.6.10 and activate it
11. Install numpy, opencv-python, dlib, elevate, pathlib, wget and argparse using pip 
12. Clone <https://github.com/LukiBa/Intuitus-intf.git> and cd into it
13. Install the interface using ``pip install -e .`` 
14. Clone <https://github.com/LukiBa/zybo_face.git> and cd into it
15. cd to intuitus.ko location and insert device driver using ``sudo insmod intuitus.ko``
16. Run application using ``python zybo/application.py``

## Usage Zybo applicaton
TODO: ADD FACE-COMMADS FOLDER

## Related Projects: 
| Project | link |
| ------ | ------ |
| Intuitus Model Converter | <https://github.com/LukiBa/Intuitus-converter.git> |
| Intuitus Interface | https://github.com/LukiBa/Intuitus-intf.git |
| Intuitus device driver | link to kernel module comming soon (contact author for sources) |
| Vivado Yolov3-tiny example project | https://github.com/LukiBa/zybo_yolo_vivado.git |
| Intuitus FPGA IP | encrypted trial version comming soon (contact author) |

## Author
Lukas Baischer   
lukas_baischer@gmx.at
