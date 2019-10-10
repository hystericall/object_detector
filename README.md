# object_detector
A work form pyimagesearch blog
# Installation
*Require python 3.6*

sudo apt-get install python3-pip # Install pip
pip3 install opencv-contrib-python numpy imutils # Install relevant packages

# Running
python3 deep_learning_object_detection.py --image images/example_01.jpg --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel
python3 real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel
