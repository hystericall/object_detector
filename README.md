# object_detector
# Installation
*Require python 3.6*
```
sudo apt-get install python3-pip # Install pip
pip3 install opencv-contrib-python numpy imutils schedule flask # Install relevant packages
```
# Running
Image detector
```
python3 deep_learning_object_detection.py --image images/example_01.jpg --pbtxt output_v2.pbtxt --model frozen_inference_graph_v2_40000.pb
```
Real-time detector
```
python3 real_time_object_detection.py --pbtxt output_v2.pbtxt --model frozen_inference_graph_v2_40000.pb
```

Local website application:
```
python3 webstreaming.py --ip 0.0.0.0 --port 8000
```
Go to 0.0.0.0:8000 to see the live-streaming or 0.0.0.0:8000/upload/ to upload images
