import imutils
import time
import cv2
# python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel
# rtsp://admin:1234qwer@192.168.1.4:554/onvif1
class ObjectDetector:
  def __init__(self, model='frozen_inference_graph_v2_40000.pb',
    graph='output_v2.pbtxt'):
    self.net = cv2.dnn.readNetFromTensorflow(model, graph)

  def detect(self, blob):
    self.net.setInput(blob)
    detections = self.net.forward()
    return detections

