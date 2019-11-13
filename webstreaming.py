# USAGE
# python webstreaming.py --ip 0.0.0.0 --port 8000

# import the necessary packages
from back_end.object_detection import ObjectDetector
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import os
import numpy as np
import threading
import argparse
import datetime
import imutils
import time
import cv2
# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing tthe stream)
outputFrame = None
lock = threading.Lock()

# initialize a flask object
app = Flask(__name__)

# initialize enviroment variable
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "nguoi", "xe may", "o to"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
# initialize the video stream and allow the camera sensor to
# warmup
#vs = VideoStream(usePiCamera=1).start()
# vs = cv2.VideoCapture('rtsp://admin:1234admin@192.168.1.7:554/onvif1')
# TODO doc video tu tap anh
vs = cv2.VideoCapture('videofromcam.mp4')
time.sleep(2.0)

@app.route("/")
def index():
  # return the rendered template
  return render_template("index.html")

def detect_object(confidence):
  # grab global references to the video stream, output frame, and
  # lock variables
  global vs, outputFrame, lock

  # initialize the motion detector and the total number of frames
  # read thus far
  od = ObjectDetector()

  # loop over frames from the video stream
  while True:
    # read the next frame from the video stream, resize it,
    # get the frame dimension and convert to a blob
    ret, frame = vs.read()
    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False)

    # detect object in the image
    detections = od.detect(blob)

      # check to see if there are detected objects
    if detections is not None:
      for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
          # extract the index of the class label from the
          # `detections`, then compute the (x, y)-coordinates of
          # the bounding box for the object
          idx = int(detections[0, 0, i, 1])
          box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
          (startX, startY, endX, endY) = box.astype("int")

          # draw the prediction on the frame
          label = "{}: {:.2f}%".format(CLASSES[idx],
                                      confidence * 100)
          cv2.rectangle(frame, (startX, startY), (endX, endY),
                                  COLORS[idx], 2)
          y = startY - 15 if startY - 15 > 15 else startY + 15
          cv2.putText(frame, label, (startX, y),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    # acquire the lock, set the output frame, and release the
    # lock
    with lock:
      outputFrame = frame.copy()

def generate():
  # grab global references to the output frame and lock variables
  global outputFrame, lock

  # loop over frames from the output stream
  while True:
    # wait until the lock is acquired
    with lock:
      # check if the output frame is available, otherwise skip
      # the iteration of the loop
      if outputFrame is None:
        continue

      # encode the frame in JPEG format
      (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

      # ensure the frame was successfully encoded
      if not flag:
        continue

    # yield the output frame in the byte format
    yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
      bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
  # return the response generated along with the specific media
  # type (mime type)
  return Response(generate(),
    mimetype = "multipart/x-mixed-replace; boundary=frame")

# check to see if this is the main thread of execution
if __name__ == '__main__':
  # construct the argument parser and parse command line arguments
  ap = argparse.ArgumentParser()
  ap.add_argument("-i", "--ip", type=str, required=True,
    help="ip address of the device")
  ap.add_argument("-o", "--port", type=int, required=True,
    help="ephemeral port number of the server (1024 to 65535)")
  ap.add_argument("-c", "--confidence", type=float, default=0.3,
    help="the threshold confidence value")
  args = vars(ap.parse_args())

  # start a thread that will perform motion detection
  t = threading.Thread(target=detect_object, args=(
    args["confidence"],))
  t.daemon = True
  t.start()

  # start the flask app
  app.run(host=args["ip"], port=args["port"], debug=True,
    threaded=True, use_reloader=False)

# release the video stream pointer
vs.stop()
