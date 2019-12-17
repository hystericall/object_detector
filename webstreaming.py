# USAGE
# python webstreaming.py --ip 0.0.0.0 --port 8000

# import the necessary packages
from back_end.object_detection import ObjectDetector
# from back_end.logger import Logger
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
from flask import request
import glob
import logging.config
import yaml
import schedule
import os
import numpy as np
import threading
import argparse
import imutils
import time
import cv2
# initialize the output frame, log message and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing tthe stream)
mes = []
job = None
outputFrame = None
lock = threading.Lock()

# initialize a flask object, flask log config
app = Flask(__name__)

# log = create_logger(app)
# logging.basicConfig(filename='logs/app.log', filemode='a',
#                     format='[%(asctime)s] %(levelname)s: %(message)s',
#                     datefmt='%d-%b-%y %H:%M:%S',
#                     level=logging.CRITICAL)
logging.config.dictConfig(yaml.load(open('logging.conf')))
logfile = logging.getLogger('file')


# initialize enviroment variable
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "nguoi", "xe may", "o to"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
UPLOAD_FOLDER = 'static/tmp/'
# initialize the video stream and allow the camera sensor to
# warmup
# vs = cv2.VideoCapture('rtsp://admin:1234qwer@192.168.1.5:554/onvif1')
# vs = cv2.VideoCapture("videofromcam.mp4")
time.sleep(2.0)


def detect_from_image(frame, detector, confidence_score=0.5):
  frame = imutils.resize(frame, width=400)
  (h, w) = frame.shape[:2]
  blob = cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False)
  detections = detector.detect(blob)
  for i in np.arange(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > confidence_score:
      idx = int(detections[0, 0, i, 1])
      box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
      (startX, startY, endX, endY) = box.astype("int")
      label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
      print("[INFO] {}".format(label))
      cv2.rectangle(frame, (startX, startY), (endX, endY),
                    COLORS[idx], 2)
      y = startY - 15 if startY - 15 > 15 else startY + 15
      cv2.putText(frame, label, (startX, y),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
  return frame

def detect_object(confidence, target):
  # grab global references to the video stream, output frame, and
  # lock variables
  global vs, outputFrame, lock, mes

  # initialize the detector and the total number of frames
  # read thus far
  od = ObjectDetector()
  detectedMes = []
  # loop over frames from the video stream
  while True:
    # read the next frame from the video stream, resize it,
    # get the frame dimension and convert to a blob
    ret, frame = vs.read()
    frame = imutils.resize(frame, width=1000)
    # grab the frame dimensions and convert it to a blob as opencv use
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
          if target is not None:
            if target not in CLASSES[idx]:
              continue
          box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
          (startX, startY, endX, endY) = box.astype("int")

          # draw the prediction on the frame
          label = "{}: {:.2f}%".format(CLASSES[idx],
                                      confidence * 100)
          detectedMes.append(label)
          cv2.rectangle(frame, (startX, startY), (endX, endY),
                                  COLORS[idx], 2)
          y = startY - 15 if startY - 15 > 15 else startY + 15
          cv2.putText(frame, label, (startX, y),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    # acquire the lock, set the output frame, and release the
    # lock
    with lock:
      outputFrame = frame.copy()
      mes = detectedMes.copy()
      detectedMes = []

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

def generateLog():
  global mes
  if mes == []:
    return
  logfile.critical(mes)

def start():
  global job
  generateLog()
  job = schedule.every().second.do(generateLog)

def stop():
  global job
  if job is not None:
    schedule.cancel_job(job)

def run_schedule():
  while True:
    schedule.run_pending()
    time.sleep(1)

@app.route("/")
def index():
  # return the rendered template
  return render_template("index.html")

@app.route("/video_feed")
def video_feed():
  # return the response generated along with the specific media
  # type (mime type)
  return Response(generate(),
    mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route('/upload/', methods=['GET', 'POST'])
def upload():
  if request.method == 'POST':
    uploaded_files = request.files.getlist("file[]")
    tmp_files = glob.glob(UPLOAD_FOLDER + '*')
    if tmp_files != []:
      for f in tmp_files:
          os.remove(f)
    for file in uploaded_files:
      file.save(os.getcwd() + "/" + UPLOAD_FOLDER + file.filename)
    fileNames = os.listdir(UPLOAD_FOLDER)
    od = ObjectDetector()
    for fileName in  fileNames:
      image = cv2.imread(UPLOAD_FOLDER + fileName)
      image = detect_from_image(frame=image, detector=od)
      cv2.imwrite(UPLOAD_FOLDER + fileName, image)
    return render_template('image_slide.html', names = fileNames)
  return render_template('upload.html')

@app.route('/result/')
def result():
  fileNames = os.listdir(UPLOAD_FOLDER)
  return render_template('image_slide.html', names = fileNames )

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
  ap.add_argument("-t", "--target", type=str, default= None,
                  help="which object to detect")
  ap.add_argument("-s", "--source", type=str,
                  default="videofromcam.mp4",
                  help="rstp video url including username(admin) and password")
  args = vars(ap.parse_args())

  vs = cv2.VideoCapture(args["source"])
  # vs = cv2.VideoCapture(0)
  # start a thread that will perform motion detection
  t = threading.Thread(target=detect_object, args=(
      args["confidence"], args["target"]))
  t.daemon = True
  t.start()

  schedule.every().day.at("23:00").do(start)
  schedule.every().day.at("05:00").do(stop)
  t1 = threading.Thread(target=run_schedule)
  t1.daemon = True
  t1.start()
  # start the flask app
  app.run(host=args["ip"], port=args["port"], debug=True,
    threaded=True, use_reloader=True)

# release the video stream pointer
vs.stop()
