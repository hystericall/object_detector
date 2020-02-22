import threading
from threading import Lock
import cv2

class Camera:
  last_frame = None
  last_ready = None
  capture = None
  thread = None
  lock = Lock()

  def __init__(self, rtsp_link):
    self.capture = cv2.VideoCapture(rtsp_link, cv2.CAP_FFMPEG)
    self.thread = threading.Thread(target=self.rtsp_cam_buffer,
      name="rtsp_read_thread")
    self.thread.daemon = True
    self.thread.start()

  def rtsp_cam_buffer(self):
    while True:
      with self.lock:
        self.last_ready, self.last_frame = self.capture.read()

  def getFrame(self):
    if (self.last_ready is not None) and (self.last_frame is not None):
      return self.last_frame.copy()
    else:
      return None

  def release(self):
    self.capture.release()
