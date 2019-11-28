import schedule
import time
from flask.logging import create_logger

class Logger:
  def __init__(self, flask_app):
    self.LOG = create_logger(flask_app)
    self.job = None

  def generateLog(self, message):
    if message == []:
      return
    self.LOG.warning(message)

  def start(self, message):
    self.generateLog(message)
    self.job = schedule.every().second.do(self.generateLog, message)

  def stop(self):
    if self.job is not None:
      schedule.cancel_job(self.job)

