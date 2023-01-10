import cv2
import numpy as np

def read_video(path):
  vid = cv2.VideoCapture(path)
  frames = []
  while vid.isOpened():
    _, frame = vid.read()
    if frame is None:
      break
    frames.append(np.transpose(frame.tolist(), (1, 0, 2)))

  return frames