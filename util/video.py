import cv2
import numpy as np
import os

def read_video(path):
  vid = cv2.VideoCapture(path)
  frames = []
  while vid.isOpened():
    _, frame = vid.read()
    if frame is None:
      break
    frames.append(np.transpose(frame.tolist(), (1, 0, 2)))

  return frames

def get_extension(path : str):
  return path.split(".")[-1]

def get_file_name(path : str):
  return ".".join(path.split("/")[-1].split(".")[:-1])

def get_file_path(path : str):
  return os.path.join(path.split("/")[:-1])