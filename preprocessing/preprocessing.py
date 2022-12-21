from preprocessing.align_processing import process_file, add_padding
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split

def get_training_data(videos_path : str, align_path : str, test_size, n_videos = None):
  x, y = [], []
  folder = os.listdir(videos_path)
  i = 0
  max_y_size = 0
  for elem in folder:
    vid = cv2.VideoCapture(os.path.join(videos_path, elem))
    frames = []
    while vid.isOpened():
      _, frame = vid.read()
      if frame is None:
        break
      frames.append(np.transpose(frame.tolist(), (1, 0, 2)))

    file_name = ".".join(elem.split(".")[:-1])
    char_string = process_file(os.path.join(align_path, file_name + ".align"))

    max_y_size = max(max_y_size, len(char_string))

    x.append(frames)
    y.append(char_string)

    i += 1
    if n_videos is not None and i >= n_videos: break

  y = [add_padding(elem, max_y_size) for elem in y]

  x = np.array(x)
  y = np.array(y)
  if test_size is not None:
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    return {"train": (X_train, y_train), "test": (X_test, y_test)}

  return {"train": (x, y), "test": None}
  

  

