from preprocessing.align_processing import process_file, add_padding
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
import tqdm
import random
from generator.batch_generator import BatchGenerator

def get_training_data(videos_path : str, align_path : str, batch_size = 1, val_size : float = 0.2, n_videos = None):
  file_names = os.listdir(videos_path)
  videos = [os.path.join(videos_path, elem) for elem in file_names]
  aligns = [os.path.join(align_path, ".".join(elem.split(".")[:-1]) + ".align") for elem in file_names] # get align files path list

  data = list(zip(videos, aligns))
  num_videos = len(data)
  val_num = int(num_videos*(val_size))
  random.shuffle(data)

  val_data = data[:val_num]
  train_data = data[val_num:]

  train = BatchGenerator(train_data, batch_size)
  val = None if val_size is None else BatchGenerator(val_data, batch_size)

  return {"train": train, "validation": val}
  

  

