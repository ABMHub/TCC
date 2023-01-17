import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from tensorflow.keras.utils import Sequence
import numpy as np
from util.video import read_video
from preprocessing.align_processing import process_file, add_padding

class BatchGenerator(Sequence):

  def __init__(self, data : tuple, batch_size : int) -> None:
    super().__init__()
    self.data = data
    self.batch_size = batch_size
    self.data_number = len(data)
    self.generator_steps = int(np.ceil(self.data_number / self.batch_size))

  def __len__(self) -> int:
    return self.generator_steps

  def __getitem__(self, index : int):
    split_start = index * self.batch_size
    split_end   = split_start + self.batch_size

    if split_end > self.data_number:
      split_end = self.data_number

    batch_videos = self.data[split_start:split_end]

    x, y = [], []
    max_y_size = 0

    for elem in batch_videos:
      x.append(np.load(elem[0]))
      y.append(process_file(elem[1]))
      max_y_size = max(max_y_size, len(y[-1]))

    y = [add_padding(elem, max_y_size) for elem in y]

    return np.array(x), np.array(y)
    
