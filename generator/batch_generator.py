import os

import tensorflow as tf
import numpy as np
from util.video import read_video
from preprocessing.align_processing import read_file, add_padding, sentence2number

RANDOM_SEED = 42

class BatchGenerator(tf.keras.utils.Sequence):
  def __init__(self, data : tuple, batch_size : int, augmentation : bool = False, preserve_strings : bool = False) -> None:
    super().__init__()

    self.video_loader = self.__init_video_loader(data[0][0])

    self.batch_size = batch_size
    self.data_number = len(data[0])
    self.generator_steps = int(np.ceil(self.data_number / self.batch_size))
    self.augmentation = augmentation

    if augmentation:
      aug_x = np.copy(data[0])
      np.random.seed(RANDOM_SEED)
      np.random.shuffle(aug_x)

      self.batch_size = int(self.batch_size / 2)
      self.generator_steps *= 2

    y_dict = self.__load_y(data[1], preserve_strings=preserve_strings)

    self.strings = y_dict["strings"]

    self.data = list(zip(data[0], y_dict["regular"]))

    self.aug_data = None
    if augmentation:
      self.aug_data = list(zip(aug_x, y_dict["augmentation"]))

  def __init_video_loader(self, file):
    loaders = {
      "npy": np.load,
      "npz": lambda path: np.load(path)["arr_0"],
      # ".avi"
    }
    
    extension = file.split(".")[-1]

    return loaders[extension] # essa decisao pode ser feita para cada video... mas adiciona o custo de tempo do split para achar a extensao

  def __load_y(self, y, preserve_strings : bool = False):
    print("Carregando alinhamentos...")
    read_y = [read_file(elem) for elem in y]
    processed_y = [sentence2number(elem) for elem in read_y]

    read_y = read_y if preserve_strings else None

    aug_y = None

    if self.augmentation:
      aug_y = np.copy(processed_y)
      np.random.seed(RANDOM_SEED)
      np.random.shuffle(aug_y)

    return {"regular": processed_y, "augmentation": aug_y, "strings": read_y}

  def __len__(self) -> int:
    return self.generator_steps

  def __get_split_tuple(self, index : int):
    split_start = index * self.batch_size
    split_end   = split_start + self.batch_size

    if split_end > self.data_number:
      split_end = self.data_number

    return split_start, split_end

  def __getitem__(self, index : int):
    split = self.__get_split_tuple(index)

    batch_videos = self.data[split[0]:split[1]]

    x, y = [], []

    max_y_size = 0
    for elem in batch_videos:
      x.append(self.video_loader(elem[0]))
      y.append(elem[1])
      max_y_size = max(max_y_size, len(y[-1]))
      
    if self.augmentation:
      augment_batch = self.aug_data[split[0]:split[1]]

      for elem in augment_batch:
        x.append(np.flip(self.video_loader(elem[0]), axis=2))
        y.append(elem[1])
        max_y_size = max(max_y_size, len(y[-1]))

    y = [add_padding(elem, max_y_size) for elem in y]    

    return np.array(x)/255, np.array(y) # necessario retirar o /255 e substituir por uma standalizacao
