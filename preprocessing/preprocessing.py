import os
import random
from generator.batch_generator import BatchGenerator

def get_training_data(videos_path : str, align_path : str, batch_size = 1, val_size : float = 0.2, n_videos = None, data_augmentation=True, validation_only = False):
  file_names = os.listdir(videos_path)

  random.seed(42)
  random.shuffle(file_names)

  num_videos = len(file_names)
  val_num = int(num_videos*(val_size))

  val_files = file_names[:val_num]
  train_files = file_names[val_num:]

  videos = [[os.path.join(videos_path, elem) for elem in file_set] for file_set in [train_files, val_files]]
  aligns = [[os.path.join(align_path, ".".join(elem.split(".")[:-1]) + ".align") for elem in file_set] for file_set in [train_files, val_files]]  # get align files path list
   
  train = None if validation_only is True           else BatchGenerator((videos[0], aligns[0]), batch_size, augmentation=data_augmentation)
  val =   None if val_size is None or val_size == 0 else BatchGenerator((videos[1], aligns[1]), batch_size, augmentation=False, preserve_strings=True)

  return {"train": train, "validation": val}
  

  

