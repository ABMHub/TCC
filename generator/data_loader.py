import os
import json
import random
import numpy as np
from generator.batch_generator import BatchGenerator

RANDOM_SEED = 42

class DataConfig:
  def __init__(self, data_dict : dict, split : dict, mode : str, videos_path : str, align_path : str) -> None:
    self.mean, self.std = None, None
    self.mode = mode

    if data_dict is not None and self.mode in data_dict:
      self.mean = np.array(data_dict[mode]["mean"])
      self.std = np.array(data_dict[mode]["std"])

    train_split, val_split = split[mode]["train"], split[mode]["test"]
    videos = [[os.path.join(videos_path, elem + ".npz") for elem in file_set] for file_set in [train_split, val_split]]
    aligns = [[os.path.join(align_path, elem + ".align") for elem in file_set] for file_set in [train_split, val_split]]

    self.train = (videos[0], aligns[0])
    self.test = (videos[1], aligns[1])

  @staticmethod
  def save_config(mean : np.ndarray, std : np.ndarray, mode : str, file_path : str) -> None:
    json_dict = {
      "mean": list(mean),
      "std": list(std),
    }

    if os.path.isfile(file_path):
      f = open(file_path, "r")
      config_dict = json.load(f)
      f.close()

    else:
      config_dict = {}

    config_dict[mode] = json_dict
    f = open(file_path, "w")
    json.dump(config_dict, f)
    f.close()

def get_training_data(videos_path : str, align_path : str, batch_size = 1, validation_only = False, unseen_speakers = False):
  config_file = os.path.join(videos_path, "config.json")
  config_exists = os.path.isfile(config_file)

  split_file = "./splits.json"
  split_exists = os.path.isfile(split_file)
  config = None

  mode = "unseen" if unseen_speakers else "overlapped"

  if config_exists:
    f = open(config_file, "r")
    config = json.load(f)
    f.close()
        
  assert split_exists, "Split file not found"

  f = open(split_file, "r")
  split_dict = json.load(f)
  f.close()

  dataconfig = DataConfig(config, split_dict, mode, videos_path, align_path)

  if dataconfig.mean is None:
    train = BatchGenerator(dataconfig.train, batch_size, training=True, mean_and_std=None)
    val = BatchGenerator(dataconfig.test, batch_size, training=False, mean_and_std=(train.mean, train.std_var))

    DataConfig.save_config(train.mean, train.std_var, mode, config_file)

  else:
    train = None if validation_only else BatchGenerator(dataconfig.train, batch_size, training=True, mean_and_std=(dataconfig.mean, dataconfig.std))
    val = BatchGenerator(dataconfig.test, batch_size, training=False, mean_and_std=(dataconfig.mean, dataconfig.std))

  return {"train": train, "validation": val}