import os
import json
import random
import numpy as np
from generator.batch_generator import BatchGenerator

RANDOM_SEED = 42

class DataConfig:
  def __init__(self, data_dict : dict, split : dict, mode : str, videos_path : str, align_path : str, landmark_features : dict = None) -> None:
    self.mean, self.std = None, None
    self.mode = mode
    self.lm = landmark_features is not None

    if data_dict is not None and self.mode in data_dict:
      self.mean = np.array(data_dict[mode]["mean"])
      self.std = np.array(data_dict[mode]["std"])

    train_split, val_split = split[mode]["train"], split[mode]["test"]
    videos = [[os.path.join(videos_path, elem + ".npz") for elem in file_set] for file_set in [train_split, val_split]]
    aligns = [[os.path.join(align_path, elem + ".align") for elem in file_set] for file_set in [train_split, val_split]]

    self.train = (videos[0], aligns[0])
    self.test = (videos[1], aligns[1])

    print(landmark_features)
    if landmark_features is not None:
      lms_train = [p.replace("npz_mouths", "landmark_features") for p in videos[0]]
      lms_test = [p.replace("npz_mouths", "landmark_features") for p in videos[1]]

      self.train = self.train + (lms_train,)
      self.test = self.test + (lms_test,)

      self.lm_mean, self.lm_std = None, None

      if self.mode in landmark_features:
        self.lm_mean, self.lm_std = landmark_features[mode]["mean"], landmark_features[mode]["std"]

  @staticmethod
  def save_config(mean : np.ndarray, std : np.ndarray, mode : str, file_path : str) -> None:
    if not isinstance(mean, np.ndarray): 
      json_dict = {
        "mean": mean,
        "std": std,
      }
    else:
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

def get_training_data(videos_path : str, align_path : str, batch_size = 1, validation_only = False, unseen_speakers = False, landmark_features : bool = False):
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

  landmark_dict = None
  if landmark_features:
    print(config_file)
    landmark_config_file = config_file.replace("npz_mouths", "landmark_features")
    if os.path.isfile(landmark_config_file):
      f = open(landmark_config_file, "r")
      landmark_dict = json.load(f)
      f.close()
    else:
      landmark_dict = {}

  dataconfig = DataConfig(config, split_dict, mode, videos_path, align_path, landmark_dict)

  train = None
  if not validation_only or dataconfig.mean is None or (dataconfig.lm and dataconfig.lm_mean is None):
    train = BatchGenerator(dataconfig, batch_size, training=True)

  dataconfig.mean, dataconfig.std = train.mean, train.std_var
  DataConfig.save_config(train.mean, train.std_var, mode, config_file)

  if dataconfig.lm:
    dataconfig.lm_mean, dataconfig.lm_std = train.lm_mean, train.lm_std_var
    DataConfig.save_config(train.lm_mean, train.lm_std_var, mode, landmark_config_file)

  val = BatchGenerator(dataconfig, batch_size, training=False)

  return {"train": train, "validation": val}