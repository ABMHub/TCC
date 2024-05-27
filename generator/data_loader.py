import os
import json
import random
import numpy as np

from generator.augmentation import Augmentation
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
    videos_set = set([elem.split(".")[0] for elem in os.listdir(videos_path)])

    train_split = list(videos_set.intersection(train_split))
    val_split   = list(videos_set.intersection(val_split))

    random.seed(42)
    random.shuffle(train_split)
    
    videos = [[os.path.join(videos_path, elem + ".npz") for elem in file_set] for file_set in [train_split, val_split]]
    aligns = [[os.path.join(align_path, elem + ".align") for elem in file_set] for file_set in [train_split, val_split]]

    self.train = (videos[0], aligns[0])
    self.test = (videos[1], aligns[1])

    if self.lm:
      lms_train = [p.replace("npz_mouths", "landmark_features") for p in videos[0]]
      lms_test = [p.replace("npz_mouths", "landmark_features") for p in videos[1]]

      self.train = self.train + (lms_train,)
      self.test = self.test + (lms_test,)

      self.lm_mean, self.lm_std = None, None

      for i, elem in enumerate(lms_train):
        if not os.path.isfile(elem):
          del self.train[0][i]
          del self.train[1][i]
          del self.train[2][i]

      for i, elem in enumerate(lms_test):
        if not os.path.isfile(elem):
          del self.test[0][i]
          del self.test[1][i]
          del self.test[2][i]

      

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

def get_training_data(videos_path       : str, 
                      align_path        : str, 
                      batch_size        : int  = 1, 
                      validation_only   : bool = False, 
                      unseen_speakers   : bool = False, 
                      landmark_features : bool = False, 
                      post_processing   : Augmentation = None,
                      augmentation      : list[Augmentation] = None,
                      is_time_series    : bool = False,
                      standardize       : bool = True,
                      ) -> dict[str, BatchGenerator]:
  
  config_file = os.path.join(videos_path, "config.json")
  config_exists = os.path.isfile(config_file)

  split_file = os.path.join(os.path.dirname(__file__), "..", "splits.json")
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
    train = BatchGenerator(
      config          = dataconfig, 
      batch_size      = batch_size, 
      training        = True, 
      post_processing = post_processing, 
      augmentation    = augmentation, 
      is_time_series  = is_time_series,
      standardize     = standardize, 
      )

  dataconfig.mean, dataconfig.std = train.mean, train.std_var
  DataConfig.save_config(train.mean, train.std_var, mode, config_file)

  if dataconfig.lm:
    dataconfig.lm_mean, dataconfig.lm_std = train.lm_mean, train.lm_std_var
    DataConfig.save_config(train.lm_mean, train.lm_std_var, mode, landmark_config_file)

  val = BatchGenerator(
    config          = dataconfig, 
    batch_size      = batch_size, 
    training        = False, 
    post_processing = post_processing, 
    augmentation    = [], 
    is_time_series  = is_time_series,
    standardize     = standardize, 
    )
  
  test_len = len(dataconfig.test[0])
  random.seed(42)
  sample = random.sample(range(test_len), max(test_len//10, 1))
  dataconfig.test = [np.array(dataconfig.test[i])[sample] for i in range(len(dataconfig.test))]

  wer_val = BatchGenerator(
    config          = dataconfig, 
    batch_size      = batch_size, 
    training        = False, 
    post_processing = post_processing, 
    augmentation    = [], 
    is_time_series  = is_time_series,
    standardize     = standardize, 
    )

  return {"train": train, "validation": val, "wer_validation": wer_val}