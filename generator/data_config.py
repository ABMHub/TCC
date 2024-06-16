import os
import json
import random
import numpy as np

RANDOM_SEED = 42

class DataConfig:
  """Configurations from the dataset

  @properties:
    `self.mode`: `str` = unseen or split
    `self.mean`, `self.std`: list of means and stds of every data stream of the dataset.
      None if not stored
    `self.train`, `self.test`: train and test splits. tuples of x and y
      x has shape of [modal_quantity, data_size]
  """
  def __init__(self,
               data_dict         : list[dict], 
               split             : dict,
               mode              : str, 

               data_paths        : list[os.PathLike], 
               align_path        : os.PathLike,
               on_test_augs      : list[str]
               ) -> None:
    
    self.mode = mode
    self.mean, self.std = [], []
    self.data_paths = data_paths
    self.on_test_augs = on_test_augs
    
    for i, dic in enumerate(data_dict):
      augs = on_test_augs[i]
      if dic is not None and augs is None and self.mode in dic:
        self.mean.append(np.array(dic[mode]["mean"]))
        self.std.append(np.array(dic[mode]["std"]))

      elif dic is not None and augs is not None and augs in dic and self.mode in dic[augs]:
        self.mean.append(np.array(dic[augs][mode]["mean"]))
        self.std.append(np.array(dic[augs][mode]["std"]))

      else:
        self.mean.append(None)
        self.std.append(None)

    train_split, val_split = split[mode]["train"], split[mode]["test"]
    for path in self.data_paths:
      data_set = set([elem.split(".")[0] for elem in os.listdir(path)])

      train_split = list(data_set.intersection(train_split))
      val_split   = list(data_set.intersection(val_split))

    random.seed(42)
    random.shuffle(train_split)
    
    x_data = []
    for file_set in [train_split, val_split]:
      x_split = []
      for root_folder in self.data_paths:
        modal = []
        extension = os.listdir(root_folder)[0].split(".")[-1]
        for elem in file_set:
          modal.append(os.path.join(root_folder, elem + "." + extension))

        x_split.append(modal)

      x_data.append(x_split)

    aligns = [[os.path.join(align_path, elem + ".align") for elem in file_set] for file_set in [train_split, val_split]]

    self.train = (x_data[0], aligns[0])
    self.test = (x_data[1], aligns[1])

  def save_config(self) -> None:
    """Saves configuration in a config.json file

    Saves a different configuration for every data stream

    Every config.json is stored in the dataset root folder

    """
    for i, path in enumerate(self.data_paths):
      file_path = os.path.join(path, "config.json")
      mean = self.mean[i]
      std = self.std[i]

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

      if self.on_test_augs is not None and self.on_test_augs[i] is not None:
        key = self.on_test_augs[i]
        d = dict()
        d[self.mode] = json_dict
      
      else:
        key = self.mode
        d = json_dict

      config_dict[key] = d
      f = open(file_path, "w")
      json.dump(config_dict, f)
      f.close()

  def set_mean_std(self, mean : list[float], std : list[float]) -> None:
    self.mean = mean
    self.std = std

  @property
  def mean_std_avaiable(self) -> bool:
    """Checks if every mean and std is avaible

    Returns:
        bool: `True` if every single mean and std is already calculated. Else, `False`
    """
    s = np.array(self.mean, dtype=bool).sum() + np.array(self.std, dtype=bool).sum()
    return s == len(self.mean) * 2
