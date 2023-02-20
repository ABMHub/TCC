import os
import json
import random
from generator.batch_generator import BatchGenerator

RANDOM_SEED = 42

class DataConfig:
  def __init__(self, data_dict : dict) -> None:
    self.random_seed : int = data_dict["random_seed"]
    self.val_size : float = data_dict["val_size"]

    self.mean : float = data_dict["mean"]
    self.std : float = data_dict["std"]

    self.train : tuple[list[str]] = data_dict["train"]["videos"], data_dict["train"]["aligns"]
    self.validation : tuple[list[str]] = data_dict["validation"]["videos"], data_dict["validation"]["aligns"]

  @staticmethod
  def save_config(random_seed : int, val_size : float, mean : float, std : float, train_data : tuple[list[str]], validation_data : tuple[list[str]], file_path : str) -> None:
    json_dict = {
      "random_seed": random_seed,
      "val_size": val_size,
      "mean": mean,
      "std": std,
      "train": {
        "videos": train_data[0],
        "aligns": train_data[1]
      },
      "validation": {
        "videos": validation_data[0],
        "aligns": validation_data[1]
      }
    }

    if os.path.isfile(file_path):
      f = open(file_path, "r")
      config_list = json.load(f)
      f.close()

    else:
      config_list = []

    config_list.append(json_dict)
    f = open(file_path, "w")
    json.dump(config_list, f)
    f.close()

def get_training_data(videos_path : str, align_path : str, batch_size = 1, val_size : float = 0.2, validation_only = False, curriculum_steps=None):
  config_file = os.path.join(videos_path, "config.json")
  config_exists = os.path.isfile(config_file)
  config = None

  if config_exists:
    f = open(config_file, "r")
    config_list = json.load(f)

    for config_dict in config_list:
      if config_dict["random_seed"] == RANDOM_SEED and config_dict["val_size"] == val_size:
        config = DataConfig(config_dict)
        break

    f.close()
        
  if config is None:
    file_names = os.listdir(videos_path)
    if config_exists:
      file_names.remove("config.json")

    random.seed(RANDOM_SEED)
    random.shuffle(file_names)

    num_videos = len(file_names)
    val_num = int(num_videos*(val_size))

    val_files = file_names[:val_num]
    train_files = file_names[val_num:]

    videos = [[os.path.join(videos_path, elem) for elem in file_set] for file_set in [train_files, val_files]]
    aligns = [[os.path.join(align_path, ".".join(elem.split(".")[:-1]) + ".align") for elem in file_set] for file_set in [train_files, val_files]]  # get align files path list

    train = BatchGenerator((videos[0], aligns[0]), batch_size, training=True, curriculum_steps=curriculum_steps, mean_and_std=None)
    val = None if val_size is None or val_size == 0 else BatchGenerator((videos[1], aligns[1]), batch_size, training=False, mean_and_std=(train.mean, train.std_var))

    DataConfig.save_config(RANDOM_SEED, val_size, train.mean, train.std_var, (videos[0], aligns[0]), (videos[1], aligns[1]), config_file)

  else:
    train_data = config.train
    validation_data = config.validation

    train = None if validation_only else BatchGenerator(train_data, batch_size, training=True, curriculum_steps=curriculum_steps, mean_and_std=(config.mean, config.std))
    val = None if val_size is None or val_size == 0 else BatchGenerator(validation_data, batch_size, training=False, mean_and_std=(config.mean, config.std))

  return {"train": train, "validation": val}