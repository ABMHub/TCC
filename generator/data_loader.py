import os
import json
import random
import numpy as np

from generator.augmentation import Augmentation
from generator.batch_generator import BatchGenerator
from generator.data_config import DataConfig

def get_training_data(x_path            : list[os.PathLike], 
                      y_path            : os.PathLike, 
                      batch_size        : int  = 1, 
                      validation_only   : bool = False, 
                      unseen_speakers   : bool = False, 
                      augmentation      : list[Augmentation] = None,
                      standardize       : bool = True,
                      ) -> dict[str, BatchGenerator]:
  
  split_file = os.path.join(os.path.dirname(__file__), "..", "splits.json")
  assert os.path.isfile(split_file), "Split file not found"

  f = open(split_file, "r")
  split_dict = json.load(f)
  f.close()

  configs = []
  for modality in x_path:
    config_path = os.path.join(modality, "config.json")
    config_exists = os.path.isfile(config_path)
    config = None

    if config_exists:
      f = open(config_path, "r")
      config = json.load(f)
      f.close()

    configs.append(config)

  mode = "unseen" if unseen_speakers else "overlapped"

  name_on_test = aug_name_on_test(augmentation)

  dataconfig = DataConfig(
    data_dict=configs, 
    split=split_dict, 
    mode=mode, 
    data_paths=x_path, 
    align_path=y_path,
    on_test_augs=name_on_test
  )

  train = None
  if not validation_only or not dataconfig.mean_std_avaible:
    train = BatchGenerator(
      config          = dataconfig, 
      batch_size      = batch_size, 
      training        = True, 
      augmentation    = augmentation, 
      standardize     = standardize, 
      )

  dataconfig.set_mean_std(train.mean, train.std_var)
  dataconfig.save_config()

  val = BatchGenerator(
    config          = dataconfig, 
    batch_size      = batch_size, 
    training        = False, 
    augmentation    = augmentation, 
    standardize     = standardize, 
    )
  
  test_len = len(dataconfig.test[0])
  random.seed(42)
  sample = random.sample(range(test_len), max(test_len//10, 1))
  dataconfig.test = (
    [np.array(dataconfig.test[0][i])[sample] for i in range(len(dataconfig.test[0]))],
    np.array(dataconfig.test[1])[sample]
  )

  wer_val = BatchGenerator(
    config          = dataconfig, 
    batch_size      = batch_size, 
    training        = False, 
    augmentation    = augmentation, 
    standardize     = standardize, 
    )

  return {"train": train, "validation": val, "wer_validation": wer_val}

def aug_name_on_test(augs):
  if augs is None or len(augs) == 0:
    return None
  
  modality_quant = len(augs[0].mask)

  ret = []
  for i in range(modality_quant):
    names = []
    for aug in augs:
      if aug.mask[i] and aug.on_test:
        names.append(aug.name)

    if len(names) == 0:
      ret.append(None)
      
    else:
      ret.append(", ".join(names))

  return ret