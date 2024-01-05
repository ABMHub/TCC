from generator.augmentation import Augmentation
import numpy as np
from util.video import loaders
from math import ceil

class VideoGenerator:
  def __init__(self, 
               augs : list[Augmentation],
               training : bool,
               mean : float,
               std : float,

               landmark_mean = None,
               landmark_std = None,
               
               post_processing : Augmentation = None,
               apply_padding   : bool         = True,
               standardize     : bool         = True
               ):
    
    self.augs = augs
    self.info = None
    self.training = training
    self.mean, self.std = mean, std

    self.lm_mean, self.lm_std = landmark_mean, landmark_std
    self.post_processing = post_processing or Augmentation()
    self.n_frames = 75
    self.apply_padding = apply_padding
    self.standardize = standardize

    if post_processing and post_processing.name == "frame sampler":
      self.n_frames = ceil(self.n_frames / post_processing.rate)

  @property
  def aug_name(self):
    if len(self.augs) == 0:
      return None
    
    return ", ".join([aug.name for aug in self.augs])
  
  @property
  def post_name(self):
    if isinstance(self.post_processing, Augmentation):
      return self.post_processing.name
    
    return None

  def load_video(self,
                 video_path,
                 align, epoch,
                 standardize = None):
    extension = video_path.split(".")[-1]
    video_loader = loaders[extension]

    y = None

    video = video_loader(video_path)
    _, video = self.post_processing(video, align)
    if self.training is True and self.augs is not None:
      for aug in self.augs:
        align, video = aug(video, align, epoch=epoch)

    y = align.number_string

    x = np.array(video)
    standardize = self.standardize
    if standardize or (standardize is None and self.standardize):
      x = (x - self.mean)/self.std

    if self.apply_padding:
      pad_size = self.n_frames - x.shape[0]
      x = np.pad(x, [(0, pad_size), (0, 0), (0, 0), (0, 0)], "constant", constant_values=0)

    return x, y

  def load_landmark(self, landmark_path = None,): # as landmarks precisam sofrer augmentation tambem. provavelmente calcular angulos durante o treino
    assert landmark_path is not None and self.lm_mean is not None, "Landmark feature load error"

    extension = landmark_path.split(".")[-1]
    loader = loaders[extension]

    return (loader(landmark_path) - self.lm_mean)/self.lm_std