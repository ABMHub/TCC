from generator.augmentation import Augmentation
import numpy as np
from util.video import loaders
import os
from generator.align_processing import Align

class VideoGenerator:
  def __init__(self, 
               augs     : list[Augmentation],
               training : bool,
               mean     : list[float],
               std      : list[float],

               apply_padding   : bool         = True,
               standardize     : bool         = True
               ):

    self.augs = augs

    self.info = None
    self.training = training
    self.mean, self.std = mean, std

    # self.post_processing = post_processing or Augmentation()
    self.n_frames = 75
    self.apply_padding = apply_padding
    self.standardize = standardize

    # if post_processing and post_processing.name == "frame sampler":
      # self.n_frames = ceil(self.n_frames / post_processing.rate)

  @property
  def aug_name(self):
    if len(self.augs) == 0:
      return None
    
    return ", ".join([aug.name for aug in self.augs])
  
  # @property
  # def post_name(self):
  #   if isinstance(self.post_processing, Augmentation):
  #     return self.post_processing.name
    
  #   return None

  def load_data(self,
                video_path  : os.PathLike,
              ) -> np.ndarray:
    """Loads data from given path

    Args:
        video_path (os.PathLike): data path

    Returns:
        ndarray: data numpy array
    """

    extension = video_path.split(".")[-1]
    video_loader = loaders[extension]

    return video_loader(video_path)

  def augment_data(
      self,
      data        : tuple[np.ndarray],
      align       : Align, 
      epoch       : int  = 0,
      standardize : bool = None
  ):
    y = None

    if self.augs is not None:
      for aug in self.augs:
        if self.training or aug.on_test:
          data, align = aug(data, align, epoch=epoch)

    if align is not None:
      y = align.number_string

    x = []
    for i, modal in enumerate(data):
      xt = np.array(modal)
      if standardize or (standardize is None and self.standardize):
        xt = (xt - self.mean[i])/self.std[i]

      if self.apply_padding:
        pad_size = self.n_frames - xt.shape[0]
        padding = [(0, pad_size)]
        for _ in range(len(xt.shape)-1):
          padding.append((0, 0))

        xt = np.pad(xt, padding, "constant", constant_values=0)

      x.append(xt)

    return x, y
