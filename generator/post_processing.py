from generator.augmentation import Augmentation
import numpy as np
import random

class HalfFrame(Augmentation):
  def __init__(self, **kwargs):
    self.name = "half frame"
    self.chance = 0.5

  def __call__(self, video, align, **kwargs):
    dice = random.random()
    dim = video.shape[2]
    if dice < self.chance:
      video = np.array(video[:,:,0:dim//2,:])

    else:
      video = np.flip(video[:,:,dim//2:dim,:], axis=2)

    return align, video

class FrameSampler(Augmentation):
  def __init__(self, rate = 2, **kwargs):
    self.name = "frame sampler"
    self.rate = rate
    self.indexes = []

    i = 0

    while i < 75:
      self.indexes.append(int(i))
      i += self.rate

    self.indexes = np.array(self.indexes)

  def __call__(self, video, align, **kwargs):
    return align, video[self.indexes]
  
class MouthOnly(Augmentation):
  def __init__(self, **kwargs):
    self.name = "mouth only"

  def __call__(self, video, align, **kwargs):
    return align, video[:, 48:60]
  
class MouthOnlyCentroid(Augmentation):
  def __init__(self, **kwargs):
    self.name = "mouth only"

  def __call__(self, video, align, **kwargs):
    mouth = video[:, 48:60]
    centroid = mouth.mean(axis=1)
    coords = mouth-np.expand_dims(centroid, 1)
    mx = np.abs(coords).max()

    return align, coords/mx