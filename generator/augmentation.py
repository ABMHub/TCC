import numpy as np
from util.video import loaders
from generator.align_processing import Align
import random

class Augmentation():
  def __init__(self, **kwargs):
    self.name = "Empty"
  
  def __call__(self, video, align, **kwargs):
    return align, video
  
class MirrorAug(Augmentation):
  def __init__(self, **kwargs):
    self.name = "Horizontal Flip"

  def __call__(self, video, align, **kwargs):
    reverse = bool(random.getrandbits(1))
    if reverse is True:
      video = np.flip(video, axis=2)

    return align, video
  
class JitterAug(Augmentation):
  def __init__(self, **kwargs):
    self.name = "Jittering"
    self.chance = 0.05

  def generate_jitter(self, timesteps : int = 75) -> list[int]:
    frames = list(range(timesteps))
    j = 0
    while j < len(frames):
      rand_num = random.random()
      if rand_num < self.chance/2:
        frames.insert(j, frames[j])
        j += 1
      elif rand_num >= 1 - (self.chance/2):
        del frames[j]
        j -= 1
      j += 1

    extra_frames = len(frames) - timesteps
    if extra_frames > 0:
      frames = frames[0:timesteps]

    return frames

  def __call__(self, video, align, **kwargs):
    jit = self.generate_jitter(len(video))
    return align, video[jit]
  
class SingleWords(Augmentation):
  def __init__(self, **kwargs):
    self.name = "single words"
    self.chance = 0.2

  def generate_subsentence_mask(self, align : Align):
    subsentence_idx = random.randint(0, 5)

    return align.get_sub_sentence(subsentence_idx, 1)

  def __call__(self, video, align, **kwargs):
    dice = random.random()
    if dice < self.chance:
      begin, end, y = self.generate_subsentence_mask(align)
      return y, video[begin:end+1]

    return align, video
    