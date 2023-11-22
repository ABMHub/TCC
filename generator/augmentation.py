import numpy as np
from util.video import loaders
from generator.align_processing import Align
import random

class Augmentation():
  def __init__(self):
    self.name = "Empty"
  
  def __call__(self, video, align, **kwargs):
    raise NotImplementedError
  
class MirrorAug(Augmentation):
  def __init__(self):
    self.name = "Horizontal Flip"

  def __call__(self, video, align, **kwargs):
    reverse = bool(random.getrandbits(1))
    if reverse is True:
      video = np.flip(video, axis=2)

    return align, video
  
class JitterAug(Augmentation):
  def __init__(self):
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

    extra_frames = len(frames) - 75
    if extra_frames > 0:
      frames = frames[0:75]

    return frames

  def __call__(self, video, align, **kwargs):
    jit = self.generate_jitter(len(video))
    return align, video[jit]
  
class SingleWords(Augmentation):
  def __init__(self):
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
  
class HalfFrame(Augmentation):
  def __init__(self):
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
  
class VideoGenerator:
  def __init__(self, 
               augs : list[Augmentation],
               training : bool,
               mean : float,
               std : float,

               landmark_mean = None,
               landmark_std = None,
               
               half_frame : bool = False):
    
    self.augs = augs
    self.info = None
    self.training = training
    self.mean, self.std = mean, std

    self.lm_mean, self.lm_std = landmark_mean, landmark_std
    self.half_frame = HalfFrame()
    if half_frame is False: self.half_frame = lambda a: a

  @property
  def aug_name(self):
    if len(self.augs) == 0:
      return None
    
    return ", ".join([aug.name for aug in self.augs])

  def load_video(self,
                 video_path,
                 align, epoch,
                 standardize = True):
    extension = video_path.split(".")[-1]
    video_loader = loaders[extension]

    y = None

    video = video_loader(video_path)
    _, video = self.half_frame(video, align)
    if self.training is True:
      for aug in self.augs:
        align, video = aug(video, align, epoch=epoch)

    y = align.number_string

    x = np.array(video)
    if standardize:
      x = (x - self.mean)/self.std

    pad_size = 75 - x.shape[0]
    x = np.pad(x, [(0, pad_size), (0, 0), (0, 0), (0, 0)], "constant", constant_values=0)

    return x, y

  def load_landmark(self, landmark_path = None,): # as landmarks precisam sofrer augmentation tambem. provavelmente calcular angulos durante o treino
    assert landmark_path is not None and self.lm_mean is not None, "Landmark feature load error"

    extension = landmark_path.split(".")[-1]
    loader = loaders[extension]

    return (loader(landmark_path) - self.lm_mean)/self.lm_std
