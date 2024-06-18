import numpy as np
from util.video import loaders
from generator.align_processing import Align
import random
import math

class Augmentation():
  def __init__(self, **kwargs):
    self.name = "Empty"
    self.lm_only = False
    self.mask : list[bool] = None
    self.on_test = False

  def __call__(self, data : list[np.ndarray], align : Align, **kwargs):
    return data, align
  
class MirrorAug(Augmentation):
  def __init__(self, mask : list[bool], lm_mask : list[bool], axis : list[int], **kwargs):
    super().__init__()
    self.name = "Horizontal Flip"
    self.mask = mask
    self.lm_mask = lm_mask
    self.on_test = False
    self.axis = axis
    self.lm_mirror_map = [
      16, 15, 14, 13, 12, 11, 10, 9, # right jaw
      8, 7, 6, 5, 4, 3, 2, 1, 0, # left jaw
      26, 25, 24, 23, 22, 21, 20, 19, 18, 17, # eyebrows
      27, 28, 29, 30, 35, 34, 33, 32, 31, # nose
      45, 44, 43, 42, 47, 46, # right eye
      39, 38, 37, 36, 41, 40, # left eye
      54, 53, 52, 41, 50, 49, 48, # outer upper lip
      59, 58, 57, 56, 55, # outer bottom lip
      64, 63, 62, 61, 60, # inner upper lip
      67, 66, 65, # inner bottom lip
    ]

  def __call__(self, data : list[np.ndarray], align : Align, **kwargs):
    reverse = bool(random.getrandbits(1))
    for i, allow in enumerate(self.mask):
      if allow and data[i] is not None and reverse:
        if self.lm_mask[i]:
          for j in range(len(data[i])): # timesteps
            data[i][j] = data[i][j][self.lm_mirror_map]
        else:
          data[i] = np.flip(data[i], axis=self.axis[i])

    return data, align
  
class JitterAug(Augmentation):
  def __init__(self, mask : list[bool], **kwargs):
    super().__init__()
    self.name = "Jittering"
    self.chance = 0.05
    self.mask = mask
    self.on_test = False

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

  def __call__(self, data : list[np.ndarray], align : Align, **kwargs):
    jit = self.generate_jitter(len(data[0]))
    for i, allow in enumerate(self.mask):
      if allow and data[i] is not None:
        data[i] = data[i][jit]
        
    return data, align
  
# class SingleWords(Augmentation):
#   def __init__(self, mask : list[bool], **kwargs):
#     super().__init__()
#     self.name = "single words"
#     self.chance = 0.2
#     self.mask = mask
#     self.on_test = False

#     # todo adaptar para mask

#   def generate_subsentence_mask(self, align : Align):
#     subsentence_idx = random.randint(0, 5)

#     return align.get_sub_sentence(subsentence_idx, 1)

#   def __call__(self, video, align, **kwargs):
#     dice = random.random()
#     if dice < self.chance:
#       begin, end, y = self.generate_subsentence_mask(align)
#       return y, video[begin:end+1]

#     return align, video
    
class CosineLandmarkFeatures(Augmentation):
  def __init__(self, mask : list[bool]):
    self.face_countour = range(17)
    self.lip_marks     = range(48, 68)
    self.name          = "Cosine"
    self.mask          = mask
    self.on_test       = True

  def __call__(self, data : list[np.ndarray], align : Align, **kwargs):
    for k, allow in enumerate(self.mask):
      if allow and data[k] is not None:
        out = []
        for frame in range(len(data[k])):
          cosines = []
          points = data[k][frame]
          for i in self.face_countour:
            for j in self.lip_marks:
              sub = points[i][0] - points[j][0]
              hip = math.sqrt(sub**2 + (points[i][1] - points[j][1])**2)
              cosines.append(abs(sub)/hip)
          out.append(cosines)
      
        data[k] = np.array(out)
    return data, align
  
class CosineDiff(Augmentation):
  def __init__(self, mask : list[bool]):
    self.name          = "CosineDiff"
    self.mask          = mask
    self.on_test       = True

  def __diff(self, x):
    dx = np.convolve(x, [0.5,0,-0.5], mode='same')
    dx[0] = dx[1]
    dx[-1] = dx[-2]
    return dx

  def __call__(self, data : list[np.ndarray], align : Align, **kwargs):
    for k, allow in enumerate(self.mask):
      if allow and data[k] is not None:
        for frame in range(len(data[k])):
          data[k][frame] = self.__diff(data[k][frame])
    return data, align

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
    return align, video[:, 48:68]
  
class MouthOnlyCentroid():
  def __init__(self, **kwargs):
    self.name = "mouth only"

  def __call__(self, video, align, **kwargs):
    mouth = video[:, 48:68]
    centroid = mouth.mean(axis=0)
    coords = mouth-np.expand_dims(centroid, 0)
    mx = np.abs(coords).max(axis=0)

    return align, coords/mx
  
class MouthJP():
  def __init__(self, **kwargs):
    self.name = "mouth only"

  def _diff(self, x):
    dx = np.convolve(x, [0.5,0,-0.5], mode='same')
    dx[0] = dx[1]
    dx[-1] = dx[-2]
    return dx

  def _diffTheta(self, x):
    dx = np.zeros_like(x)
    dx[1:-1] = x[2:] - x[0:-2];dx[-1] = dx[-2]; dx[0] = dx[1]
    temp = np.where(np.abs(dx)>np.pi)
    dx[temp] -= np.sign(dx[temp]) * 2 * np.pi
    dx *= 0.5
    return dx

  def __call__(self, video, align, **kwargs):
    mouth = video[:, 48:68]
    # centroid = mouth.mean(axis=0)
    # coords = mouth-np.expand_dims(centroid, 0)
    # mx = np.abs(coords).max(axis=0)

    features = []

    for i in range(mouth.shape[1]):
      x = mouth[:, i, 0]
      y = mouth[:, i, 1]

      dx = self._diff(x)
      dy = self._diff(y)
      v = np.sqrt(dx**2+dy**2)
      theta = np.arctan2(dy, dx)
      cos = np.cos(theta)
      sin = np.sin(theta)
      dv = self._diff(v)
      dtheta = np.abs(self._diffTheta(theta))
      logCurRadius = np.log((v+0.05) / (dtheta+0.05))
      dv2 = np.abs(v*dtheta)
      totalAccel = np.sqrt(dv**2 + dv2**2)
      c = v * dtheta

      features.append([dx, dy, v, theta, cos, sin, dv, dtheta, logCurRadius, dv2, totalAccel, c])

    return align, np.array(features).transpose((2, 0, 1))