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
    dx[1:-1] = x[2:] - x[0:-2]; dx[-1] = dx[-2]; dx[0] = dx[1]
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