import tensorflow as tf
import numpy as np
from preprocessing.align_processing import Align
from util.video import loaders
import tqdm
import random

RANDOM_SEED = 42

class VideoData:
  def __init__(self, video_path : str, align : Align, training : bool, mean : float, std : float):
    self.video_path = video_path
    self.info = None
    self.align = align
    self.training = training
    self.mean, self.std = mean, std
    self.reversed = bool(random.getrandbits(1))
    self.subsentences = False
    self.jitter = True

  def load_video(self, epoch):
    extension = self.video_path.split(".")[-1]
    video_loader = loaders[extension]

    y = None
    mask = None
    jitter = None

    video = video_loader(self.video_path)
    if self.training is True:
      if self.subsentences is True:
        mask, y = self.generate_subsentence_mask(epoch)

      elif self.jitter is True:
        jitter = self.generate_jitter()
        y = self.align.number_string

      else:
        y = self.align.number_string

      if self.reversed is True:
        video = np.flip(video, axis=2)

    else:
      y = self.align.number_string

    x = (np.array(video) - self.mean)/self.std

    assert mask is None or jitter is None, "Proibido jitter e subsentence ao mesmo tempo!"

    if mask is not None:
      x = x * mask

    if jitter is not None:
      x = x[jitter]
      pad_size = 75 - x.shape[0]
      x = np.pad(x, [(0, pad_size), (0, 0), (0, 0), (0, 0)], "constant", constant_values=0)

    return x, y

  def generate_subsentence_mask(self, epoch):
    # number_of_sentences = 6
    # interval_size = 0.3

    # mean = epoch * interval_size
    # size = int(random.normalvariate(mean, 1))
    # size = max(size, 1)
    # subsentence_idx = random.randint(0, (number_of_sentences)-size) if size < 6 else 0
    subsentence_idx = random.randint(0, 5)

    begin, end, y = self.align.get_sub_sentence(subsentence_idx, 1)

    mask = np.array([0]*begin + [1]*((end-begin)+1) + [0]*(74-end)).reshape(75, 1, 1, 1)
    return mask, y

  def generate_jitter(self, timesteps : int = 75) -> list[int]:
    frames = list(range(timesteps))
    j = 0
    while j < len(frames):
      rand_num = random.random()
      if rand_num < 0.025:
        frames.insert(j, frames[j])
        j += 1
      elif rand_num >= 0.975:
        del frames[j]
        j -= 1
      j += 1

    extra_frames = len(frames) - 75
    if extra_frames > 0:
      frames = frames[0:75]

    return frames
  
class BatchGenerator(tf.keras.utils.Sequence):
  def __init__(self, data : tuple, batch_size : int, training : bool, mean_and_std : tuple[np.ndarray, np.ndarray] = None) -> None:
    super().__init__()

    random.seed(42)

    self.video_paths = data[0]

    self.epoch = 0

    self.batch_size = batch_size
    self.training = training

    self.aligns = [Align(elem) for elem in tqdm.tqdm(data[1], desc=f"{'Treinamento' if self.training else 'Validação'}: carregando os alinhamentos")]
    self.data = []

    for i in range(len(self.video_paths)):
      self.data.append((self.video_paths[i], self.aligns[i]))

    self.generator_steps = int(np.ceil(len(self.data) / self.batch_size))

    if mean_and_std is None:
      self.mean, self.std_var = np.zeros(3), np.ones(3)
      self.mean, self.std_var = self.__get_std_params()

    else:
      self.mean, self.std_var = mean_and_std

  def get_strings(self):
    return [" ".join(elem.sentence) for elem in self.aligns]

  def on_epoch_end(self):
    self.epoch += 1

  def __get_std_params(self):
    pbar = tqdm.tqdm(desc='Calculando media e desvio padrão', total=len(self.video_paths)*2, disable=False)
    videos_mean = []
    for i in range(len(self.video_paths)):
      data = VideoData(self.video_paths[i], self.aligns[i], False, 0, 1).load_video(None)[0]
      videos_mean.append(np.mean(data, axis=(0, 1, 2)))
      pbar.update()

    dataset_mean = np.mean(videos_mean, axis=0) # a media nao esta exata. o ultimo batch eh menor
    dataset_std = np.zeros(3)
    print(dataset_mean)

    for i in range(len(self.video_paths)):
      data = VideoData(self.video_paths[i], self.aligns[i], False, 0, 1).load_video(None)[0]
      dataset_std += np.sum(np.square(data - dataset_mean), axis=(0, 1, 2))
      pbar.update()

    dataset_std = np.sqrt(dataset_std/(len(self.video_paths)*75*50*100))
    pbar.close()

    print(f"Média: {dataset_mean}\nDesvio Padrão: {dataset_std}")

    return dataset_mean, dataset_std    

  def __get_split_tuple(self, index : int):
    split_start = index * self.batch_size
    split_end   = split_start + self.batch_size

    if split_end > len(self.data):
      split_end = len(self.data)

    return split_start, split_end

  def __len__(self) -> int:
    return self.generator_steps

  def __getitem__(self, index : int):
    split = self.__get_split_tuple(index)

    batch_videos = self.data[split[0]:split[1]]
    x = []
    y = []
    max_y_size = 0
    for vid in batch_videos:
      vid_obj = VideoData(vid[0], vid[1], self.training, self.mean, self.std_var)

      xp, yp = vid_obj.load_video(self.epoch)
      x.append(xp)
      y.append(yp)
      max_y_size = max(max_y_size, len(yp))
  
    x = np.array(x)
    y = np.array([Align.add_padding(elem, max_y_size) for elem in y])
    return x, y