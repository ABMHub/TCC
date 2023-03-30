import math
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
    self.reversed = bool(random.getrandbits(1)) if training is True else False

  def load_video(self, epoch):
    extension = self.video_path.split(".")[-1]
    video_loader = loaders[extension]

    number_of_sentences = 6
    interval_size = 5

    y = None

    video = video_loader(self.video_path)
    if self.training is True:
      size = int(epoch/interval_size)+1
      subsentence_idx = random.randint(0, (number_of_sentences)-size)

      info = self.align.get_sub_sentence(subsentence_idx, size)

      video = video[info[0]:info[1]+1]
      
      if self.reversed is True:
        video = np.flip(video, axis=2)

      y = info[2]

    else:
      y = self.align.number_string

    pad_size = 75 - video.shape[0]
    x = (np.array(video) - self.mean)/self.std
    x = np.pad(video, [(0, pad_size), (0, 0), (0, 0), (0, 0)], "constant", constant_values=0)
    return x, y
  
  def generate_jitter(self, timesteps : int) -> list[int]:
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
      for _ in range(extra_frames):
        choice = random.choice(range(len(frames)))
        del frames[choice]

    return frames
  
class BatchGenerator(tf.keras.utils.Sequence):
  def __init__(self, data : tuple, batch_size : int, training : bool, mean_and_std : tuple[float, float] = None) -> None:
    # assert batch_size > 1 or training is False, "Não é possível fazer augmentation em um batch-size de 1. É necessário batch-size ao menos de 2."
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
      self.mean, self.std_var = 0, 1
      self.mean, self.std_var = self.__get_std_params()

    else:
      self.mean, self.std_var = mean_and_std

    self.jitter = 0
    self.regular = 0
    self.single = 0
    self.reversed = 0

  def get_strings(self):
    return [" ".join(elem.sentence) for elem in self.aligns]

  def on_epoch_end(self):
    self.epoch += 1

    print(f"\nSingle: {self.single}\nRegular: {self.regular}\nJitter: {self.jitter}")
    print(f"Reversed: {self.reversed}/{len(self.data)}")

    self.jitter = 0
    self.regular = 0
    self.single = 0
    self.reversed = 0

  def __get_std_params(self): # standardizacao deve ser por canal de cor
    pbar = tqdm.tqdm(desc='Calculando media e desvio padrão', total=len(self.video_paths)*2, disable=False)
    videos_mean = []
    for i in range(len(self.video_paths)):
      data = self.video_loader(self.video_paths[i])
      videos_mean.append(np.mean(data))
      pbar.update()

    dataset_mean = np.mean(videos_mean) # a media nao esta exata. o ultimo batch eh menor
    dataset_std = 0

    for i in range(len(self.video_paths)):
      data = self.video_loader(self.video_paths[i])
      dataset_std += np.sum(np.square(data - dataset_mean))
      pbar.update()

    dataset_std = math.sqrt(dataset_std/(len(self.video_paths)*75*50*100*3))
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

      xy = vid_obj.load_video(self.epoch)
      x.append(xy[0])
      y.append(xy[1])
      max_y_size = max(max_y_size, len(y[-1]))
  
    x = np.array(x)
    y = np.array([Align.add_padding(elem, max_y_size) for elem in y]) # adicionar padding no y no carregamento dos aligns
    return x, y