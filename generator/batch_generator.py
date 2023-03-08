import math
import tensorflow as tf
import numpy as np
from preprocessing.align_processing import Align
from util.video import loaders
import tqdm
import random

RANDOM_SEED = 42

class BatchGenerator(tf.keras.utils.Sequence):
  def __init__(self, data : tuple, batch_size : int, training : bool, curriculum_steps : tuple[int, int] = None, mean_and_std : tuple[float, float] = None) -> None:
    # assert batch_size > 1 or training is False, "Não é possível fazer augmentation em um batch-size de 1. É necessário batch-size ao menos de 2."
    assert training is False or curriculum_steps is not None, "É preciso passar as etapas do treinamento por curriculo no treinamento."
    super().__init__()

    self.video_paths = data[0]

    self.epoch = -1
    self.training_mode = 0
    self.curriculum_steps = curriculum_steps
    self.video_loader = self.__init_video_loader(self.video_paths[0])

    self.batch_size = batch_size
    self.training = training

    self.aligns = [Align(elem) for elem in tqdm.tqdm(data[1], desc=f"{'Treinamento' if self.training else 'Validação'}: carregando os alinhamentos")]
    self.data = []

    if training:
      for i in range(len(self.video_paths)):
        self.data.append((self.video_paths[i], self.aligns[i].number_string, False))
        self.data.append((self.video_paths[i], self.aligns[i].number_string, True))
      np.random.seed(RANDOM_SEED)
      np.random.shuffle(self.data)

    else:
      for i in range(len(self.video_paths)):
        self.data.append((self.video_paths[i], self.aligns[i].number_string))

    self.generator_steps = int(np.ceil(len(self.data) / self.batch_size))

    if mean_and_std is None:
      self.mean, self.std_var = 0, 1
      self.mean, self.std_var = self.__get_std_params()

    else:
      self.mean, self.std_var = mean_and_std

    self.on_epoch_end()

  def get_strings(self):
    return [" ".join(elem.sentence) for elem in self.aligns]

  def on_epoch_end(self):
    if self.training is False or self.curriculum_steps is None: return
    self.epoch += 1

    if self.epoch == self.curriculum_steps[0] and self.epoch != self.curriculum_steps[1]:
      self.data = []
      for i in range(len(self.video_paths)):
        for j in range(len(self.aligns[i])):
          self.data.append((self.video_paths[i], self.aligns[i][j]))

      self.generator_steps = int(np.ceil(len(self.data) / self.batch_size))
      np.random.seed(RANDOM_SEED)
      np.random.shuffle(self.data)

      print("Entrando no modo palavras soltas")
      self.training_mode = 1

    elif self.epoch == self.curriculum_steps[1]:
      self.data = []
      random.seed(RANDOM_SEED)
      for i in range(len(self.video_paths)):
        frames = list(range(75))
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
          for i in range(extra_frames):
            choice = random.choice(range(len(frames)))
            del frames[choice]
            
        if extra_frames < 0:
          for i in range(extra_frames*-1):
            choice = random.choice(range(len(frames)))
            frames.insert(choice, frames[choice])

        assert len(frames) == 75, f"{len(frames)}"

        self.data.append((self.video_paths[i], self.aligns[i].number_string, frames))

      self.generator_steps = int(np.ceil(len(self.data) / self.batch_size))

      print("Entrando no modo jitter")
      self.training_mode = 2

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

  def __init_video_loader(self, file):
    extension = file.split(".")[-1]
    return loaders[extension] # essa decisao pode ser feita para cada video... mas adiciona o custo de tempo do split para achar a extensao

  def __get_split_tuple(self, index : int):
    split_start = index * self.batch_size
    split_end   = split_start + self.batch_size

    if split_end > len(self.data):
      split_end = len(self.data)

    return split_start, split_end

  def __len__(self) -> int:
    return self.generator_steps

  def __process_videos(self, videos):
    x, y = [], []
    max_y_size = 0
    if not self.training:
      for elem in videos:
        x.append(self.video_loader(elem[0]))
        y.append(elem[1])
        max_y_size = max(max_y_size, len(y[-1]))
      
      x = (np.array(x) - self.mean)/self.std_var
      y = np.array([Align.add_padding(elem, max_y_size) for elem in y]) # adicionar padding no y no carregamento dos aligns

    else:
      if self.training_mode == 0:
        for elem in videos:
          npy_video = self.video_loader(elem[0])
          x.append(np.flip(npy_video, axis=2) if elem[2] is True else npy_video)
          y.append(elem[1])
          max_y_size = max(max_y_size, len(y[-1]))

        x = (np.array(x) - self.mean)/self.std_var
        y = np.array([Align.add_padding(elem, max_y_size) for elem in y])

      elif self.training_mode == 1:
        for elem in videos:
          npy_video = self.video_loader(elem[0])
          npy_video = npy_video[elem[1][0]:elem[1][1]+1]
          npy_video = (npy_video - self.mean)/self.std_var

          pad_size = 75 - npy_video.shape[0]
          npy_video = np.pad(npy_video, [(0, pad_size), (0, 0), (0, 0), (0, 0)], "constant", constant_values=0)

          x.append(npy_video)
          y.append(Align.sentence2number(elem[1][2]))
          max_y_size = max(max_y_size, len(y[-1]))

        x = np.array(x)
        y = np.array([Align.add_padding(elem, max_y_size) for elem in y])

      elif self.training_mode == 2:
        for elem in videos:
          npy_video = self.video_loader(elem[0])
          npy_video = npy_video[elem[2]]

          x.append(npy_video)
          y.append(elem[1])
          max_y_size = max(max_y_size, len(y[-1]))

        x = (np.array(x) - self.mean)/self.std_var
        y = np.array([Align.add_padding(elem, max_y_size) for elem in y])

    return x, y

  def __getitem__(self, index : int):
    split = self.__get_split_tuple(index)

    batch_videos = self.data[split[0]:split[1]]

    return self.__process_videos(batch_videos)