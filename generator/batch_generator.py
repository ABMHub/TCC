import tensorflow as tf
import numpy as np
from preprocessing.align_processing import Align
from util.video import loaders
import tqdm
import random
# from generator.data_loader import DataConfig

RANDOM_SEED = 42

class VideoData:
  def __init__(self, video_path : str, align : Align, training : bool, mean : float, std : float, landmark_path = None, landmark_mean = None, landmark_std = None):
    self.video_path = video_path
    self.info = None
    self.align = align
    self.training = training
    self.mean, self.std = mean, std
    self.reversed = bool(random.getrandbits(1))

    randnum = random.random()
    self.subsentences = randnum < 0.2
    self.jitter = randnum >= 0.2

    self.landmark_path = landmark_path
    self.lm_mean, self.lm_std = landmark_mean, landmark_std

  def load_video(self, epoch, standardize = True):
    extension = self.video_path.split(".")[-1]
    video_loader = loaders[extension]

    y = None
    mask = None
    jitter = None

    video = video_loader(self.video_path)
    if self.training is True:
      if self.subsentences is True:
        begin, end, y = self.generate_subsentence_mask(epoch)
        limits = (begin, end)

      elif self.jitter is True:
        jitter = self.generate_jitter()
        y = self.align.number_string

      else:
        y = self.align.number_string

      if self.reversed is True:
        video = np.flip(video, axis=2)

    else:
      y = self.align.number_string

    x = np.array(video)
    if standardize:
      x = (x - self.mean)/self.std

    assert mask is None or jitter is None, "Proibido jitter e subsentence ao mesmo tempo!"

    if limits is not None:
      x = x[mask[0]:mask[1]+1]

    if jitter is not None:
      x = x[jitter]

    pad_size = 75 - x.shape[0]
    x = np.pad(x, [(0, pad_size), (0, 0), (0, 0), (0, 0)], "constant", constant_values=0)

    return x, y

  def load_landmark(self): # as landmarks precisam sofrer augmentation tambem. provavelmente calcular angulos durante o treino
    assert self.landmark_path is not None and self.lm_mean is not None, "Landmark feature load error"

    extension = self.landmark_path.split(".")[-1]
    loader = loaders[extension]

    return (loader(self.landmark_path) - self.lm_mean)/self.lm_std

  def generate_subsentence_mask(self, epoch):
    subsentence_idx = random.randint(0, 5)

    return self.align.get_sub_sentence(subsentence_idx, 1)

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
  def __init__(self, config, batch_size : int, training : bool) -> None:
    super().__init__()

    random.seed(RANDOM_SEED)

    data = config.train if training else config.test
    self.mean, self.std_var = config.mean, config.std
    # self.lm_mean, self.lm_std_var = None, None

    self.video_paths = data[0]

    self.epoch = 0

    self.batch_size = batch_size
    self.training = training

    self.aligns = [Align(elem) for elem in tqdm.tqdm(data[1], desc=f"{'Treinamento' if self.training else 'Validação'}: carregando os alinhamentos")]
    self.data = []

    for i in range(len(self.video_paths)):
      self.data.append((self.video_paths[i], self.aligns[i]))

    self.generator_steps = int(np.ceil(len(self.data) / self.batch_size))

    if self.mean is None:
      self.mean, self.std_var = np.zeros(3), np.ones(3)
      self.mean, self.std_var = self.__get_std_params()

    self.lm = config.lm
    if self.lm:
      for i in range(len(self.data)):
        self.data[i] = self.data[i] + (data[2][i],)

      self.lm_mean, self.lm_std_var = config.lm_mean, config.lm_std
      if self.lm_mean is None:
        self.lm_mean, self.lm_std_var = 0, 1
        self.lm_mean, self.lm_std_var = self.__get_std_params_landmarks()

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

    shape = data.shape

    dataset_mean = np.mean(videos_mean, axis=0) # a media nao esta exata. o ultimo batch eh menor
    dataset_std = np.zeros(3)

    for i in range(len(self.video_paths)):
      data = VideoData(self.video_paths[i], self.aligns[i], False, 0, 1).load_video(None)[0]
      dataset_std += np.sum(np.square(data - dataset_mean), axis=(0, 1, 2))
      pbar.update()

    dataset_std = np.sqrt(dataset_std/(len(self.video_paths)*shape[0]*shape[1]*shape[2]))
    pbar.close()

    print(f"Média: {dataset_mean}\nDesvio Padrão: {dataset_std}")

    return dataset_mean, dataset_std    
  
  def __get_std_params_landmarks(self):
    pbar = tqdm.tqdm(desc='Calculando media e desvio padrão das landmark features', total=len(self.data)*2, disable=False)
    lm_mean = []
    for i in range(len(self.data)):
      data = VideoData(None, None, False, 0, 1, self.data[i][2], self.lm_mean, self.lm_std_var).load_landmark()
      lm_mean.append(np.mean(data))
      pbar.update()

    shape = data.shape

    dataset_mean = np.mean(lm_mean, axis=0) # a media nao esta exata. o ultimo batch eh menor
    dataset_std = 0

    for i in range(len(self.data)):
      data = VideoData(None, None, False, 0, 1, self.data[i][2], self.lm_mean, self.lm_std_var).load_landmark()
      dataset_std += np.sum(np.square(data - dataset_mean))
      pbar.update()

    dataset_std = np.sqrt(dataset_std/(len(self.data)*shape[0]*shape[1]))
    pbar.close()

    print(f"Landmarks\nMédia: {dataset_mean}\nDesvio Padrão: {dataset_std}")

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
    return self.getitem(index)

  def getitem(self, index : int, standardize = True):
    split = self.__get_split_tuple(index)

    batch_videos = self.data[split[0]:split[1]]
    x = []
    y = []
    lm = []
    max_y_size = 0
    for vid in batch_videos:
      lm_dict = {}
      if self.lm:
        lm_dict = {
          "landmark_path": vid[2],
          "landmark_mean": self.lm_mean,
          "landmark_std": self.lm_std_var
        }

      vid_obj = VideoData(vid[0], vid[1], self.training, self.mean, self.std_var, **lm_dict)

      xp, yp = vid_obj.load_video(self.epoch, standardize)
      x.append(xp)
      y.append(yp)
      if self.lm: lm.append(vid_obj.load_landmark())
      max_y_size = max(max_y_size, len(yp))
  
    x = np.array(x)
    y = np.array([Align.add_padding(elem, max_y_size) for elem in y])

    if self.lm: 
      return {"visual_input": x, "landmark_input": np.array(lm)}, y
    
    return x, y