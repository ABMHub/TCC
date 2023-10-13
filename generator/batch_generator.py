import tensorflow as tf
import numpy as np
from generator.align_processing import Align
import tqdm
import random
from generator.augmentation import VideoGenerator, MirrorAug, JitterAug, SingleWords

RANDOM_SEED = 42
  
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
    self.lm_mean, self.lm_std_var = None, None
    if self.lm:
      for i in range(len(self.data)):
        self.data[i] = self.data[i] + (data[2][i],)

      self.lm_mean, self.lm_std_var = config.lm_mean, config.lm_std
      if self.lm_mean is None:
        self.lm_mean, self.lm_std_var = 0, 1
        self.lm_mean, self.lm_std_var = self.__get_std_params_landmarks()

    self.video_gen = VideoGenerator([MirrorAug(), SingleWords(), JitterAug()], self.training, self.mean, self.std_var, self.lm_mean, self.lm_std_var)

  def get_strings(self):
    return [" ".join(elem.sentence) for elem in self.aligns]

  def on_epoch_end(self):
    self.epoch += 1

  def __get_std_params(self):
    pbar = tqdm.tqdm(desc='Calculando media e desvio padrão', total=len(self.video_paths)*2, disable=False)
    videos_mean = []
    temp_video_gen = VideoGenerator([], False, 0, 1, 0, 1)
    for i in range(len(self.video_paths)):
      data = temp_video_gen.load_video(self.video_paths[i], self.aligns[i], 0, False)[0]
      videos_mean.append(np.mean(data, axis=(0, 1, 2)))
      pbar.update()

    shape = data.shape

    dataset_mean = np.mean(videos_mean, axis=0) # a media nao esta exata. o ultimo batch eh menor
    dataset_std = np.zeros(3)

    for i in range(len(self.video_paths)):
      data = temp_video_gen.load_video(self.video_paths[i], self.aligns[i], 0, False)[0]
      # todo salvar a soma durante o passo anterior, subtrair mean*100*50*3 da soma para nao carregar 2x
      dataset_std += np.sum(np.square(data - dataset_mean), axis=(0, 1, 2))
      pbar.update()

    dataset_std = np.sqrt(dataset_std/(len(self.video_paths)*shape[0]*shape[1]*shape[2]))
    pbar.close()

    print(f"Média: {dataset_mean}\nDesvio Padrão: {dataset_std}")

    return dataset_mean, dataset_std    
  
  def __get_std_params_landmarks(self):
    pbar = tqdm.tqdm(desc='Calculando media e desvio padrão das landmark features', total=len(self.data)*2, disable=False)
    lm_mean = []
    temp_video_gen = VideoGenerator([], False, 0, 1, 0, 1)
    for i in range(len(self.data)):
      data = temp_video_gen.load_landmark(self.data[i][2])
      lm_mean.append(np.mean(data))
      pbar.update()

    shape = data.shape

    dataset_mean = np.mean(lm_mean, axis=0) # a media nao esta exata. o ultimo batch eh menor
    dataset_std = 0

    for i in range(len(self.data)):
      data = temp_video_gen.load_landmark(self.data[i][2])
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
      xp, yp = self.video_gen.load_video(vid[0], vid[1], self.epoch, standardize)

      x.append(xp)
      y.append(yp)
      if self.lm: lm.append(self.video_gen.load_landmark(vid[2]))
      max_y_size = max(max_y_size, len(yp))
  
    x = np.array(x)
    y = np.array([Align.add_padding(elem, max_y_size) for elem in y])

    if self.lm: 
      return {"visual_input": x, "landmark_input": np.array(lm)}, y
    
    return x, y