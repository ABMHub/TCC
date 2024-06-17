import tensorflow as tf
import numpy as np
from generator.align_processing import Align
import tqdm
import random
from generator.augmentation import Augmentation
from generator.video_generator import VideoGenerator
from generator.data_config import DataConfig

RANDOM_SEED = 42
  
class BatchGenerator(tf.keras.utils.Sequence):
  def __init__(self, 
               config          : DataConfig, 
               batch_size      : int, 
               training        : bool,
               augmentation    : list[Augmentation] = None,
               standardize     : bool               = True
               ) -> None:
    super().__init__()

    random.seed(RANDOM_SEED)

    data = config.train if training else config.test
    self.mean, self.std_var = config.mean, config.std

    self.x_paths = np.array(data[0])
    self.modal_quant = len(self.x_paths)

    self.epoch = 0

    self.batch_size = batch_size
    self.training = training
    self.augs = augmentation

    mode_str = 'Treinamento' if self.training else 'Validação'
    pbar = tqdm.tqdm(data[1], desc=f"{mode_str}: carregando os alinhamentos")
    self.aligns = [Align(elem) for elem in pbar]
    # self.data = list(zip(self.x_paths, self.aligns))
    self.data_len = len(self.x_paths[0])

    self.generator_steps = int(np.ceil(self.data_len / self.batch_size))

    for i in range(len(self.mean)):
      if self.mean[i] is None or self.std_var[i] is None:
        self.mean[i], self.std_var[i] = self.__get_std_params(modal_position = i)

    self.video_gen = VideoGenerator(
      augs            = self.augs, 
      training        = self.training, 
      mean            = self.mean, 
      std             = self.std_var, 
      apply_padding   = True, 
      standardize     = standardize
      )

  def get_strings(self):
    return [" ".join(elem.sentence) for elem in self.aligns]

  def on_epoch_end(self):
    self.epoch += 1

  def __get_std_params(self, modal_position : int = 0, no_channels : bool = False):
    pbar = tqdm.tqdm(
      desc=f'{modal_position}: Calculando media e desvio padrão', 
      total=len(self.x_paths)*2, 
      disable=False
    )

    temp_video_gen = VideoGenerator(
      augs          = self.augs, 
      training      = False,
      mean          = 0.,
      std           = 1.,
      apply_padding = False,
      standardize   = False,
    )

    def __sub(data):
      inp = [None] * self.modal_quant
      inp[modal_position] = data
      return inp

    temp_data = temp_video_gen.load_data(self.x_paths[modal_position, 0])
    temp_data = temp_video_gen.augment_data(__sub(temp_data), None)[0][modal_position]
    
    shape = temp_data.shape
    if shape[-1] > 12: no_channels = True # ! alerta de gambiarra
    axis = None if no_channels else tuple(range(len(shape)-1)) 

    videos_mean = [] 
    for i in range(self.data_len):
      temp_data = temp_video_gen.load_data(self.x_paths[modal_position, i])
      temp_data = temp_video_gen.augment_data(__sub(temp_data), None)[0][modal_position]

      videos_mean.append(np.mean(temp_data, axis=axis))
      pbar.update()

    dataset_mean = np.mean(videos_mean, axis=0) # a media nao esta exata. o ultimo batch eh menor
    if axis is None: dataset_mean = np.array([dataset_mean]) # ! continuacao da gambiarra
    dataset_std = np.zeros(len(dataset_mean))

    for i in range(self.data_len):
      temp_data = temp_video_gen.load_data(self.x_paths[modal_position, i])
      temp_data = temp_video_gen.augment_data(__sub(temp_data), None)[0][modal_position]

      dataset_std += np.sum(np.square(temp_data - dataset_mean), axis=axis)
      pbar.update()

    number_of_elements = np.prod(shape) // len(dataset_mean)
    dataset_std = np.sqrt(dataset_std/(self.data_len*number_of_elements))
    pbar.close()

    print(f"Média: {dataset_mean}\nDesvio Padrão: {dataset_std}")

    return dataset_mean, dataset_std    

  def __get_split_tuple(self, index : int) -> tuple[int, int]:
    """Get data slice from the batch index

    Args:
        index (int): batch index

    Returns:
        tuple[int, int]: start and end of batch slice
    """
    split_start = index * self.batch_size
    split_end   = split_start + self.batch_size

    if split_end > self.data_len:
      split_end = self.data_len

    return split_start, split_end

  def __len__(self) -> int:
    return self.generator_steps

  def __getitem__(self, index : int):
    return self.getitem(index)

  def getitem(self, index : int, standardize = True): # ! atualmente sem augmentation
    split = self.__get_split_tuple(index)

    batch_x = self.x_paths[:, split[0]:split[1]]
    x = [[] for _ in range(self.modal_quant)]

    batch_y = self.aligns[split[0]:split[1]]
    max_y_size = 0
    y = []

    for i in range(len(batch_x[0])): # batch size
      xp = []

      for modality in range(len(batch_x)): # number of data streams
        raw_data = self.video_gen.load_data(batch_x[modality][i])
        xp.append(raw_data)

      augmented_x, augmented_y = self.video_gen.augment_data(xp, batch_y[i])

      max_y_size = max(max_y_size, len(augmented_y))
      y.append(augmented_y)

      _ = [x[i].append(elem) for i, elem in enumerate(augmented_x)]

    x = [np.array(elem) for elem in x]

    y = np.array([Align.add_padding(elem, max_y_size) for elem in y])

    if self.modal_quant > 1: 
      return {f"modal{i+1}": x[i] for i in range(self.modal_quant)}, y
    
    return x[0], y