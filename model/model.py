import datetime
import time
from jiwer import cer, wer

# import absl.logging
import os
# absl.logging.set_verbosity(absl.logging.ERROR)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from keras import backend as K

from model.loss import CTCLoss
from model.callbacks import MinEarlyStopping, TimePerBatch, WERCheckpoint
from model.layers.layers import Highway, CascadedAttention, LipformerEncoder, ChannelAttention, LipNetEncoder
from generator.data_loader import get_training_data
from generator.batch_generator import BatchGenerator
from generator.augmentation import Augmentation

from model.decoder import NgramCTCDecoder, Decoder

from model.evaluation import bleu, Evaluation
from model.architectures.video import LipNet, Architecture


class LipReadingModel():
  def __init__(
      self, 
      model_path : str = None,
      multi_gpu = False,
      pre_processing : str = None,
      architecture : Architecture = LipNet(),
      post_processing : Decoder = NgramCTCDecoder(),
      experiment_name : str = None,
      description : str = None
    ):
    self.model : tf.keras.Model = None
    self.data : dict[str, BatchGenerator] = None
    self.post_processing = post_processing
    self.experiment_name = experiment_name
    self.evaluation = Evaluation()
    self.model_path = model_path

    self.chars = dict()
    self.chars[26] = " "
    self.chars[27] = "_"
    self.chars[28] = ""
    for i in range(26):
      self.chars[i] = chr(i + 97)

    if model_path is not None:
      self.load_model(model_path)
    else:
      self.model = architecture.get_model()

    self.evaluation.data["preprocessing_type"] = pre_processing or self.evaluation.data["preprocessing_type"]
    if architecture is not None:
      self.evaluation.data["architecture_name"] = architecture.name
    if self.post_processing is not None:
      self.evaluation.data["postprocessing_type"] = self.post_processing.name

    self.evaluation.data["experiment_name"] = self.experiment_name or self.evaluation.data["experiment_name"]
    self.evaluation.data["description"] = description or self.evaluation.data["description"]
    self.evaluation.data["params"] = self.model.count_params() or self.evaluation.data["params"]

  def load_model(self, path : str, inplace = True):
    K.clear_session()
    self.model_path = path
    model = tf.keras.models.load_model(path, custom_objects={'CTCLoss': CTCLoss(), "CascadedAttention": CascadedAttention})
    self.evaluation.from_csv(path)

    if inplace:
      self.model = model

    return model

  def save_model(self, path : str):
    self.model.save(path)
    self.evaluation.to_csv(path)
    self.model_path = path

  def load_data(
      self,
      x_path            : str,
      y_path            : str,
      batch_size        : int                = 32, 
      validation_only   : bool               = False, 
      unseen_speakers   : bool               = False, 
      landmark_features : bool               = False, 
      post_processing   : Augmentation       = None,
      augmentation      : list[Augmentation] = None,
      is_time_series    : bool               = False,
      standardize       : bool               = True,
    ):
    
    """Carrega dados e geradores no objeto Lipreading.
    As seeds dos geradores são fixas.

    Args:
        x_path (str): caminho para a pasta com todos os vídeos
        y_path (str): caminho para a pasta com todos os alignments
        batch_size (int, optional): tamanho de cada batch/lote de treinamento. Defaults to 32.
        validation_slice (float, optional): porcentagem de dados separados para validação. Defaults to 0.2.
        validation_only (bool, optional): _description_. Defaults to False.
    """
    self.data = get_training_data(
      x_path, 
      y_path, 
      batch_size        = batch_size, 
      validation_only   = validation_only, 
      unseen_speakers   = unseen_speakers, 
      landmark_features = landmark_features, 
      post_processing   = post_processing,
      augmentation      = augmentation,
      is_time_series    = is_time_series,
      standardize       = standardize,
    )
    
    self.evaluation.data["augmentation"] = self.evaluation.data["augmentation"] or self.data["train"].video_gen.aug_name
    self.evaluation.data["lazy_process"] = self.evaluation.data["lazy_process"] or self.data["train"].video_gen.post_name
    self.evaluation.data["data_split"]   = self.evaluation.data["data_split"]   or "unseen" if unseen_speakers else "ovelapped"

  def fit(self, epochs : int = 1, tensorboard_logs : str = None, checkpoint_path : str = None, patience = 0) -> None:
    """Realiza o treinamento do modelo.

    Args:
      epochs (int, optional): número de épocas a serem executadas. Defaults to 1.
      tensorboard_logs (str, optional): caminho para salvar logs do tensorboard. Não salva nada em caso de None.
    """
    assert self.data is not None

    # self.tpb = TimePerBatch()
    # callback_list = [self.tpb]
    callback_list = []
    tensorboard_path = os.path.join(tensorboard_logs, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tb = tf.keras.callbacks.TensorBoard(tensorboard_path, histogram_freq=1)
    callback_list.append(tb)

    if checkpoint_path is not None:
      model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True
      )
      callback_list.append(model_checkpoint_callback)
      callback_list.append(MinEarlyStopping(monitor="val_loss", patience=patience, min_epoch=0))
      callback_list.append(WERCheckpoint(checkpoint_path + "_wer", self.model, generator=self.data["wer_validation"], log_dir=tensorboard_path))
    
    # self.evaluation.data["batches_per_epoch"] = self.data[1].generator_steps

    self.model.fit(x=self.data["train"], validation_data=self.data["validation"], epochs = epochs, callbacks=callback_list)#, use_multiprocessing=True, workers=2)
    self.evaluation.data["epochs_trained"] = self.data["train"].epoch

    if checkpoint_path is not None:
      self.evaluation.to_csv(checkpoint_path)
      self.evaluation.to_csv(checkpoint_path + "_wer")

  def predict(self, greedy = False) -> list[str]: # pd.dataframe?
    """Gera predição em string, utilizando beam_search.
    É necessário carregar os dados com a função `self.get_data` antes.

    Returns:
        list[str]: Lista de predições contendo todas as predições em cima do dataset de validação
    """
    assert self.data is not None
    
    print("Realizando predições...")
    raw_pred = self.model.predict(self.data["validation"])

    workers = 8

    strings = self.data["train"].get_strings()
    result = self.post_processing(workers, raw_pred, strings, greedy=greedy)
    # result_nolm = self.post_processing(sections, workers, strings, greedy=greedy, language_model=False)

    return result

  def evaluate_model(self, save_metrics_folder_path : str = None) -> tuple[float, float, float]:
    assert self.data is not None
    assert self.data["validation"] is not None

    pred_time = time.time()
    predictions = self.predict()
    pred_time = int(time.time() - pred_time)

    true = self.data["validation"].get_strings()
    
    self.evaluation.data["postprocessing_type"] = self.post_processing.name
    
    self.evaluation.data["cer"] = cer(true, predictions)
    self.evaluation.data["wer"] = wer(true, predictions)
    self.evaluation.data["bleu"] = bleu(true, predictions, False)
    self.evaluation.data["bleu_multigram"] = bleu(true, predictions, True)

    self.evaluation.data["params"] = self.model.count_params()
    self.evaluation.data["prediction_time"] = pred_time/len(predictions)

    self.evaluation.data["datetime"] = str(datetime.datetime.now())

    best_last = "last"
    if self.model_path.endswith("best"): best_last = "best"
    if self.model_path.endswith("wer"):  best_last = "wer"
    self.evaluation.data["best_last"] = best_last

    self.evaluation.to_csv(save_metrics_folder_path)
    self.evaluation.to_csv(self.model_path)

  def get_model(self):
    raise NotImplemented("Arquitetura não implementada.")

  def compile_model(self, model : tf.keras.Model, learning_rate : float = 1e-4, loss = CTCLoss()):
    raise NotImplementedError("Loss não implementada")
