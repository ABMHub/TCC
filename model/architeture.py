import datetime
from ctc_decoder import beam_search
from jiwer import cer, wer
from multiprocessing import Pool
from keras import backend as K
from nltk.translate.bleu_score import corpus_bleu
import tqdm

import absl.logging
import os
absl.logging.set_verbosity(absl.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

from model.loss import CTCLoss
from model.highway_layer import Highway
from generator.data_loader import get_training_data

class LCANet():
  def __init__(self, model_path : str = None, architecture : str = "LCANet"):
    self.model = None
    self.data = None

    architectures = {
      "lcanet": self.__get_model_lcanet,
      "lipnet": self.__get_model_lipnet
    }

    if model_path is None:
      self.model = architectures[architecture.lower()]()

    else:
      self.load_model(model_path)

  def load_model(self, path : str, inplace = True):
    K.clear_session()
    model = tf.keras.models.load_model(path, compile=False)
    self.__compile_model(model)

    if not inplace:
      return model

    self.model = model

  def save_model(self, path : str):
    self.model.save(path)

  def load_data(self, x_path : str, y_path : str, batch_size : int = 32, validation_slice : float = 0.2, validation_only = False, curriculum_steps = None):
    """Carrega dados e geradores no objeto LCANet.
    As seeds dos geradores são fixas.

    Args:
        x_path (str): caminho para a pasta com todos os vídeos
        y_path (str): caminho para a pasta com todos os alignments
        batch_size (int, optional): tamanho de cada batch/lote de treinamento. Defaults to 32.
        validation_slice (float, optional): porcentagem de dados separados para validação. Defaults to 0.2.
        validation_only (bool, optional): _description_. Defaults to False.
    """
    self.data = get_training_data(x_path, y_path, batch_size = batch_size, val_size = validation_slice, validation_only = validation_only, curriculum_steps=curriculum_steps)

  def fit(self, epochs : int = 1, tensorboard_logs : str = None, checkpoint_path : str = None) -> None:
    """Realiza o treinamento do modelo.

    Args:
      epochs (int, optional): número de épocas a serem executadas. Defaults to 1.
      tensorboard_logs (str, optional): caminho para salvar logs do tensorboard. Não salva nada em caso de None.
    """
    assert self.data is not None

    callback_list = []
    if tensorboard_logs is not None:
      tb = tf.keras.callbacks.TensorBoard(os.path.join(tensorboard_logs, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")), histogram_freq=1)
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

    self.model.fit(x=self.data["train"], validation_data=self.data["validation"], epochs = epochs, callbacks=callback_list, verbose=1)

  def predict(self) -> list[str]: # pd.dataframe?
    """Gera predição em string, utilizando beam_search.
    É necessário carregar os dados com a função `self.get_data` antes.

    Returns:
        list[str]: Lista de predições contendo todas as predições em cima do dataset de validação
    """
    assert self.data is not None
    

    print("Realizando predições...")
    raw_pred = self.model.predict(self.data["validation"])
    ret = []

    pool = Pool(processes=8)
    for res in tqdm.tqdm(pool.imap(beam_wrapper, raw_pred), "Convertendo predições para strings", total=len(raw_pred)):
      ret.append(res)

    return ret

  def evaluate_model(self, predictions = None, save_metrics_file_path : str = None) -> tuple[float, float, float]:
    assert self.data is not None
    assert self.data["validation"] is not None

    if predictions is None:
      predictions = self.predict()

    true = self.data["validation"].get_strings()
    cer_m, wer_m, bleu_m = cer(true, predictions), wer(true, predictions), self.__bleu(true, predictions)

    if save_metrics_file_path is not None:
      result_string = f"\nCER: {cer_m}\nWER: {wer_m}\nBLEU: {bleu_m}\n"
      print(result_string)
      with open(os.path.join(save_metrics_file_path), "w") as f:
        f.write(result_string)

    return cer_m, wer_m, bleu_m

  def __bleu(self, references : list[str], predictions : list[str]) -> float:
    references_p = [[reference.split()] for reference in references]
    predictions_p = [prediction.split() for prediction in predictions]
    return corpus_bleu(references_p, predictions_p)

  def __get_model_lcanet(self):
    K.clear_session()
    # LCANet
    input = tf.keras.layers.Input(shape=(75, 100, 50, 3))
    model = tf.keras.layers.ZeroPadding3D(padding=(1, 2, 2))(input)
    model = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 5, 5), strides=(1, 2, 2))(model)
    model = tf.keras.layers.BatchNormalization()(model)
    model = tf.keras.layers.Activation("relu")(model)
    model = tf.keras.layers.MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(model)
    model = tf.keras.layers.SpatialDropout3D(0.5)(model)

    model = tf.keras.layers.ZeroPadding3D(padding=(1, 2, 2))(model)
    model = tf.keras.layers.Conv3D(filters=64, kernel_size=(3, 5, 5), strides=(1, 1, 1))(model)
    model = tf.keras.layers.BatchNormalization()(model)
    model = tf.keras.layers.Activation("relu")(model)
    model = tf.keras.layers.MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(model)
    model = tf.keras.layers.SpatialDropout3D(0.5)(model)

    model = tf.keras.layers.ZeroPadding3D(padding=(1, 2, 2))(model)
    model = tf.keras.layers.Conv3D(filters=96, kernel_size=(3, 5, 5), strides=(1, 1, 1))(model) # ? ta falando 1 2 2 no artigo, mas nao bate com o summar
    model = tf.keras.layers.BatchNormalization()(model)
    model = tf.keras.layers.Activation("relu")(model)
    model = tf.keras.layers.MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(model)
    model = tf.keras.layers.SpatialDropout3D(0.5)(model)

    model = Highway()(model) # foi removida porque é facil implementar com a api do keras (pesquisar)
    model = Highway()(model) # aprender a usar direito, é pra fazer o reshape diretamente nela

    model = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(model)

    gru_output1, gru_output2 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(512, return_sequences=True), None)(model)

    # model = tf.keras.layers.TimeDistributed(tf.keras.layers.Attention())([gru_output1, gru_output2])
    model = tf.keras.layers.MultiHeadAttention(75, 28)(gru_output1, gru_output2)

    model = tf.keras.layers.GRU(28, return_sequences=True, activation="softmax")(model)
    # model = Activation("softmax")(model)

    model = tf.keras.Model(input, model)
    self.__compile_model(model)

    return model
  
  def __get_model_lipnet(self):
    K.clear_session()
    # Lipnet
    input = tf.keras.layers.Input(shape=(75, 100, 50, 3))
    model = tf.keras.layers.ZeroPadding3D(padding=(1, 2, 2))(input)
    model = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 5, 5), strides=(1, 2, 2))(model)
    model = tf.keras.layers.BatchNormalization()(model)
    model = tf.keras.layers.Activation("relu")(model)
    model = tf.keras.layers.MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(model)
    model = tf.keras.layers.SpatialDropout3D(0.5)(model)

    model = tf.keras.layers.ZeroPadding3D(padding=(1, 2, 2))(model)
    model = tf.keras.layers.Conv3D(filters=64, kernel_size=(3, 5, 5), strides=(1, 1, 1))(model)
    model = tf.keras.layers.BatchNormalization()(model)
    model = tf.keras.layers.Activation("relu")(model)
    model = tf.keras.layers.MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(model)
    model = tf.keras.layers.SpatialDropout3D(0.5)(model)

    model = tf.keras.layers.ZeroPadding3D(padding=(1, 2, 2))(model)
    model = tf.keras.layers.Conv3D(filters=96, kernel_size=(3, 5, 5), strides=(1, 1, 1))(model)
    model = tf.keras.layers.BatchNormalization()(model)
    model = tf.keras.layers.Activation("relu")(model)
    model = tf.keras.layers.MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(model)
    model = tf.keras.layers.SpatialDropout3D(0.5)(model)

    model = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(model)

    model = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256, return_sequences=True))(model)
    model = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256, return_sequences=True))(model)

    model = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(28))(model)
    model = tf.keras.layers.Activation("softmax")(model)

    model = tf.keras.Model(input, model)
    self.__compile_model(model)

    return model

  def __compile_model(self, model : tf.keras.Model):
    model.compile(tf.keras.optimizers.Adam(learning_rate=1e-4), loss=CTCLoss())

def beam_wrapper(pred):
  return beam_search(pred, 'abcdefghijklmnopqrstuvwxyz ', beam_width = 200)
