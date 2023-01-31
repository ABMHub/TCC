import datetime
from preprocessing.preprocessing import get_training_data
from ctc_decoder import beam_search
from jiwer import cer, wer

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf

import tqdm

from model.loss import CTCLoss

class LCANet():
  def __init__(self, model_path : str = None):
    self.model = None
    self.data = None
    if model_path is None:
      self.model = self.__get_model()

    else:
      self.load_model(model_path)

  def load_model(self, path : str, inplace = True):
    model = tf.keras.models.load_model(path, compile=False)
    self.__compile_model(model)

    if not inplace:
      return model

    self.model = model

  def save_model(self, path : str):
    self.model.save(path)

  def load_data(self, x_path : str, y_path : str, batch_size : int = 32, validation_slice : float = 0.2, validation_only = False):
    """Carrega dados e geradores no objeto LCANet.
    As seeds dos geradores são fixas.

    Args:
        x_path (str): caminho para a pasta com todos os vídeos
        y_path (str): caminho para a pasta com todos os alignments
        batch_size (int, optional): tamanho de cada batch/lote de treinamento. Defaults to 32.
        validation_slice (float, optional): porcentagem de dados separados para validação. Defaults to 0.2.
        validation_only (bool, optional): _description_. Defaults to False.
    """
    self.data = get_training_data(x_path, y_path, batch_size = batch_size, val_size = validation_slice, validation_only = validation_only)

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

    self.model.fit(x=self.data["train"], validation_data=self.data["validation"], epochs = epochs, callbacks=callback_list)

  def predict(self) -> list[str]: # pd.dataframe?
    """Gera predição em string, utilizando beam_search.
    É necessário carregar os dados com a função `self.get_data` antes.

    Returns:
        list[str]: Lista de predições contendo todas as predições em cima do dataset de validação
    """
    assert self.data is not None
    vocab = "".join([chr(x) for x in range(97, 123)]) + " "

    print("Realizando predições...")
    raw_pred = self.model.predict(self.data["validation"])
    ret = []
    for pred in tqdm.tqdm(raw_pred, "Convertendo predições para strings"):
      ret.append(beam_search(pred, vocab))

    return ret

  def evaluate_model(self, predictions = None):
    assert self.data is not None
    assert self.data["validation"] is not None
    assert self.data["validation"].strings is not None

    if predictions is None:
      predictions = self.predict()

    true = [" ".join(x) for x in self.data["validation"].strings]

    return cer(true, predictions), wer(true, predictions)

  def __get_model(self):
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

    # model = Highway()(model) # foi removida porque é facil implementar com a api do keras (pesquisar)
    # model = Highway()(model) # aprender a usar direito, é pra fazer o reshape diretamente nela

    model = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(model)

    gru_output1, gru_output2 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(512, return_sequences=True), None)(model)

    model = tf.keras.layers.MultiHeadAttention(75, 28)(gru_output1, gru_output2)

    model = tf.keras.layers.GRU(28, return_sequences=True, activation="softmax")(model)
    # model = Activation("softmax")(model)

    model = tf.keras.Model(input, model)
    self.__compile_model(model)

    return model

  def __compile_model(self, model : tf.keras.Model):
    model.compile(tf.keras.optimizers.Adam(learning_rate=1e-4), loss=CTCLoss())
