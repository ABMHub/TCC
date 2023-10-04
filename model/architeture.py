import datetime
from jiwer import cer, wer
from nltk.translate.bleu_score import sentence_bleu

import numpy as np

# import absl.logging
import os
# absl.logging.set_verbosity(absl.logging.ERROR)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from keras import backend as K

from model.loss import CTCLoss
from model.callbacks import MinEarlyStopping
from model.layers import Highway, CascadedAttention, LipformerEncoder, ChannelAttention, LipNetEncoder
from generator.data_loader import get_training_data
from generator.batch_generator import BatchGenerator

from model.decoder import ctc_decode_multiprocess

from keras_nlp.layers import TransformerEncoder

class LipReadingModel():
  def __init__(self, model_path : str = None, multi_gpu = False):
    self.model : tf.keras.Model = None
    self.data : dict[str, BatchGenerator] = None

    self.chars = dict()
    self.chars[26] = " "
    self.chars[27] = "_"
    self.chars[28] = ""
    for i in range(26):
      self.chars[i] = chr(i + 97)

    if model_path is not None:
      if multi_gpu:
        with tf.distribute.MirroredStrategy().scope():
          self.load_model(model_path)
      else:
        self.load_model(model_path)

    elif architecture is not None:
      if multi_gpu:
        with tf.distribute.MirroredStrategy().scope():
          self.model = self.architectures[architecture.lower()]()
      else:
        self.model = self.architectures[architecture.lower()]()

  def load_model(self, path : str, inplace = True):
    K.clear_session()
    model = tf.keras.models.load_model(path, custom_objects={'CTCLoss': CTCLoss(), "CascadedAttention": CascadedAttention})

    if inplace:
      self.model = model

    return model

  def save_model(self, path : str):
    self.model.save(path)

  def load_data(self, x_path : str, y_path : str, batch_size : int = 32, validation_only : bool = False, unseen_speakers : bool = False, landmark_features : bool = False):
    """Carrega dados e geradores no objeto LCANet.
    As seeds dos geradores são fixas.

    Args:
        x_path (str): caminho para a pasta com todos os vídeos
        y_path (str): caminho para a pasta com todos os alignments
        batch_size (int, optional): tamanho de cada batch/lote de treinamento. Defaults to 32.
        validation_slice (float, optional): porcentagem de dados separados para validação. Defaults to 0.2.
        validation_only (bool, optional): _description_. Defaults to False.
    """
    self.data = get_training_data(x_path, y_path, batch_size = batch_size, validation_only = validation_only, unseen_speakers = unseen_speakers, landmark_features = landmark_features)

  def fit(self, epochs : int = 1, tensorboard_logs : str = None, checkpoint_path : str = None, patience = 0) -> None:
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
      callback_list.append(MinEarlyStopping(monitor="val_loss", patience=patience, min_epoch=0))

    self.model.fit(x=self.data["train"], validation_data=self.data["validation"], epochs = epochs, callbacks=callback_list)#, use_multiprocessing=True, workers=2)

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
    sections = []
    section_size = len(raw_pred)/workers
    for i in range(workers):
      start = section_size * i
      end = section_size + start
      sections.append(raw_pred[int(round(start, 0)):int(round(end, 0))])

    strings = self.data["train"].get_strings()
    result = ctc_decode_multiprocess(sections, workers, strings, greedy=greedy, language_model=True)
    result_nolm = ctc_decode_multiprocess(sections, workers, strings, greedy=greedy, language_model=False)

    ret = []
    for res in [result, result_nolm]:
      decoded = []
      for vec in res:
        decoded += list(vec)

      sentences = []
      for p in decoded:
        sentence = []
        for chr in p[0][0]:
          sentence.append(self.chars[int(chr)])

        sentences.append("".join(sentence))

      ret.append(sentences)

    return ret

  def evaluate_model(self, predictions = None, save_metrics_file_path : str = None) -> tuple[float, float, float]:
    assert self.data is not None
    assert self.data["validation"] is not None

    if predictions is None:
      predictions, predictions_nolm = self.predict()

    true = self.data["validation"].get_strings()
    cer_m, wer_m, bleu_m = cer(true, predictions), wer(true, predictions), self.__bleu(true, predictions)
    cer_m_nolm, wer_m_nolm, bleu_m_nolm = cer(true, predictions_nolm), wer(true, predictions_nolm), self.__bleu(true, predictions_nolm)

    result_string  = f"With Language Model\nCER: {cer_m}\nWER: {wer_m}\nBLEU: {bleu_m}\n\n"
    result_string += f"Without Language Model\nCER: {cer_m_nolm}\nWER: {wer_m_nolm}\nBLEU: {bleu_m_nolm}\n"
    print(result_string)
    if save_metrics_file_path is not None:
      with open(os.path.join(save_metrics_file_path), "w") as f:
        f.write(result_string)

    return cer_m, wer_m, bleu_m

  def __bleu(self, references : list[str], predictions : list[str]) -> float:
    references_p = [[reference.split()] for reference in references]
    predictions_p = [prediction.split() for prediction in predictions]
    return np.mean([sentence_bleu(ref, pred, [1, 0, 0, 0]) for ref, pred in zip(references_p, predictions_p)])

  def get_model(self):
    raise NotImplemented("Arquitetura não implementada.")

  def compile_model(self, model : tf.keras.Model, learning_rate : float = 1e-4, loss = CTCLoss()):
    model.compile(tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=loss)

class LipNet(LipReadingModel):
  def get_model(self):
    K.clear_session()
    input = tf.keras.layers.Input(shape=(75, 100, 50, 3))

    model = LipNetEncoder()(input)

    model = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(model)

    model = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256, return_sequences=True))(model)
    model = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256, return_sequences=True))(model)

    model = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(28))(model)
    model = tf.keras.layers.Activation("softmax")(model)

    model = tf.keras.Model(input, model)
    self.compile_model(model)

    return model

class LCANet(LipReadingModel):
  def get_model(self):
    K.clear_session()
    # LCANet
    input = tf.keras.layers.Input(shape=(75, 100, 50, 3))
    
    model = LipNetEncoder()(input)

    model = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(model)
    model = Highway()(model)
    model = Highway()(model)

    model = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256, return_sequences=True))(model)
    model = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256, return_sequences=True))(model)

    model = CascadedAttention(512, 28)(model)

    # model = tf.keras.layers.Dense(28)(model)
    model = tf.keras.layers.Activation("softmax")(model)

    model = tf.keras.Model(input, model)
    self.compile_model(model)

    return model
  
class LipFormer(LipReadingModel):
  def get_model(self):
    visual_input = tf.keras.layers.Input(shape=(75, 160, 80, 3), name="visual_input")
    landmark_input = tf.keras.layers.Input(shape=(75, 340), name="landmark_input")

    visual_model = LipNetEncoder()(visual_input)

    ch_att_output = ChannelAttention(16)(visual_model)
    visual_model = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(visual_model)

    visual_model = visual_model * K.expand_dims(ch_att_output)

    visual_model = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(512, return_sequences=True))(visual_model)
    visual_model = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(512, return_sequences=True))(visual_model)

    landmark_model = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(512, return_sequences=True))(landmark_input)
    landmark_model = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(512, return_sequences=True))(landmark_model)

    model = LipformerEncoder(1024, 512)(visual_model, landmark_model)

    model = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256, return_sequences=True))(model)
    model = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256, return_sequences=True))(model)

    model = tf.keras.layers.Dense(28, activation="softmax")(model)

    model = tf.keras.Model([visual_input, landmark_input], model)
    self.compile_model(model, learning_rate=3e-4)

    return model
  
class m3D_2D_BLSTM(LipReadingModel):
  def get_model(self):
    input = tf.keras.layers.Input(shape=(75, 100, 50, 3))

    model = tf.keras.layers.BatchNormalization()(input)

    model = tf.keras.layers.ZeroPadding3D(padding=(1, 2, 2))(model)
    model = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 5, 5), strides=(1, 2, 2))(model)
    model = tf.keras.layers.BatchNormalization()(model)
    model = tf.keras.layers.Activation("relu")(model)
    model = tf.keras.layers.MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(model)

    model = tf.keras.layers.ZeroPadding3D(padding=(1, 2, 2))(model)
    model = tf.keras.layers.Conv3D(filters=64, kernel_size=(3, 5, 5), strides=(1, 1, 1))(model)
    model = tf.keras.layers.BatchNormalization()(model)
    model = tf.keras.layers.Activation("relu")(model)
    model = tf.keras.layers.MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(model)

    model = tf.keras.layers.TimeDistributed(tf.keras.layers.ZeroPadding2D(padding=(2, 2)))(model)
    model = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=128, kernel_size=(5, 5), strides=(2, 2)))(model)
    model = tf.keras.layers.BatchNormalization()(model)
    model = tf.keras.layers.Activation("relu")(model)

    model = tf.keras.layers.TimeDistributed(tf.keras.layers.ZeroPadding2D(padding=(1, 1)))(model)
    model = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), strides=(2, 2)))(model)
    model = tf.keras.layers.BatchNormalization()(model)
    model = tf.keras.layers.Activation("relu")(model)

    model = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(model)
    model = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(200, return_sequences=True))(model)
    model = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(200, return_sequences=True))(model)

    model = tf.keras.layers.Dense(28, activation="softmax")(model)
    model = tf.keras.Model(input, model)

    self.compile_model(model)

    return model