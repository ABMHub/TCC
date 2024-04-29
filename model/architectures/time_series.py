import tensorflow as tf
from keras import backend as K

from model.layers.conformer import ConformerBlock
from model.architectures.architecture import Architecture

class LAS(Architecture):
  def __init__(self, **kwargs):
    
    self.name = "LipNet 1D"

    self.shape = (75, 20, 12)

    self.rnn_size = 256

    super().__init__(**kwargs)

  def get_model(self):
    K.clear_session()
    input = tf.keras.layers.Input(shape=self.shape)

    


class LipNet1D(Architecture):
  def __init__(self, **kwargs):
    
    self.name = "LipNet 1D"

    self.shape = (75, 20, 12)

    self.rnn_size = 256

    super().__init__(**kwargs)

  def get_model(self):
    K.clear_session()
    input = tf.keras.layers.Input(shape=self.shape)

    model = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(input)

    model = tf.keras.layers.ZeroPadding1D(padding=2)(model)
    model = tf.keras.layers.Conv1D(32, 5, 1)(model)
    model = tf.keras.layers.BatchNormalization()(model)
    model = tf.keras.layers.Activation("relu")(model)
    model = tf.keras.layers.MaxPool1D(pool_size=2, strides=1)(model)
    # model = tf.keras.layers.SpatialDropout1D(0.5)(model)

    model = tf.keras.layers.ZeroPadding1D(padding=2)(model)
    model = tf.keras.layers.Conv1D(64, 5, 1)(model)
    model = tf.keras.layers.BatchNormalization()(model)
    model = tf.keras.layers.Activation("relu")(model)
    model = tf.keras.layers.MaxPool1D(pool_size=2, strides=1)(model)
    # model = tf.keras.layers.SpatialDropout1D(0.5)(model)

    model = tf.keras.layers.ZeroPadding1D(padding=3)(model)
    model = tf.keras.layers.Conv1D(96, 7, 1)(model)
    model = tf.keras.layers.BatchNormalization()(model)
    model = tf.keras.layers.Activation("relu")(model)
    model = tf.keras.layers.MaxPool1D(pool_size=2, strides=1)(model)
    # model = tf.keras.layers.SpatialDropout1D(0.5)(model)

    model = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True, kernel_initializer="orthogonal"))(model)
    model = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True, kernel_initializer="orthogonal"))(model)

    model = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(28, kernel_initializer="he_normal"))(model)
    model = tf.keras.layers.Activation("softmax")(model)

    model = tf.keras.Model(input, model)
    self.compile_model(model)

    return model
  
class Conformer(Architecture):
  def __init__(self, **kwargs):
    self.name = 'Conformer'
    self.shape = (75, 20, 12)

    self.rnn_size = 256

    super().__init__(**kwargs)

  def get_model(self):
    K.clear_session()
    # LCANet
    input = tf.keras.layers.Input(shape=self.shape)
    model = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(input)
    
    model = tf.keras.layers.ZeroPadding1D(padding=2)(model)
    model = tf.keras.layers.Conv1D(32, 5, 1)(model)
    model = tf.keras.layers.BatchNormalization()(model)
    model = tf.keras.layers.Activation("relu")(model)
    model = tf.keras.layers.MaxPool1D(pool_size=2, strides=1)(model)

    model = tf.keras.layers.Dense(74)(model)
    model = tf.keras.layers.Dropout(0.1)(model)

    for _ in range(6):
      model = ConformerBlock(74, conv_expansion_factor=1)(model)

    model = tf.keras.layers.LSTM(28, return_sequences=True, kernel_initializer="orthogonal")(model)

    # model = tf.keras.layers.Dense(28)(model)
    model = tf.keras.layers.Activation("softmax")(model)

    model = tf.keras.Model(input, model)
    self.compile_model(model)

    return model