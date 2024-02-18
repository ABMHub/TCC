import tensorflow as tf
from keras import backend as K

from model.loss import CTCLoss
from model.layers import Highway, CascadedAttention, LipformerEncoder, ChannelAttention, LipNetEncoder

from math import ceil

class Architecture():
  def __init__(self, half_frame : bool = False, frame_sample = 1, **kwargs):
    self.name = self.name or "Empty"

    if len(self.shape) == 4:
      f, w, h, c = self.shape

      self.rnn_size = 256
      if half_frame:
        self.rnn_size /= 2
        self.shape = (f, w//2, h, c)

      if frame_sample and frame_sample > 1:
        self.rnn_size /= frame_sample
        self.shape = (ceil(f/frame_sample), w, h, c)
    
    self.rnn_size = int(self.rnn_size)
    self.metrics = []

    self.model : tf.keras.Model = None
    
  def get_model(self):
    raise NotImplementedError 
  
  def compile_model(self, model : tf.keras.Model, learning_rate : float = 1e-4, loss = CTCLoss(), **kwargs):
    model.compile(tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=loss, **kwargs)

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

    # model = tf.keras.layers.ZeroPadding1D(padding=2)(model)
    model = tf.keras.layers.Conv1D(32, 5, 1)(model)
    model = tf.keras.layers.BatchNormalization()(model)
    model = tf.keras.layers.Activation("relu")(model)
    model = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)(model)
    # model = tf.keras.layers.SpatialDropout1D(0.5)(model)

    # model = tf.keras.layers.ZeroPadding1D(padding=2)(model)
    model = tf.keras.layers.Conv1D(32, 7, 1)(model)
    model = tf.keras.layers.BatchNormalization()(model)
    model = tf.keras.layers.Activation("relu")(model)
    model = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)(model)
    # model = tf.keras.layers.SpatialDropout1D(0.5)(model)

    # model = tf.keras.layers.ZeroPadding1D(padding=1)(model)
    # model = tf.keras.layers.Conv1D(32, 3, 1)(model)
    # model = tf.keras.layers.BatchNormalization()(model)
    # model = tf.keras.layers.Activation("relu")(model)
    # model = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)(model)
    # model = tf.keras.layers.SpatialDropout1D(0.5)(model)

    model = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(self.rnn_size, return_sequences=True, kernel_initializer="orthogonal"))(model)
    model = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(self.rnn_size, return_sequences=True, kernel_initializer="orthogonal"))(model)

    model = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(28, kernel_initializer="he_normal"))(model)
    model = tf.keras.layers.Activation("softmax")(model)

    model = tf.keras.Model(input, model)
    self.compile_model(model)

    return model

class LipNet(Architecture):
  def __init__(self, **kwargs):
    self.name = 'LipNet'
    self.shape = (75, 100, 50, 3)

    self.rnn_size = 256
    
    super().__init__(**kwargs)

  def get_model(self):
    K.clear_session()
    input = tf.keras.layers.Input(shape=self.shape)

    model = LipNetEncoder()(input)

    model = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(model)

    model = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(self.rnn_size, return_sequences=True, kernel_initializer="orthogonal"))(model)
    model = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(self.rnn_size, return_sequences=True, kernel_initializer="orthogonal"))(model)

    model = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(28, kernel_initializer="he_normal"))(model)
    model = tf.keras.layers.Activation("softmax")(model)

    model = tf.keras.Model(input, model)
    self.compile_model(model)

    return model

class LCANet(Architecture):
  def __init__(self, **kwargs):
    self.name = 'LCANet'
    self.shape = (75, 100, 50, 3)

    super().__init__(**kwargs)

  def get_model(self):
    K.clear_session()
    # LCANet
    input = tf.keras.layers.Input(shape=self.shape)
    
    model = LipNetEncoder()(input)

    model = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(model)
    model = Highway()(model)
    model = Highway()(model)

    model = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256, return_sequences=True, kernel_initializer="orthogonal"))(model)
    model = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256, return_sequences=True, kernel_initializer="orthogonal"))(model)

    model = CascadedAttention(512, 28)(model)

    # model = tf.keras.layers.Dense(28)(model)
    model = tf.keras.layers.Activation("softmax")(model)

    model = tf.keras.Model(input, model)
    self.compile_model(model)

    return model

class LipFormer(Architecture):
  def __init__(self, **kwargs):
    self.name = 'LipFormer'  
    self.shape = (75, 160, 80, 3)

    super().__init__(**kwargs)

  def get_model(self):
    visual_input = tf.keras.layers.Input(shape=self.shape, name="visual_input")
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
  
class m3D_2D_BLSTM(Architecture):
  def __init__(self, **kwargs):
    self.name = '3D 2D BLSTM'
    self.shape = (75, 100, 50, 3)

    super().__init__(**kwargs)
    
  def get_model(self):
    input = tf.keras.layers.Input(shape=self.shape)

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
  

architecture_list = [LipNet(), LipNet1D(), LCANet(), LipFormer(), m3D_2D_BLSTM()]