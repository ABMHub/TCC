import tensorflow as tf
from keras import backend as K

from model.layers.layers import LipformerEncoder, ChannelAttention, LipNetEncoder
from model.layers.lipformer_decoder import LipFormerDecoder
from model.layers.cascaded_attention import CascadedAttention, Highway

from model.architectures.architecture import Architecture

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
    visual_input = tf.keras.layers.Input(shape=self.shape, name="modal1") # video stream
    landmark_input = tf.keras.layers.Input(shape=(75, 340), name="modal2") # landmark branch

    visual_model = LipNetEncoder(attention=True)(visual_input)

    # ch_att_output = ChannelAttention(16)(visual_model)
    visual_model = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(visual_model)

    # visual_model = visual_model * K.expand_dims(ch_att_output)

    visual_model = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256, return_sequences=True))(visual_model)
    visual_model = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256, return_sequences=True))(visual_model)

    landmark_model = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256, return_sequences=True))(landmark_input)
    landmark_model = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256, return_sequences=True))(landmark_model)

    model = LipformerEncoder(512, 256)(visual_model, landmark_model)

    model = LipFormerDecoder(256, 28)(model)
    model = tf.keras.activations.softmax(model)

    # model = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256, return_sequences=True))(model)
    # model = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256, return_sequences=True))(model)

    # model = tf.keras.layers.Dense(28, activation="softmax")(model)

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