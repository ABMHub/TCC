import tensorflow as tf
from keras import backend as K

from model.loss import CTCLoss
from model.layers import Highway, CascadedAttention, LipformerEncoder, ChannelAttention, LipNetEncoder, LipNetREncoder

class Architecture():
  def __init__(self):
    self.name = "Empty"

  def get_model(self):
    raise NotImplementedError 
  
  def compile_model(self, model : tf.keras.Model, learning_rate : float = 1e-4, loss = CTCLoss()):
    raise NotImplementedError
    

class LipNet(Architecture):
  def __init__(self, **kwargs):
    self.name = 'LipNet'

  def get_model(self):
    K.clear_session()
    input = tf.keras.layers.Input(shape=(75, 100, 50, 3))

    model = LipNetEncoder()(input)

    model = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(model)

    model = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256, return_sequences=True, kernel_initializer="orthogonal"))(model)
    model = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256, return_sequences=True, kernel_initializer="orthogonal"))(model)

    model = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(28, kernel_initializer="he_normal"))(model)
    model = tf.keras.layers.Activation("softmax")(model)

    model = tf.keras.Model(input, model)
    self.compile_model(model)

    return model

  def compile_model(self, model : tf.keras.Model, learning_rate : float = 1e-4, loss = CTCLoss()):
    model.compile(tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=loss)

class RLipNet(Architecture):
  def __init__(self, reflections = 3, **kwargs):
    self.name = 'Relfexive LipNet'
    self.reflections = reflections

  def get_model(self):
    K.clear_session()
    input = tf.keras.layers.Input(shape=(75, 100, 50, 3))

    model = LipNetREncoder(self.reflections)(input)

    model = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(model)

    model = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256, return_sequences=True, kernel_initializer="orthogonal"))(model)
    model = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256, return_sequences=True, kernel_initializer="orthogonal"))(model)

    model = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(28, kernel_initializer="he_normal"))(model)
    model = tf.keras.layers.Activation("softmax")(model)

    model = tf.keras.Model(input, model)
    self.compile_model(model)

    return model

  def compile_model(self, model : tf.keras.Model, learning_rate : float = 1e-4, loss = CTCLoss()):
    model.compile(tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=loss)

class LCANet(Architecture):
  def __init__(self, **kwargs):
    self.name = 'LCANet'

  def get_model(self):
    K.clear_session()
    # LCANet
    input = tf.keras.layers.Input(shape=(75, 100, 50, 3))
    
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

  def compile_model(self, model : tf.keras.Model, learning_rate : float = 1e-4, loss = CTCLoss()):
    model.compile(tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=loss)

class RLCANet(Architecture):
  def __init__(self, reflections = 3, **kwargs):
    self.name = 'Reflexive LCANet'
    self.reflections = reflections

  def get_model(self):
    K.clear_session()
    # LCANet
    input = tf.keras.layers.Input(shape=(75, 100, 50, 3))
    
    model = LipNetREncoder(self.reflections)(input)

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

  def compile_model(self, model : tf.keras.Model, learning_rate : float = 1e-4, loss = CTCLoss()):
    model.compile(tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=loss)
  
class LipFormer(Architecture):
  def __init__(self, **kwargs):
    self.name = 'LipFormer'  

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
  
  def compile_model(self, model : tf.keras.Model, learning_rate : float = 1e-4, loss = CTCLoss()):
    model.compile(tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=loss)

class m3D_2D_BLSTM(Architecture):
  def __init__(self, **kwargs):
    self.name = '3D 2D BLSTM'
    
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
  
  def compile_model(self, model : tf.keras.Model, learning_rate : float = 1e-4, loss = CTCLoss()):
    model.compile(tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=loss)