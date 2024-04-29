import tensorflow as tf
from keras import backend as K

from model.loss import CTCLoss
from model.layers.layers import Highway, CascadedAttention, LipformerEncoder, ChannelAttention, LipNetEncoder

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