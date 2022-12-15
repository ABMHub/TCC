import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from keras.layers import GRU, Bidirectional, Conv3D, Attention, Input, BatchNormalization, Activation, Dropout, MaxPool3D, ZeroPadding3D, Flatten, Reshape, Dense
# from highway_layer import Highway
from keras import Model, Sequential
from tensorflow.nn import ctc_loss

# drop = # ? talvez exista uma dropout layer

import tensorflow as tf
from tensorflow import keras


class CTCLoss(keras.losses.Loss):
    """ A class that wraps the function of tf.nn.ctc_loss. 
    
    Attributes:
        logits_time_major: If False (default) , shape is [batch, time, logits], 
            If True, logits is shaped [time, batch, logits]. 
        blank_index: Set the class index to use for the blank label. default is
            -1 (num_classes - 1). 
    """

    def __init__(self, logits_time_major=False, blank_index=-1, 
                 name='ctc_loss'):
        super().__init__(name=name)
        self.logits_time_major = logits_time_major
        self.blank_index = blank_index

    def call(self, y_true, y_pred):
        """ Computes CTC (Connectionist Temporal Classification) loss. work on
        CPU, because y_true is a SparseTensor.
        """
        y_true = tf.cast(y_true, tf.int32)
        y_pred_shape = tf.shape(y_pred)
        logit_length = tf.fill([y_pred_shape[0]], y_pred_shape[1])
        print(y_true)
        print(y_pred)
        loss = tf.nn.ctc_loss(
            labels=y_true,
            logits=y_pred,
            label_length=None,
            logit_length=logit_length,
            logits_time_major=self.logits_time_major,
            blank_index=self.blank_index)
        return tf.math.reduce_mean(loss)

def get_model():
  input = Input(shape=(75, 100, 50, 3))
  model = ZeroPadding3D(padding=(1, 2, 2))(input)
  model = Conv3D(filters=32, kernel_size=(3, 5, 5), strides=(1, 2, 2))(model)
  model = BatchNormalization()(model)
  model = Activation("relu")(model)
  model = MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(model)

  model = ZeroPadding3D(padding=(1, 2, 2))(model)
  model = Conv3D(filters=64, kernel_size=(3, 5, 5), strides=(1, 1, 1))(model)
  model = BatchNormalization()(model)
  model = Activation("relu")(model)
  model = MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(model)

  model = ZeroPadding3D(padding=(1, 2, 2))(model)
  model = Conv3D(filters=96, kernel_size=(3, 5, 5), strides=(1, 1, 1))(model) # ? ta falando 1 2 2 no artigo, mas nao bate com o summar
  model = BatchNormalization()(model)
  model = Activation("relu")(model)
  model = MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(model)

  # model = Highway()(model) # foi removida porque é facil implementar com a api do keras (pesquisar)
  # model = Highway()(model) # aprender a usar direito, é pra fazer o reshape diretamente nela

  model = Reshape((75, 1728))(model)

  gru = Bidirectional(GRU(512, return_sequences=True, return_state=True))
  encoder_outputs, encoder_state_fwd_h, encoder_state_bkw_h = gru(model)

  model = Attention()([encoder_state_fwd_h, encoder_state_bkw_h])
  # model = Dense(1, activation="sigmoid")(model)

  model = Model(input, model)

  ctc_l = ctc_loss#(["sil", "bin", "blue", "at", "f", "two", "now", "sil"])

  model.compile("adam", loss=CTCLoss())

  # model.summary()

  return model

get_model()