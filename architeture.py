import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf

from keras.layers import GRU, Bidirectional, Conv3D, Attention, Input, BatchNormalization, Activation, Dropout, MaxPool3D, ZeroPadding3D, Flatten, TimeDistributed, Concatenate
from keras import Model

import tensorflow as tf
import keras

class CTCLoss(keras.losses.Loss):

    def __init__(self, name="ctc_loss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        # Compute the training-time loss value
        y_true = tf.cast(y_true, dtype="int64")
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")

        blank_idx = tf.transpose(tf.where(tf.equal(y_true, tf.constant(-2, dtype="int64"))))[1]
        # label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
        # label_length = tf.concat([blank_idx[0], tf.reshape(label_length, shape=(1,))], 0)[0]

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = tf.math.multiply(tf.reshape(blank_idx, shape=(batch_len, 1)), tf.ones(shape=(batch_len, 1), dtype="int64"))

        loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
        return loss

def get_model():
  # LCANet
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

  model = TimeDistributed(Flatten())(model)

  gru_output1, bck, fwd = Bidirectional(GRU(512, return_sequences=True, return_state=True))(model)

  hidden_states = Concatenate()([fwd, bck])
  model = Attention()([gru_output1, hidden_states])

  model, _ = GRU(28, return_sequences=True, return_state=True, activation=None)(model)
  model = Activation("softmax")(model)

  model = Model(input, model)
  compile_model(model)

  return model

def compile_model(model : Model):
    model.compile(tf.keras.optimizers.Adam(learning_rate=1e-4), loss=CTCLoss())
