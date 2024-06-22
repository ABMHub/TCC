from keras import backend as K
import tensorflow as tf

class LipFormerDecoder(tf.keras.layers.Layer):
  def __init__(self, hidden_state_size : int, output_size : int, **kwargs):
    super(LipFormerDecoder, self).__init__(**kwargs)
    self.hidden_state_size = hidden_state_size
    self.output_size = output_size

  def build(self, input_shape): # [batch, timesteps, features]
    self.timesteps = input_shape[1]
    self.batch_size = input_shape[0]

    self.gru_vm_e = tf.keras.layers.GRU(self.hidden_state_size, activation="tanh", return_sequences=True)
    self.lfd = LipFormerDecoderCell(self.hidden_state_size, self.output_size)
    self.lfd = tf.keras.layers.RNN(self.lfd, return_sequences=True)

    super(LipFormerDecoder, self).build(input_shape)

  def call(self, inputs):
    h_p_e = self.gru_vm_e(inputs)

    out = self.lfd(inputs, constants=h_p_e)
    return out
  
  def get_config(self):
    config = super().get_config()
    config['output_size'] = self.output_size
    config['hidden_state_size'] = self.hidden_state_size
    return config

class LipFormerDecoderCell(tf.keras.layers.Layer):
  def __init__(self, hidden_state_size : int, output_size : int, **kwargs):
    super(LipFormerDecoderCell, self).__init__(**kwargs)
    self.hidden_state_size = hidden_state_size
    self.output_size = output_size
    self.state_size = (self.output_size, self.hidden_state_size)

  def build(self, input_shape):
    self.emb = self.add_weight(name='embedding', shape=(self.hidden_state_size, self.output_size), initializer=tf.keras.initializers.GlorotNormal(), trainable=True)

    self.concat1 = tf.keras.layers.Concatenate(1)
    self.concat2 = tf.keras.layers.Concatenate(2)

    self.gru_p_d_cell = tf.keras.layers.GRUCell(self.hidden_state_size, activation="tanh")

    self.d1 = tf.keras.layers.Dense(self.hidden_state_size, activation="tanh")
    self.d2 = tf.keras.layers.Dense(1)

    self.d3 = tf.keras.layers.Dense(self.output_size, activation="softmax")

  def call(self, inputs_at_t, states_at_t, constants):
    h_p_e = constants[0]
    p_i, h_p_d_i = states_at_t

    p_i = tf.expand_dims(p_i, 1)
    emb_out = tf.multiply(p_i, self.emb)
    emb_out = tf.reduce_sum(emb_out, 1)
    h_p_d_i = self.gru_p_d_cell(emb_out, h_p_d_i)[0]
    
    h_p_d_i_e = tf.repeat(tf.expand_dims(h_p_d_i, 1), 75, axis=1)
    att_in = self.concat2([h_p_d_i_e, h_p_e])

    att = self.d2(self.d1(att_in))
    att = tf.keras.activations.softmax(att, axis=-2)

    context = tf.reduce_sum(tf.multiply(att, h_p_e), 1)
    p_i = self.d3(self.concat1([h_p_d_i, context]))

    return p_i, (p_i, h_p_d_i)
