from keras import backend as K
import tensorflow as tf

class Highway(tf.keras.layers.Layer):
    """
        Highway layer made by https://github.com/ParikhKadam
        I only adapted to tensorflow 2.10
    """
    activation = None
    transform_gate_bias = None

    def __init__(self, activation='relu', transform_gate_bias=-1, **kwargs):
        self.activation = activation
        self.transform_gate_bias = transform_gate_bias
        super(Highway, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        dim = input_shape[-1]
        transform_gate_bias_initializer = tf.keras.initializers.Constant(self.transform_gate_bias)
        self.dense_1 = tf.keras.layers.Dense(units=dim, bias_initializer=transform_gate_bias_initializer)
        self.dense_1.build(input_shape)
        self.dense_2 = tf.keras.layers.Dense(units=dim)
        self.dense_2.build(input_shape)
        self._trainable_weights = self.dense_1._trainable_weights + self.dense_2._trainable_weights

        super(Highway, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        dim = K.int_shape(x)[-1]
        transform_gate = self.dense_1(x)
        transform_gate = tf.keras.layers.Activation("sigmoid")(transform_gate)
        carry_gate = tf.keras.layers.Lambda(lambda x: 1.0 - x, output_shape=(dim,))(transform_gate)
        transformed_data = self.dense_2(x)
        transformed_data = tf.keras.layers.Activation(self.activation)(transformed_data)
        transformed_gated = tf.keras.layers.Multiply()([transform_gate, transformed_data])
        identity_gated = tf.keras.layers.Multiply()([carry_gate, x])
        value = tf.keras.layers.Add()([transformed_gated, identity_gated])
        return value

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config['activation'] = self.activation
        config['transform_gate_bias'] = self.transform_gate_bias
        return config

# @tf.keras.saving.register_keras_serializable('cascaded_attention_layer')
class CascadedAttention(tf.keras.layers.Layer):
    def __init__(self, hidden_state_size : int, output_size : int, **kwargs):
        super(CascadedAttention, self).__init__(**kwargs)
        self.hidden_state_size = hidden_state_size
        self.output_size = output_size

    def build(self, input_shape): # [batch, timesteps, features]
        cell = CascadedAttentionCell(self.hidden_state_size, self.output_size)
        self.casc_att = tf.keras.layers.RNN(cell, return_sequences=True)
        super(CascadedAttention, self).build(input_shape)

    def call(self, inputs):
        return self.casc_att(inputs, constants=inputs)
    
    def get_config(self):
        config = super().get_config()
        config['output_size'] = self.output_size
        config['hidden_state_size'] = self.hidden_state_size
        return config
    
class CascadedAttentionCell(tf.keras.layers.Layer):
    def __init__(self, hidden_state_size : int, output_size : int, **kwargs):
        super(CascadedAttentionCell, self).__init__(**kwargs)
        self.output_size = output_size # tamanho da output
        self.hidden_state_size = hidden_state_size
        self.state_size = (hidden_state_size, 1, 1)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return tf.zeros([batch_size, self.hidden_state_size], dtype=tf.float32), tf.zeros([batch_size, 1], dtype=tf.int64), tf.zeros([batch_size, 1], dtype=tf.int64)
 
    def build(self, input_shape): # [batch, timesteps, features]
        self.timesteps = input_shape[1]
        
        self.att = tf.keras.layers.AdditiveAttention()

        self.gru = CascadedGruCell(self.hidden_state_size, self.output_size)
        self.gru.build(input_shape)

        self.d = tf.keras.layers.Dense(28)

        super(CascadedAttentionCell, self).build(input_shape)
 
    def call(self, inputs_at_t, states_at_t, constants):
        """_summary_

        Args:
            inputs: the hidden states of the encoder, of shape [batch, timesteps, dim]

        Returns:
            prediction: the softmaxed prediction for each timestep, of shape [batch, timestep, output_size]
        """
        prev_state, prev_pred, t = states_at_t
        constants = constants[0]

        context_vector = K.squeeze(self.att([K.expand_dims(prev_state, 1), constants]), 1)
        state = self.gru(context_vector, prev_state, prev_pred, t)
        pred = self.d(prev_state)

        return pred, (state, K.expand_dims(K.argmax(pred), -1), t+1)
    
    def get_config(self):
        config = super().get_config()
        config['output_size'] = self.output_size
        config['timesteps'] = self.timesteps
        config['hidden_state_size'] = self.hidden_state_size
        config['state_size'] = self.state_size
        return config
    
class CascadedGruCell(tf.keras.layers.Layer):
    def __init__(self, hidden_size, output_size, **kwargs):
        super(CascadedGruCell, self).__init__(**kwargs)
        self.output_size = output_size
        self.hidden_size = hidden_size
 
    def build(self, input_shape):
        self.emb  = tf.keras.layers.Embedding(self.output_size + 1, self.hidden_size)

        self.gru = ContextGRU(self.hidden_size)
        self.gru.build(input_shape)
        super(CascadedGruCell, self).build(input_shape)

    def call(self, context_vector, prev_state, prev_pred, t):
        """_summary_

        Args:
            context_vector: output of the attention cell of shape:       [batch, dim]
            prev_prediction: previous output prediction of shape:        [batch, output_size]
            prev_state: previous state of the internal GruCell of shape: [batch, dim]

        Returns:
            prediction: prediction of shape: [batch, output_size]
        """
        if t[0][0] == 0:
            batch_size = tf.shape(context_vector)[0]
            emb_in = tf.zeros([batch_size, 1], dtype=tf.int64)
        else:
            emb_in = prev_pred + 1

        emb_out = K.squeeze(self.emb(emb_in), -2)

        return self.gru(emb_out, context_vector, prev_state)
    
    def get_initial_state(self, batch_size, dtype):
        return tf.zeros([batch_size, self.output_size], dtype=dtype)
    
    def get_config(self):
        config = super().get_config()
        config['output_size'] = self.output_size
        config['hidden_size'] = self.hidden_size
        return config
    
class ContextGRU(tf.keras.layers.Layer):
    def __init__(self, output_size, **kwargs):
        super(ContextGRU, self).__init__(**kwargs)
        self.output_size = output_size

    def build(self, input_shape):
        self.feature_count = input_shape[-1]

        self.Wr  = self.add_weight(name='recurrent_gru_cell_weight_r',  shape=(self.output_size, self.output_size),   initializer=tf.keras.initializers.GlorotNormal(), trainable=True)
        self.Ur  = self.add_weight(name='gru_cell_weight_r',            shape=(self.output_size, self.output_size), initializer=tf.keras.initializers.GlorotNormal(), trainable=True)
        self.Cr  = self.add_weight(name='gru_cell_score_weight_r',      shape=(self.feature_count, self.output_size), initializer=tf.keras.initializers.GlorotNormal(), trainable=True)

        self.Wz  = self.add_weight(name='recurrent_gru_cell_weight_z',  shape=(self.output_size, self.output_size),   initializer=tf.keras.initializers.GlorotNormal(), trainable=True)
        self.Uz  = self.add_weight(name='gru_cell_weight_z',            shape=(self.output_size, self.output_size), initializer=tf.keras.initializers.GlorotNormal(), trainable=True)
        self.Cz  = self.add_weight(name='gru_cell_score_weight_z',      shape=(self.feature_count, self.output_size), initializer=tf.keras.initializers.GlorotNormal(), trainable=True)

        self.Wh  = self.add_weight(name='recurrent_gru_cell_weight_h',  shape=(self.output_size, self.output_size),   initializer=tf.keras.initializers.GlorotNormal(), trainable=True)
        self.Uh  = self.add_weight(name='gru_cell_weight_h',            shape=(self.output_size, self.output_size), initializer=tf.keras.initializers.GlorotNormal(), trainable=True)
        self.Ch  = self.add_weight(name='gru_cell_score_weight_h',      shape=(self.feature_count, self.output_size), initializer=tf.keras.initializers.GlorotNormal(), trainable=True)

        self.br   = self.add_weight(name='cascaded_gru_cell_biasr',  shape=(1, self.output_size),   initializer=tf.keras.initializers.GlorotNormal(), trainable=True)
        self.bz   = self.add_weight(name='cascaded_gru_cell_biasz',  shape=(1, self.output_size),   initializer=tf.keras.initializers.GlorotNormal(), trainable=True)
        super(ContextGRU, self).build(input_shape)

    def call(self, inputs, context_vector, prev_state):
        z = K.sigmoid(K.dot(inputs,self.Wz) + K.dot(prev_state, self.Uz) + K.dot(context_vector, self.Cz) + self.bz)
        r = K.sigmoid(K.dot(inputs,self.Wr) + K.dot(prev_state, self.Ur) + K.dot(context_vector, self.Cr) + self.br)
        hh = K.tanh(K.dot(inputs,self.Wh) + K.dot((prev_state * r), self.Uh) + K.dot(context_vector, self.Ch))
        h = (1 - z) * hh + z * prev_state

        return h
    
    def get_config(self):
        config = super().get_config()
        config['output_size'] = self.output_size
        config['feature_count'] = self.feature_count
        return config