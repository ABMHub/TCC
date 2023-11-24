from keras import backend as K
import tensorflow as tf

class RConv3D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, kernel_initializer, **kwargs):
        self.n_filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.kernel_initializer = kernel_initializer
        super(RConv3D, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.conv = tf.keras.layers.Conv3D(filters=self.n_filters, kernel_size=self.kernel_size, strides=self.strides, kernel_initializer=self.kernel_initializer)

        super(RConv3D, self).build(input_shape)  # Be sure to call this at the end

    def call(self, input):
        n = self.conv(input)
        r = tf.reverse(input, axis=[3])
        n2 = self.conv(r)        
        return n + tf.reverse(n2, axis=[3])

class LipNetEncoder(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(LipNetEncoder, self).__init__(**kwargs)
        self.w_init = "he_normal"

    def build(self, *args):
        self.conv1 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 5, 5), strides=(1, 2, 2), kernel_initializer=self.w_init)
        self.conv2 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 5, 5), strides=(1, 1, 1), kernel_initializer=self.w_init)
        self.conv3 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), kernel_initializer=self.w_init)

        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.batch_norm3 = tf.keras.layers.BatchNormalization()

    def call(self, input):
        model = tf.keras.layers.ZeroPadding3D(padding=(1, 2, 2))(input)
        model = self.conv1(model)
        model = self.batch_norm1(model)
        model = tf.keras.layers.Activation("relu")(model)
        model = tf.keras.layers.MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(model)
        model = tf.keras.layers.SpatialDropout3D(0.5)(model)

        model = tf.keras.layers.ZeroPadding3D(padding=(1, 2, 2))(model)
        model = self.conv2(model)
        model = self.batch_norm2(model)
        model = tf.keras.layers.Activation("relu")(model)
        model = tf.keras.layers.MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(model)
        model = tf.keras.layers.SpatialDropout3D(0.5)(model)

        model = tf.keras.layers.ZeroPadding3D(padding=(1, 1, 1))(model)
        model = self.conv3(model)
        model = self.batch_norm3(model)
        model = tf.keras.layers.Activation("relu")(model)
        model = tf.keras.layers.MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(model)
        model = tf.keras.layers.SpatialDropout3D(0.5)(model)

        return model
    
class LipNetREncoder(tf.keras.layers.Layer):
    def __init__(self, reflections = 3, **kwargs):
        super(LipNetREncoder, self).__init__(**kwargs)
        self.w_init = "he_normal"
        self.reflections = reflections

    def build(self, *args):
        self.conv = []

        for i, (kernel_size, strides) in enumerate([[(3, 5, 5), (1, 2, 2)], [(3, 5, 5), (1, 1, 1)], [(3, 3, 3), (1, 1, 1)]]):
            if self.reflections > i:
                self.conv.append(RConv3D(filters=32, kernel_size=kernel_size, strides=strides, kernel_initializer=self.w_init))
            else:
                self.conv.append(tf.keras.layers.Conv3D(filters=32, kernel_size=kernel_size, strides=strides, kernel_initializer=self.w_init))

        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.batch_norm3 = tf.keras.layers.BatchNormalization()

    def call(self, input):
        model = tf.keras.layers.ZeroPadding3D(padding=(1, 2, 2))(input)
        model = self.conv[0](model)
        model = self.batch_norm1(model)
        model = tf.keras.layers.Activation("relu")(model)
        model = tf.keras.layers.MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(model)
        model = tf.keras.layers.SpatialDropout3D(0.5)(model)

        model = tf.keras.layers.ZeroPadding3D(padding=(1, 2, 2))(model)
        model = self.conv[1](model)
        model = self.batch_norm2(model)
        model = tf.keras.layers.Activation("relu")(model)
        model = tf.keras.layers.MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(model)
        model = tf.keras.layers.SpatialDropout3D(0.5)(model)

        model = tf.keras.layers.ZeroPadding3D(padding=(1, 1, 1))(model)
        model = self.conv[2](model)
        model = self.batch_norm3(model)
        model = tf.keras.layers.Activation("relu")(model)
        model = tf.keras.layers.MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(model)
        model = tf.keras.layers.SpatialDropout3D(0.5)(model)

        return model

    def get_config(self):
        config = super().get_config()
        config['reflections'] = self.reflections
        return config
    
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
    
class LipformerEncoder(tf.keras.layers.Layer):
    def __init__(self, hidden_output_size, output_size, **kwargs):
        super(LipformerEncoder, self).__init__(**kwargs)
        self.output_size = output_size
        self.hidden_output_size = hidden_output_size

    def build(self, input_shape):
        self.timesteps = input_shape[1]
        self.dim = input_shape[2]
        self.timesteps = 1

        self.att_vis = tf.keras.layers.Attention()
        self.att_land = tf.keras.layers.Attention()
        self.cross_att_vis = tf.keras.layers.Attention()
        self.cross_att_land = tf.keras.layers.Attention()
        self.ffn1 = tf.keras.layers.Dense(self.hidden_output_size, activation="relu")
        self.ffn2 = tf.keras.layers.Dense(self.output_size)
        super(LipformerEncoder, self).build(input_shape)

    def call(self, visual_features, landmark_features):
        vis_out = self.att_vis([visual_features, visual_features, visual_features])
        land_out = self.att_land([landmark_features, landmark_features, landmark_features])
        cross_vis_out = self.cross_att_vis([vis_out, land_out, land_out])
        cross_land_out = self.cross_att_land([land_out, vis_out, vis_out])

        return self.ffn2(self.ffn1(cross_vis_out + cross_land_out))

    def get_config(self):
        config = super().get_config()
        config['output_size'] = self.output_size
        config['timesteps'] = self.timesteps
        config['dim'] = self.dim
        return config
    
# from the paper "CBAM: Convolutional Block Attention Module"
class ChannelAttention(tf.keras.layers.Layer):
    def __init__(self, ratio = 16, **kwargs):
        self.ratio = ratio
        super(ChannelAttention, self).__init__(**kwargs)
 
    def build(self, input_shape):
        self.dim = input_shape[1]

        self.ffn1 = tf.keras.layers.Dense(self.dim // self.ratio, activation="relu")
        self.ffn2 = tf.keras.layers.Dense(self.dim)
        super(ChannelAttention, self).build(input_shape)

    def _mlp(self, inputs):
        return self.ffn2(self.ffn1(inputs))

    def call(self, inputs):
        max_out = tf.keras.layers.GlobalMaxPool3D()(inputs)
        avg_out = tf.keras.layers.GlobalAvgPool3D()(inputs)

        mlp_out = K.sigmoid(self._mlp(max_out) + self._mlp(avg_out))

        return mlp_out
    
    def get_config(self):
        config = super().get_config()
        config['ratio'] = self.ratio
        config['dim'] = self.dim
        return config

class LipformerCharacterDecoder(tf.keras.layers.Layer):
    def __init__(self, output_size : int, **kwargs):
        super(LipformerCharacterDecoder, self).__init__(**kwargs)
        self.output_size = output_size # tamanho da output
 
    def build(self, input_shape): # [batch, timesteps, features]
        self.timesteps = input_shape[1]
        self.gru_vme = tf.keras.layers.GRU(256, return_sequences=True)
        self.gru_pd = tf.keras.layers.GRUCell(256)

        # self.ffn1 = tf.keras.layers.Dense(512)
        # self.ffn2 = tf.keras.layers.Dense(128)
        self.ffn3 = tf.keras.layers.Dense(self.output_size)

        self.attt = tf.keras.layers.Attention()

        self.flatten = tf.keras.layers.Flatten()

        self.emb  = self.add_weight(name='recurrent_gru_cell_weight_emb',  shape=(self.output_size, self.output_size),   initializer=tf.keras.initializers.GlorotNormal(), trainable=True)

        super(LipformerCharacterDecoder, self).build(input_shape)
 
    def call(self, inputs):
        """_summary_

        Args:
            inputs: the hidden states of the encoder, of shape [batch, timesteps, dim]

        Returns:
            prediction: the softmaxed prediction for each timestep, of shape [batch, timestep, output_size]
        """
        batch_size = tf.shape(inputs)[0]
        prev_pred  = tf.zeros([batch_size, self.output_size])
        prev_state = self.gru_pd.get_initial_state(batch_size=batch_size, dtype=tf.float32)
        output = []

        out_gru = self.gru_vme(inputs)      
        out_gru_t = tf.transpose(out_gru, [1, 0, 2])
        # out_gru_f = self.flatten(out_gru)

        for t in range(self.timesteps):
            emb_out = K.expand_dims(K.softmax(prev_pred), -1) * self.emb
            emb_out = K.sum(emb_out, 1)

            prev_state = self.gru_pd(emb_out, prev_state)[0]
            context = prev_state * self.attt([out_gru_t[t], prev_state])

            # concat = tf.concat([out_gru_f, prev_state], 1)
            # att = K.softmax(self.ffn2(K.tanh(self.ffn1(concat))))
            # att = self.attt(prev_state, out_gru_t[t])

            prev_pred = K.softmax(self.ffn3(tf.concat([prev_state, context], 1)))
            output.append(prev_pred)

        output = tf.convert_to_tensor(output, dtype=tf.float32)
        output = tf.transpose(output, [1, 0, 2])
        return output
    
    def get_config(self):
        config = super().get_config()
        config['timesteps'] = self.timesteps
        config['output_size'] = self.output_size
        return config
