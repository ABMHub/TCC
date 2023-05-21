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

class CascadedAttention(tf.keras.layers.Layer):
    def __init__(self, output_size : int, **kwargs):
        super(CascadedAttention, self).__init__(**kwargs)
        self.output_size = output_size # tamanho da output
 
    def build(self, input_shape): # [batch, timesteps, features]
        self.timesteps = input_shape[1]
        
        self.att = CascadedAttentionCell(self.output_size)
        self.att.build(input_shape)

        self.gru = CascadedGruCell(self.output_size)
        self.gru.build(input_shape)

        super(CascadedAttention, self).build(input_shape)
 
    def call(self, inputs):
        """_summary_

        Args:
            inputs: the hidden states of the encoder, of shape [batch, timesteps, dim]

        Returns:
            prediction: the softmaxed prediction for each timestep, of shape [batch, timestep, output_size]
        """
        batch_size = tf.shape(inputs)[0]
        prev_pred  = tf.zeros([batch_size, self.output_size])
        prev_state = self.gru.get_initial_state(batch_size=batch_size, dtype="float32")
        transpose_inputs = tf.transpose(inputs, [1, 0, 2])
        output = []

        for t in range(self.timesteps):
            context_vector = self.att(inputs, K.sigmoid(prev_pred))
            prev_pred = self.gru(context_vector, prev_pred, transpose_inputs[t-1] if t >= 0 else prev_state)
            output.append(prev_pred)

        output = tf.convert_to_tensor(output, dtype=tf.float32)
        output = tf.transpose(output, [1, 0, 2])
        return output
    
    def get_config(self):
        config = super().get_config()
        config['output_size'] = self.output_size
        config['timesteps'] = self.timesteps
        return config
    
class CascadedAttentionCell(tf.keras.layers.Layer):
    def __init__(self, output_size, **kwargs):
        super(CascadedAttentionCell, self).__init__(**kwargs)
        self.output_size = output_size
 
    def build(self, input_shape): # 75x1024
        dim = input_shape[-1]
        timesteps = input_shape[-2]
        # print("input_shape", input_shape)
        self.Wa  = self.add_weight(name='recurrent_attention_cell_weight',  shape=(self.output_size, self.output_size), initializer=tf.keras.initializers.GlorotNormal(), trainable=True)
        self.Ua  = self.add_weight(name='attention_cell_weight',            shape=(dim, self.output_size),              initializer=tf.keras.initializers.GlorotNormal(), trainable=True)
        self.Va  = self.add_weight(name='attention_cell_score_weight',      shape=(self.output_size, 1),                initializer=tf.keras.initializers.GlorotNormal(), trainable=True)
        self.Ba = self.add_weight(name='attention_cell_bias2',             shape=(1, self.output_size),                initializer=tf.keras.initializers.GlorotNormal(), trainable=True)

        super(CascadedAttentionCell, self).build(input_shape)

    def call(self, inputs, prev_state):
        """_summary_

        Args:
            inputs: encoder outputs of shape [batch_size, timestep, dim]
            prev_state: previous hidden state of the decoder GRUCell of shape [batch_size, output_size]

        Returns:
            context_vector: tensor of shape [batch_size, dim]
        """
        state_temp = K.expand_dims(prev_state, -2)
        WaS = tf.matmul(state_temp, self.Wa)

        UaH = tf.matmul(inputs, self.Ua)

        # WaS shape:       [batch,        1, output_size]
        # UaH shape:       [batch, timestep, output_size]
        # UaH + WaS shape: [batch, timestep, output_size]

        scores = K.tanh(UaH + WaS + self.Ba)
        scores = tf.matmul(scores, self.Va)

        # scores shape: [batch, timestep, 1]

        sm = K.softmax(scores, axis=1)                 
        context_vector = K.sum(inputs * sm, axis=1)

        return context_vector

class CascadedGruCell(tf.keras.layers.Layer):
    def __init__(self, output_size, **kwargs):
        super(CascadedGruCell, self).__init__(**kwargs)
        self.output_size = output_size
 
    def build(self, input_shape):
        feature_count = input_shape[-1]
        self.Wo  = self.add_weight(name='recurrent_gru_cell_weight',  shape=(self.output_size, 1),   initializer=tf.keras.initializers.GlorotNormal(), trainable=True)
        self.Uo  = self.add_weight(name='cascaded_gru_cell_weight',            shape=(feature_count, self.output_size),   initializer=tf.keras.initializers.GlorotNormal(), trainable=True)
        self.Co  = self.add_weight(name='context_cascaded_gru_cell_weight',            shape=(feature_count, self.output_size),   initializer=tf.keras.initializers.GlorotNormal(), trainable=True)

        self.Bo  = self.add_weight(name='cascaded_gru_cell_bias1',            shape=(1, self.output_size),   initializer=tf.keras.initializers.GlorotNormal(), trainable=True)

        self.emb  = self.add_weight(name='recurrent_gru_cell_weight_emb',  shape=(self.output_size, self.output_size),   initializer=tf.keras.initializers.GlorotNormal(), trainable=True)
        super(CascadedGruCell, self).build(input_shape)

    def call(self, context_vector, prev_prediction, prev_state):
        """_summary_

        Args:
            context_vector: output of the attention cell of shape:       [batch, dim]
            prev_prediction: previous output prediction of shape:        [batch, output_size]
            prev_state: previous state of the internal GruCell of shape: [batch, dim]

        Returns:
            prediction: prediction of shape: [batch, output_size]
        """
        emb_out = K.expand_dims(K.softmax(prev_prediction), -1) * self.emb
        emb_out = K.expand_dims(K.sum(emb_out, 1), 1)
        WoY = tf.matmul(emb_out, self.Wo)
        WoY = K.squeeze(WoY, 1)

        prev_state = K.expand_dims(prev_state, 1)
        UoH = tf.matmul(prev_state, self.Uo)
        UoH = K.squeeze(UoH, 1)

        context_vector = K.expand_dims(context_vector, 1)
        CoC = tf.matmul(context_vector, self.Co)
        CoC = K.squeeze(CoC, 1)

        # WoY, UoH, CoC shape: [batch, 28]

        pred = WoY + UoH + CoC + self.Bo

        return pred
    
    def get_initial_state(self, batch_size, dtype):
        return tf.zeros([batch_size, self.output_size], dtype=dtype)
    
class LipformerEncoder(tf.keras.layers.Layer):
    def __init__(self, output_size, **kwargs):
        super(LipformerEncoder, self).__init__(**kwargs)
        self.output_size = output_size
 
    def build(self, input_shape):
        self.timesteps = input_shape[1]
        self.dim = input_shape[2]
        self.timesteps = 1

        self.att_vis = tf.keras.layers.Attention()
        self.att_land = tf.keras.layers.Attention()
        self.cross_att_vis = tf.keras.layers.Attention()
        self.cross_att_land = tf.keras.layers.Attention()
        self.ffn1 = tf.keras.layers.Dense(self.dim, activation="relu")
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