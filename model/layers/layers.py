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
        self.norm = tf.keras.layers.BatchNormalization()

        super(RConv3D, self).build(input_shape)  # Be sure to call this at the end

    def call(self, input):
        n = self.conv(input)
        n = self.norm(n)
        n = tf.keras.layers.Activation("relu")(n)
        
        r = tf.reverse(input, axis=[3])
        n2 = self.conv(r)   
        n2 = tf.reverse(n2, axis=[3])
        n2 = self.norm(n2)
        n2 = tf.keras.layers.Activation("relu")(n2)        
        
        return n + n2

class LipNetEncoder(tf.keras.layers.Layer):
    def __init__(self, reflexive = False, **kwargs):
        super(LipNetEncoder, self).__init__(**kwargs)
        self.w_init = "he_normal"
        self.reflexive = True

    def build(self, *args):
        conv_class = RConv3D if self.reflexive else tf.keras.layers.Conv3D

        self.conv1 = conv_class(filters=32, kernel_size=(3, 5, 5), strides=(1, 2, 2), kernel_initializer=self.w_init)
        self.conv2 = conv_class(filters=64, kernel_size=(3, 5, 5), strides=(1, 1, 1), kernel_initializer=self.w_init)
        self.conv3 = conv_class(filters=96, kernel_size=(3, 3, 3), strides=(1, 1, 1), kernel_initializer=self.w_init)

        # self.batch_norm1 = tf.keras.layers.BatchNormalization()
        # self.batch_norm2 = tf.keras.layers.BatchNormalization()
        # self.batch_norm3 = tf.keras.layers.BatchNormalization()

    def call(self, input):
        model = tf.keras.layers.ZeroPadding3D(padding=(1, 2, 2))(input)
        model = self.conv1(model)
        # model = self.batch_norm1(model)
        # model = tf.keras.layers.Activation("relu")(model)
        model = tf.keras.layers.MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(model)
        model = tf.keras.layers.SpatialDropout3D(0.5)(model)

        model = tf.keras.layers.ZeroPadding3D(padding=(1, 2, 2))(model)
        model = self.conv2(model)
        # model = self.batch_norm2(model)
        # model = tf.keras.layers.Activation("relu")(model)
        model = tf.keras.layers.MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(model)
        model = tf.keras.layers.SpatialDropout3D(0.5)(model)

        model = tf.keras.layers.ZeroPadding3D(padding=(1, 1, 1))(model)
        model = self.conv3(model)
        # model = self.batch_norm3(model)
        # model = tf.keras.layers.Activation("relu")(model)
        model = tf.keras.layers.MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(model)
        model = tf.keras.layers.SpatialDropout3D(0.5)(model)

        return model
    
    def get_config(self):
        config = super().get_config()
        config['reflexive'] = self.reflexive
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
