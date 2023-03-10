
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

# Add attention layer to the deep learning network
class CascadedAttention(tf.keras.layers.Layer):
    def __init__(self, vocab_size, **kwargs):
        super(CascadedAttention, self).__init__(**kwargs)
        self.vocab_size = vocab_size
 
    def build(self, input_shape): # 75x1024
        self.frame_count = input_shape[-2]
        # print("input_shape", input_shape)
        
        self.att = CascadedAttentionCell(28)
        self.att.build(input_shape)

        self.gru = tf.keras.layers.GRUCell(28)
        self.gru.build(1024)

        # self.Emb=self.add_weight(name='attention_gru_weight',            shape=(28, 28), initializer='random_normal', trainable=True)
        # self.Emb = tf.keras.layers.Embedding(28, 28, input_length=1)
        # self.Emb.build(28)
        super(CascadedAttention, self).build(input_shape)
 
    def call(self, x):
        batch_size = tf.shape(x)[0]
        # x = tf.transpose(x, [1, 0, 2])
        Yanterior = tf.expand_dims(self.gru.get_initial_state(batch_size=batch_size, dtype="float32"), 1)
        output = []
        for _ in range(self.frame_count):
            context_vector = self.att(x, Yanterior)
            Yanterior = self.gru(context_vector, Yanterior)[0]
            output.append(Yanterior)

        output = tf.convert_to_tensor(output, dtype=tf.float32)
        output = tf.squeeze(output, -2)
        output = tf.transpose(output, [1, 0, 2])
        print("output", output.shape)
        return output
    
    def get_config(self):
        config = super().get_config()
        config['vocab_size'] = self.vocab_size
        config['frame_count'] = self.frame_count
        return config
    
class CascadedAttentionCell(tf.keras.layers.Layer):
    def __init__(self, vocab_size, **kwargs):
        super(CascadedAttentionCell, self).__init__(**kwargs)
        self.vocab_size = vocab_size
 
    def build(self, input_shape): # 75x1024
        feature_count = input_shape[-1]
        # print("input_shape", input_shape)
        self.Wa  = self.add_weight(name='recurrent_attention_cell_weight',  shape=(self.vocab_size, feature_count), initializer=tf.keras.initializers.GlorotNormal(), trainable=True)
        self.Ua  = self.add_weight(name='attention_cell_weight',            shape=(feature_count, feature_count),   initializer=tf.keras.initializers.GlorotNormal(), trainable=True)
        self.Va  = self.add_weight(name='attention_cell_score_weight',      shape=(feature_count, 1),               initializer=tf.keras.initializers.GlorotNormal(), trainable=True)
        self.Ba1 = self.add_weight(name='attention_cell_bias1',             shape=(1, feature_count),               initializer=tf.keras.initializers.GlorotNormal(), trainable=True)
        self.Ba2 = self.add_weight(name='attention_cell_bias2',             shape=(1, feature_count),               initializer=tf.keras.initializers.GlorotNormal(), trainable=True)
        self.Ba3 = self.add_weight(name='attention_cell_bias3',             shape=(1, 1),                           initializer=tf.keras.initializers.GlorotNormal(), trainable=True)

        super(CascadedAttentionCell, self).build(input_shape)

    def call(self, x, prev_prediction):
        WaS = tf.matmul(prev_prediction, self.Wa) + self.Ba1
        UaH = tf.matmul(x, self.Ua) + self.Ba2

        scores = tf.matmul(K.tanh(UaH + WaS), self.Va) + self.Ba3
        scores = K.squeeze(scores, axis=-1)

        sm = K.softmax(scores)                   
        sm = K.expand_dims(sm, axis=-1)
        
        context_vector = K.sum(x * sm, axis=1)

        context_vector = K.expand_dims(context_vector, axis=1)
        return context_vector
