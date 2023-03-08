
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
        self.Wa=self.add_weight(name='recurrent_attention_weight',      shape=(1, 28), initializer='random_normal', trainable=True)
        self.Ua=self.add_weight(name='attention_weight',                shape=(1, 1024), initializer='random_normal', trainable=True)
        self.Va=self.add_weight(name='score_weight',                    shape=(1, 1), initializer='random_normal', trainable=True)

        self.Wo=self.add_weight(name='embedding_attention_gru_weight',  shape=(28, 28), initializer='random_normal', trainable=True)
        self.Uo=self.add_weight(name='recurrent_attention_gru_weight',  shape=(28, 1024), initializer='random_normal', trainable=True)
        self.Co=self.add_weight(name='attention_gru_weight',            shape=(28, 1024), initializer='random_normal', trainable=True)
        super(CascadedAttention, self).build(input_shape)
 
    def call(self, x):
        x = x[0]
        Yanterior = tf.zeros([28, 1])
        Hanterior = tf.zeros([1024, 1])
        output = []
        for t in range(self.frame_count):
            # x = vetor de estados da bigru
            scores = []
            # scores = tf.zeros([0, ])
            # prev_score = K.dot(anterior, self.Wa)
            prev_score = K.dot(self.Wa, Yanterior)
            for j in range(self.frame_count):
                score = K.expand_dims(x[j], axis=-1)
                score = K.dot(self.Ua, score)
                score = score + prev_score
                score = K.tanh(score)
                score = K.dot(self.Va, tf.transpose(score))
                scores.append(score)

            score = None
            scores = tf.convert_to_tensor(scores, dtype=tf.float32)
            scores = K.softmax(scores)

            xtemp = K.expand_dims(x, axis=-1)

            summ = tf.zeros([1024, 1])
            for j in range(self.frame_count):
                summ = summ + xtemp[j]*scores[j]

            WoE = K.dot(self.Wo, Yanterior)
            UoH = K.dot(self.Uo, Hanterior)
            CoC = K.dot(self.Co, summ)

            y = K.sigmoid(WoE + UoH + CoC)
            
            output.append(y)
            Yanterior = y
            Hanterior = K.expand_dims(x[t], axis=-1)

        return K.expand_dims(tf.squeeze(tf.convert_to_tensor(output, dtype=tf.float32), -1), 0)



        # K.dot()
        # e = K.tanh(K.dot(x,self.Wa)+self.b)
        # # Remove dimension of size 1
        # e = K.squeeze(e, axis=-1)   
        # # Compute the weights
        # alpha = K.softmax(e)
        # # Reshape to tensorFlow format
        # alpha = K.expand_dims(alpha, axis=-1)
        # # Compute the context vector
        # context = x * alpha
        # context = K.sum(context, axis=1)
        import numpy as np
        return tf.constant(np.zeros((1, 75, 28)))