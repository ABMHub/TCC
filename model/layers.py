
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
        self.batch_size = input_shape[0]
        # print("input_shape", input_shape)
        self.Wa=self.add_weight(name='recurrent_attention_weight',      shape=(28, 1), initializer='random_normal', trainable=True)
        self.Ua=self.add_weight(name='attention_weight',                shape=(1024, 1), initializer='random_normal', trainable=True)
        self.Va=self.add_weight(name='score_weight',                    shape=(1, 1), initializer='random_normal', trainable=True)

        self.Wo=self.add_weight(name='embedding_attention_gru_weight',  shape=(28, 1), initializer='random_normal', trainable=True)
        self.Uo=self.add_weight(name='recurrent_attention_gru_weight',  shape=(1024, 28), initializer='random_normal', trainable=True)
        self.Co=self.add_weight(name='attention_gru_weight',            shape=(1024, 28), initializer='random_normal', trainable=True)
        # self.Emb=self.add_weight(name='attention_gru_weight',            shape=(28, 28), initializer='random_normal', trainable=True)
        self.Emb = tf.keras.layers.Embedding(28, 28, input_length=28)
        self.Emb.build(28)
        super(CascadedAttention, self).build(input_shape)
 
    def call(self, x):
        batch_size = tf.shape(x)[0]
        # x = tf.transpose(x, [1, 0, 2])
        Yanterior = tf.zeros([batch_size, 1, 28])
        Hanterior = tf.zeros([batch_size, 1, 1024])
        output = []
        for t in range(self.frame_count):
            Yanterior = CascadedAttentionIteration(batch_size, Yanterior, Hanterior, self.Wa, self.Ua, self.Va, x, self.frame_count, self.Wo, self.Uo, self.Co, self.Emb)
            Hanterior = K.expand_dims(tf.transpose(x, [1, 0, 2])[t], axis=1)
            output.append(Yanterior)

        output = tf.convert_to_tensor(output, dtype=tf.float32)
        output = tf.squeeze(output, -2)
        output = tf.transpose(output, [1, 0, 2])
        return output
    
@tf.function
def CascadedAttentionIteration(batch_size, Yanterior, Hanterior, Wa, Ua, Va, x, frame_count, Wo, Uo, Co, emb):
    WaS = tf.matmul(Yanterior, Wa)              # [b, 1, 1]
    UaH = tf.matmul(x, Ua)                      # [b, 75, 1]
    print("WaS", WaS.shape)
    print("UaH", UaH.shape)
    scores = tf.matmul(K.tanh(UaH + WaS), Va)   # [b, 75, 1, 1]
    
    scores = K.softmax(scores)                   
    scores = tf.transpose(scores, [1, 0, 2])                         # [75, b, 1]
    xtemp = tf.transpose(x, [1, 0, 2])         # [75, b, 1024]
    print("scores", scores.shape)                       # [b, 75, 1, 1]
    print("xtemp", xtemp.shape)                       # [b, 75, 1, 1]

    xtemp = tf.expand_dims(xtemp, -1)
    scores = tf.expand_dims(scores, -1)

    summ = tf.zeros([batch_size, 1024, 1])              # [b, 1024, 1]
    for j in range(frame_count):       
        summ = summ + (xtemp[j] * scores[j])    # [batch, 1024, 1]
    print("summ", summ.shape)

    c = tf.transpose(summ, [0, 2, 1])   # [batch, 1, 1024]

    # emb = tf.keras.layers.Embedding(28, 28)

    # emb_in = tf.squeeze(Yanterior, 1)
    # emb_res = emb(emb_in)

    emb_res = emb(Yanterior)
    WoE = tf.matmul(emb_res, Wo)
    UoH = tf.matmul(Hanterior, Uo)
    CoC = tf.matmul(c, Co)

    WoE = tf.squeeze(WoE, -1)

    y = K.sigmoid(WoE + UoH + CoC)
    print("y", y.shape)
    
    return y