import tensorflow as tf

class CTCLoss(tf.keras.losses.Loss):

    def __init__(self, name="ctc_loss", **kwargs):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        # Compute the training-time loss value
        y_true = tf.cast(y_true, dtype="int64")
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")

        blank_idx = tf.transpose(tf.where(tf.equal(y_true, tf.constant(-2, dtype="int64"))))[1]

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = tf.math.multiply(tf.reshape(blank_idx, shape=(batch_len, 1)), tf.ones(shape=(batch_len, 1), dtype="int64"))

        loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
        return loss