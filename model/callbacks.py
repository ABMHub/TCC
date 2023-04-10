import tensorflow as tf

class MinEarlyStopping(tf.keras.callbacks.EarlyStopping):
  def __init__(self, monitor='val_loss', patience=0, min_epoch = 30): # add argument for starting epoch
    super(MinEarlyStopping, self).__init__(monitor=monitor, patience=patience)
    self.min_epoch = min_epoch

  def on_epoch_end(self, epoch, logs=None):
    if epoch > self.min_epoch:
      super().on_epoch_end(epoch, logs)