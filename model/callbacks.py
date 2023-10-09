import tensorflow as tf
import time
import numpy as np

class MinEarlyStopping(tf.keras.callbacks.EarlyStopping):
  def __init__(self, monitor='val_loss', patience=0, min_epoch = 30): # add argument for starting epoch
    super(MinEarlyStopping, self).__init__(monitor=monitor, patience=patience)
    self.min_epoch = min_epoch

  def on_epoch_end(self, epoch, logs=None):
    if epoch > self.min_epoch:
      super().on_epoch_end(epoch, logs)

class TimePerBatch(tf.keras.callbacks.Callback):
    def __init__(self):
        self.batch_time = []
        self.epoch_time = []

    def on_train_batch_begin(self, *args, **kwargs):
        self.start_time = time.time()

    def on_train_batch_end(self, *args, **kwargs):
        self.batch_time.append(time.time() - self.start_time)

    def on_epoch_end(self, epoch, *args, **kwargs):
        self.epoch_time.append(np.array(self.batch_time).mean())
        self.batch_time = []