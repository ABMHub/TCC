import tensorflow as tf
import time
import numpy as np
from generator.batch_generator import BatchGenerator
from model.decoder import SpellCTCDecoder
from jiwer import cer, wer
import pandas as pd
import os

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

class WERCheckpoint(tf.keras.callbacks.Callback):
  def __init__(self, path, model, generator, log_dir):
    self.path = path
    self.model : tf.keras.Model = model
    self.generator : BatchGenerator = generator
    self.decoder = SpellCTCDecoder()
    self.log_writer = tf.summary.create_file_writer(os.path.join(log_dir, "metrics"))

    self.cer_list = []
    self.wer_list = []

    self.best_wer = 2

  def on_epoch_end(self, epoch, logs=None):
    strings = self.generator.get_strings()
    pred = self.model.predict(self.generator)
    decoded = self.decoder(8, pred, strings, greedy = False)

    cer_score = cer(strings, decoded)
    wer_score = wer(strings, decoded)

    self.cer_list.append(cer_score)
    self.wer_list.append(wer_score)

    print(f"WER: {wer_score}", end="")

    if wer_score < self.best_wer:
      self.model.save(self.path)
      self.best_wer = wer_score
      print(" (new best)", end="")

    print("")

    self._write_metric("cer_val", cer_score, epoch)
    self._write_metric("wer_val", wer_score, epoch)

  def on_train_end(self, logs=None):
    pd.DataFrame({
      "epoch": list(range(1, len(self.wer_list)+1)), 
      "cer": self.cer_list,
      "wer": self.wer_list
      }).to_csv(os.path.join(self.path, "wer.csv"), index=False)

  def _write_metric(self, name, value, epoch):
    with self.log_writer.as_default():
      tf.summary.scalar(
        name,
        value,
        step=epoch,
      )
      self.log_writer.flush()
