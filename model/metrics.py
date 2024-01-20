from jiwer import wer
from model.decoder import NgramCTCDecoder, SpellCTCDecoder
import tensorflow as tf
from generator.align_processing import Align
import numpy as np

def numpy_wer(y_true, y_pred, strings = None) -> float:
  print("chainsaw")
  workers = 8
  sections = []
  section_size = len(y_pred)/workers

  for i in range(workers):
    start = section_size * i
    end = section_size + start
    sections.append(y_pred[int(round(start, 0)):int(round(end, 0))])

  # strings = data["train"].get_strings()

  predictions = SpellCTCDecoder()(sections, workers, strings)

  y_true_str = [Align.number2sentence(elem) for elem in np.int32(y_true)]

  return np.float32(wer(y_true_str, predictions))

class WER(tf.keras.metrics.Metric):  
  def __init__(self, strings):
    print("man")
    self.decoder = NgramCTCDecoder()
    self.strings = strings
    self.valueres = 0
    self.func = lambda y_true, y_pred, strings: tf.numpy_function(numpy_wer, [y_true, y_pred, strings], tf.float32)
    super(WER, self).__init__('wer')

  # @tf.numpy_function(Tout=tf.float32)
  # def __call__(self, y_true, y_pred) -> float:
  def update_state(self, y_true, y_pred, **kwargs):
    tf.print("manga")
    print(tf.shape(y_pred))
    print(tf.shape(y_true))
    
    self.valueres = self.func(y_true, y_pred, self.strings)
    return 0
  
  def result(self, *args, **kwargs):
    return self.valueres
  
class ValidationOnlyMetric(tf.keras.metrics.MeanMetricWrapper):
  def __init__(self, fn, **kwargs):
    super().__init__(fn = lambda y_true, y_pred: tf.numpy_function(numpy_wer, [y_true, y_pred], tf.float32), **kwargs)
    self.on = tf.Variable(False)

  def update_state(self, y_true, y_pred, sample_weight=None):
    tf.print("aaa")
    tf.print(self.on)
    tf.cond(not self.on, lambda : super().update_state(y_true, y_pred, sample_weight), lambda : tf.identity(y_pred))
    # if self.on:
      # tf.print(self.on)
      # print("a")
      # super().update_state(y_true, y_pred, sample_weight)

class ToggleMetrics(tf.keras.callbacks.Callback):
  def __init__(self, name : str, **kwargs):
    super().__init__(**kwargs)
    self.metric_name = "validation_only_metric"

  def on_test_begin(self, *args, **kwargs):
    for metric in self.model.metrics:
      if self.metric_name in metric.name.lower():
        metric.on.assign(True)

  def on_test_end(self, *args, **kwargs):
    for metric in self.model.metrics:
      if self.metric_name in metric.name.lower():
        metric.on.assign(False)
  #   workers = 8
  #   sections = []
  #   section_size = len(y_pred)/workers
  #   # y_pred_np = tf.make_ndarray(y_pred) 

  #   for i in range(workers):
  #     start = section_size * i
  #     end = section_size + start
  #     sections.append(y_pred[round(start, 0):round(end, 0)])

  #   strings = self.data["train"].get_strings()

  #   predictions = self.decoder(sections, workers, strings)

  #   return wer(y_true, predictions)