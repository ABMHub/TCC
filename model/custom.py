import tensorflow as tf
from tensorflow import keras

class CustomModel(keras.Model):
  def train_step(self, data):
    # Unpack the data. Its structure depends on your model and
    # on what you pass to `fit()`.
    x, y = data

    with tf.GradientTape() as tape:
      y_pred = self(x, training=True)  # Forward pass
      # Compute the loss value
      # (the loss function is configured in `compile()`)
      loss = self.compute_loss(y=y, y_pred=y_pred)

    # Compute gradients
    trainable_vars = self.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    # Update weights
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    # Update metrics (includes the metric that tracks the loss)
    for metric in self.metrics:
      tf.print([a.name for a in self.metrics])
      if metric.name == "loss":
        metric.update_state(loss)
      # elif metric.test_only:
        # metric.update_state(y, y_pred)
    # Return a dict mapping metric names to current value
    return {m.name: m.result() for m in self.metrics}
  
  def test_step(self, data):
    # Unpack the data
    x, y = data
    # Compute predictions
    y_pred = self(x, training=False)
    # Updates the metrics tracking the loss
    self.compute_loss(y=y, y_pred=y_pred)
    # Update the metrics.
    print([a.name for a in self.metrics])
    for metric in self.metrics:
      print(metric.name)  
      if metric.name != "loss":
        metric.update_state(y, y_pred)
    # Return a dict mapping metric names to current value.
    # Note that it will include the loss (tracked in self.metrics).
    return {m.name: m.result() for m in self.metrics}