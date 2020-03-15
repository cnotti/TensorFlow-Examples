import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def generate_data():
  x_batch = np.linspace(0, 10, 100)
  y_batch = 20 + 1.5 * x_batch + np.random.randn(*x_batch.shape)
  return x_batch, y_batch

alpha = tf.Variable(np.random.normal(), name='alpha')
beta = tf.Variable(np.random.normal(), name='beta')
opt = tf.optimizers.Adam(0.1)

def model(x):
  return alpha + beta * x

def compute_loss(y, yhat):
  return tf.reduce_mean(tf.square(yhat - y))


def train():
  x, y = generate_data()
  
  def _loss_fn():
    yhat = model(x)
    loss = compute_loss(y, yhat)
    return loss
  
  opt.minimize(_loss_fn, [alpha, beta])


for _ in range(1000):
  train()

print(alpha.numpy())
print(beta.numpy())
