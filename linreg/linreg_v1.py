import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def generate_dataset():
  x_batch = np.linspace(0, 10, 100)
  y_batch = 20 + 1.5 * x_batch + np.random.randn(*x_batch.shape)
  return x_batch, y_batch
  
#x_batch, y_batch = generate_dataset()
#fig = plt.scatter(x_batch, y_batch)
#plt.show()

def linear_regression():
  x = tf.placeholder(tf.float32, shape=(None, ), name='x')
  y = tf.placeholder(tf.float32, shape=(None, ), name='y')

  with tf.variable_scope('lreg') as scope:
    alpha = tf.Variable(np.random.normal(), name='alpha')
    beta = tf.Variable(np.random.normal(), name='beta')
		
    y_pred = tf.add(tf.multiply(beta, x), alpha)

    loss = tf.reduce_mean(tf.square(y_pred - y))

  return x, y, y_pred, loss


def run():
  x_batch, y_batch = generate_dataset()
  x, y, y_pred, loss = linear_regression()

  optimizer = tf.train.GradientDescentOptimizer(0.5)
  train_op = optimizer.minimize(loss)

  with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    feed_dict = {x: x_batch, y: y_batch}
		
    for i in range(30):
      session.run(train_op, feed_dict)
      print(i, "loss:", loss.eval(feed_dict))

    print('Predicting')
    y_pred_batch = session.run(y_pred, {x : x_batch})

  plt.scatter(x_batch, y_batch)
  plt.plot(x_batch, y_pred_batch, color='red')
  plt.xlim(0, 2)
  plt.ylim(0, 2)
  plt.savefig('plot.png')

if __name__ == "__main__":
  run()
