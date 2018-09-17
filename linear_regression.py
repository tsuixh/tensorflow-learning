import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

train_x = np.linspace(-1, 1, 100)
train_y = 2 * train_x + np.random.randn(*train_x.shape) * 0.3  # y = 2x, with noise plus

# show simulated data point
plt.plot(train_x, train_y, 'ro', label='original data')
plt.legend()
plt.show()

# create model
# placeholder
X = tf.placeholder('float')
Y = tf.placeholder('float')
# model parameter
w = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.zeros([1]), name='bias')
# forward structure
z = tf.multiply(X, w) + b

# reverse optimize
cost = tf.reduce_mean(tf.square(Y - z))
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# init all variables
init = tf.global_variables_initializer()
# define parameter
training_epochs = 20
display_step = 2

plotdata = {'batchsize': [], 'loss': []}


def moving_average(a, w=10):
    if len(a) < w:
        return a[:]
    return [val if idx < w else sum(a[(idx - w): idx]) / w for idx, val in enumerate(a)]


# start session
with tf.Session() as sess:
    sess.run(init)
    plotdata = {'batchsize': [], 'loss': []}  # store serial value and lose value
    # input data into model
    for epoch in range(training_epochs):
        for (x, y) in zip(train_x, train_y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

            # show detail info in train process
            if epoch % display_step == 0:
                loss = sess.run(cost, feed_dict={X: train_x, Y: train_y})
                print('Epoch', epoch + 1, 'cost=', loss, 'W=', sess.run(w),
                      'b=', sess.run(b))
                if not (loss == 'NA'):
                    plotdata['batchsize'].append(epoch)
                    plotdata['loss'].append(loss)

    print('Finished!')
    print('cost=', sess.run(cost, feed_dict={X: train_x, Y: train_y}), 'W=', sess.run(w), 'b=', sess.run(b))

    plt.plot(train_x, train_y, 'ro', label='orignal data')
    plt.plot(train_x, sess.run(w) * train_x + sess.run(b), label='fitted line')
    plt.legend()
    plt.show()

    plotdata['avgloss'] = moving_average(plotdata['loss'])
    plt.figure(1)
    plt.subplot(211)
    plt.plot(plotdata['batchsize'], plotdata['avgloss'], 'b--')
    plt.xlabel('minibatch number')
    plt.ylabel('loss')
    plt.title('minibatch run vs training loss')

    plt.show()

    print('x=0.2, z=', sess.run(z, feed_dict={X: 0.2}))