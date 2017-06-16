import tensorflow as tf

x = [1,2,3]
y = [1,2,3]

w = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

hypothesis = x * w + b

cost = tf.reduce_mean(tf.square(hypothesis - y))

# minimize optimize

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
#initialize global variables
sess.run(tf.initialize_all_variables())

for step in range(2001):
    if step % 20 == 0:
        print step, sess.run(cost), sess.run(w), sess.run(b)