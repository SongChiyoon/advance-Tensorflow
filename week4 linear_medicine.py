# Logistic Regression Classifier for diabetes#
# code by song
import tensorflow as tf
import numpy as np
import csv
tf.set_random_seed(777)  # for reproducibility

xy = np.loadtxt('data-03-diabetes.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]


with open("train.csv", 'r') as f:
    train = list(csv.reader(f, delimiter=","))

train = train[1:]

#index
isex = 4
iage = 5
isib = 6
iparch = 7
ifare = 9

#for index in xrange()
train = np.array(train[1:], dtype=None)

age = train[:,iage]
sib = train[:,isib]
parch = train[:,iparch]
fare = train[:,ifare]
ageMean = np.mean(np.float32(age[age[:] != ""]))
sibMean = np.mean(np.float32(sib[sib[:] != ""]))
parchMean = np.mean(np.float32(parch[parch[:] != ""]))
fareMean = np.mean(np.float32(fare[fare[:] != ""]))

for index in xrange(train.shape[0]):
    if train[index][isex] == 'male':
        train[index][isex] = 0
    else:
        train[index][isex] = 1
    if train[index][iage] == "":
        train[index][iage] = ageMean
    if train[index][isib] == "":
        train[index][isib] = sibMean
    if train[index][iparch] == "":
        train[index][iparch] = parchMean
    if train[index][ifare] == "":
        train[index][ifare] = fareMean


train_x = np.array(train[:,[2,4,5,6,7,9]], dtype=float)
train_y = np.array(train[:,4],dtype=float)
train_y = np.reshape(train_y, (-1,1))


print x_data[:20]
print train_x[:20]
# placeholders for a tensor that will be always fed.

X = tf.placeholder(tf.float32, shape=[None,6])
Y = tf.placeholder(tf.float32, shape=[None,1])

w1 = tf.Variable(tf.random_normal([6,8]), name='weight1')
b1 = tf.Variable(tf.random_normal([8]), name = 'bias1')
layer1 = tf.sigmoid(tf.matmul(X,w1)+b1)
w2 = tf.Variable(tf.random_normal([8,6]), name='weight2')
b2 = tf.Variable(tf.random_normal([6]), name = 'bias2')
layer2 = tf.sigmoid(tf.matmul(layer1,w2)+b2)
w3 = tf.Variable(tf.random_normal([6,1]), name='weight3')
b3 = tf.Variable(tf.random_normal([1]), name = 'bias3')
hypothesis = tf.sigmoid(tf.matmul(layer2,w3)+b3)

# cost/loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *
                       tf.log(1 - hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: train_x, Y: train_y})
        if step % 200 == 0:
            print(step, cost_val)

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                       feed_dict={X: train_x, Y: train_y})
    print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)
