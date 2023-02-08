import tensorflow as tf
import numpy as np

lr_rate=1e-4
n_input=2
n_label=1
n_hidden=2

x=tf.placeholder(np.float32,[None,n_input])
y=tf.placeholder(np.float32,[None,n_label])
weights={
    'h1':tf.Variable(tf.truncated_normal([n_input,n_hidden],stddev=0.1)),
    'h2':tf.Variable(tf.truncated_normal([n_hidden,n_label],stddev=0.1))
}
bias={
    'h1':tf.Variable(tf.zeros([n_hidden])),
    'h2':tf.Variable(tf.zeros([n_label])),
}
layer_1=tf.nn.relu(tf.add(tf.matmul(x,weights['h1']),bias['h1']))
y_pred=tf.nn.tanh(tf.add(tf.matmul(layer_1,weights['h2']),bias['h2']))
loss=tf.reduce_mean((y_pred-y)**2)
train_step=tf.train.AdamOptimizer(learning_rate=lr_rate).minimize(loss)

X=[[0,0],[0,1],[1,0],[1,1]]
Y=[[0],[1],[1],[0]]
X=np.array(X).astype(np.float32)
Y=np.array(Y).astype(np.int16)

sess=tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in range(10000):
    sess.run(train_step,feed_dict={x:X,y:Y})
print(sess.run(y_pred,feed_dict={x:X}))
print(sess.run(layer_1,feed_dict={x:X}))