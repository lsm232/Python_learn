w=tf.Variable(tf.random_normal([2,1]),name='weight')
b=tf.Variable(tf.zeros([1]),name='bias')
pred=tf.matmul(input_features,w)+b
out=tf.nn.sigmoid(pred)
loss=tf.reduce_mean(-(input_labels*tf.log(out)+(1-input_labels)*tf.log(1-out)))
optimizer=tf.train.AdamOptimizer(0.002).minimize(loss)