import tensorflow as tf

var1=tf.Variable(1.0,name='firstvar')
print('var1:',var1.name)
var1=tf.Variable(2.0,name='firstvar')
print('var1:',var1.name)
v=1