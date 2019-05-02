import tensorflow as tf
import numpy as np

a=np.random.random((3,5))

print(a)

sess=tf.Session()


v1=tf.tile(a,[1,11])

v2=tf.reshape(v1,[3,11,5])

x,y=sess.run([v1,v2])

print(x)
print(y)