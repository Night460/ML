import tensorflow as tf

a = tf.constant([2, 3], dtype=tf.int32)
b = tf.constant([4, 5], dtype=tf.int32)
c = tf.add(a, b)

print("Sum:", c.numpy())
