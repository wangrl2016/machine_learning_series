import tensorflow as tf

if __name__ == '__main__':
    rank_0_tensor = tf.constant(4)
    print(rank_0_tensor)
    rank_1_tensor = tf.constant([2.0, 3.0, 4.0])
    print(rank_1_tensor)
    rank_2_tensor = tf.constant([[1, 2],
                                 [3, 4],
                                 [5, 6]], dtype=tf.float16)
    print(rank_2_tensor)
    rank_3_tensor = tf.constant([
        [[0, 1, 2, 3, 4],
         [5, 6, 7, 8, 9]],
        [[10, 11, 12, 13, 14],
         [15, 16, 17, 18, 19]],
        [[20, 21, 22, 23, 24],
         [25, 26, 27, 28, 29]],])
    print(rank_3_tensor)

    a = tf.constant([[1, 2], [3, 4]])
    b = tf.constant([[6, 7], [8, 9]])
    print(tf.add(a, b))
    print(tf.multiply(a, b))
    print(tf.matmul(a, b))
