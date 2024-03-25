# TensorFlow 라이브러리를 사용합니다.
import tensorflow as tf
# TensorFlow 버전을 출력합니다.
print("TensorFlow version:", tf.__version__)
# 상수를 정의합니다.
hello = tf.constant('Hello, TensorFlow!')

# hello를 출력합니다.
print(hello)