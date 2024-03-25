# TensorFlow 라이브러리를 사용합니다.
import tensorflow as tf
# TensorFlow 버전을 출력합니다.
print("TensorFlow version:", tf.__version__)

# MNIST 데이터세트를 로드하고 준비합니다. 
mnist = tf.keras.datasets.mnist

# 샘플 데이터를 정수에서 부동 소수점 숫자로 변환합니다.
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


