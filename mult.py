import numpy as np
import tensorflow as tf

# 입력 정의
a = tf.keras.Input(shape=(), dtype=tf.float32)
b = tf.keras.Input(shape=(), dtype=tf.float32)

# 곱셈 연산 정의 (Keras 레이어 사용)
multiply_layer = tf.keras.layers.Multiply()
y = multiply_layer([a, b])

# 모델 생성
multiply_model = tf.keras.Model(inputs=[a, b], outputs=y)

# 입력 데이터 제공 및 모델 실행
input_data_a = np.array([3.0])  # 첫 번째 입력 데이터
input_data_b = np.array([3.0])  # 두 번째 입력 데이터
result = multiply_model.predict([input_data_a, input_data_b])
print(result)
