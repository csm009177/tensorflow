#apple.py
#! 사과와 오렌지를 구분하는 이진 분류 모델을 만들고 실행하는 예제 코드
import tensorflow as tf
import numpy as np

# 데이터 준비
# 사과: 0, 오렌지: 1
X_train = np.array([[100], [130], [150], [170]])
y_train = np.array([0, 0, 1, 1])

# 모델 구성
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation='sigmoid', input_dim=1)
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 모델 학습
model.fit(X_train, y_train, epochs=100)
#? epochs? 전체 데이터셋에 대해 학습을 반복하는 횟수입니다.

# 새로운 데이터에 대한 예측
X_new = np.array([[110], [160]])
predictions = model.predict(X_new)
print(predictions)

#? 왜 컴파일을 한거지? 모델을 학습하기 전에 컴파일을 해야함