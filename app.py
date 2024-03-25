# TensorFlow 라이브러리를 사용합니다.
import tensorflow as tf
# TensorFlow 버전을 출력합니다.
print("TensorFlow version:", tf.__version__)

#! MNIST 데이터세트를 로드하고 준비합니다. 
# MNIST 데이터세트는 0-9까지의 손으로 쓴 숫자(0-9)의 이미지와 레이블(0-9)로 구성되어 있습니다.
mnist = tf.keras.datasets.mnist


# 샘플 데이터를 정수에서 부동 소수점 숫자로 변환합니다.
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


#! 층을 차례대로 쌓아 tf.keras.Sequential 모델을 만듭니다. 
model = tf.keras.models.Sequential([
  # 옵티마이저는 모델이 손실 함수를 줄이는 방향으로 가중치를 업데이트하는 방법을 결정합니다.
  # 손실 함수는 모델의 출력이 원하는 출력에서 얼마나 떨어져 있는지 측정합니다.
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  # Flatten 층은 28x28 픽셀의 이미지 포맷을 784 픽셀의 1차원 배열로 변환합니다.
  tf.keras.layers.Dense(128, activation='relu'),
  # Dense 층은 노드가 128개인 Dense 층을 추가합니다.
  tf.keras.layers.Dropout(0.2),
  # Dropout 층은 과대적합을 방지하기 위해 추가합니다.
  tf.keras.layers.Dense(10, activation='softmax')
  # 마지막 Dense 층은 10개의 노드를 가진 softmax 층입니다.
])

#! 훈련에 사용할 옵티마이저(optimizer)와 손실 함수를 선택합니다
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 각 예시에서 모델은 각 클래스에 대해 하나씩, logits 또는 log-odds 스코어 벡터를 반환합니다.
predictions = model(x_train[:1]).numpy()
predictions
#? predictions 배열의 각 요소는 해당 클래스에 속할 확률을 나타냅니다.

#! tf.nn.softmax 함수는 다음과 같이 이러한 로짓을 각 클래스에 대한 확률로 변환합니다.
# tf.nn.softmax 함수를 네트워크의 마지막 레이어에 대한 활성화 함수로 베이킹할 수 있습니다. 
# 이렇게 하면 모델 출력을 더 직접적으로 해석할 수 있지만 
# 이 접근법은 소프트맥스 출력을 사용할 경우 모든 모델에 대해 
# 정확하고 수치적으로 안정적인 손실 계산을 제공하는 것이 불가능하므로 권장하지 않습니다.
tf.nn.softmax(predictions).numpy()

#! losses.SparseCategoricalCrossentropy를 사용하여 로짓의 벡터와 True 인덱스를 사용하고 
#! 각 예시에 대해 스칼라 손실을 반환하는 훈련용 손실 함수를 정의합니다.
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)