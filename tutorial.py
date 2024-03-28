# app.py
#! tensorflow 튜토리얼
# TensorFlow 라이브러리를 사용합니다.
import tensorflow as tf
# TensorFlow 버전을 출력합니다.
print("TensorFlow version:", tf.__version__)

# MNIST 데이터세트를 로드하고 준비합니다. 
mnist = tf.keras.datasets.mnist
#? MNIST 데이터세트는 0-9까지의 손으로 쓴 
#? 숫자(0-9)의 이미지와 레이블(0-9)로 구성되어 있습니다.

# 샘플 데이터를 정수에서 부동 소수점 숫자로 변환합니다.
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
#? x_train은 학습 데이터, 
#? y_train은 학습 데이터의 레이블, 
#? x_test는 테스트 데이터, 
#? y_test는 테스트 데이터의 레이블입니다.
#? 레이블? 데이터의 정답을 의미합니다.
#? 테스트 데이터? 모델이 학습하지 않은 데이터를 의미합니다.
#? 학습 데이터? 모델이 학습하는 데이터를 의미합니다.

# CSV 파일에서 데이터 로드
# file_path = 'path_to_your_csv_file.csv'
# data = tf.data.experimental.make_csv_dataset(file_path, batch_size=32)
#? make_csv_dataset? CSV 파일에서 데이터세트를 만드는 함수
#? batch_size? 배치 크기를 지정하는 매개변수

# layers를 차례대로 쌓아 tf.keras.Sequential 모델을 만듭니다.
#? 층이 뭔가? 신경망의 구성 요소로, 입력 데이터에서 특성을 추출하는 역할을 합니다.
model = tf.keras.models.Sequential([
  #? Sequential 모델? 층을 차례대로 쌓아 만드는 모델
  # Flatten 층은 28x28 픽셀의 이미지 포맷을 784 픽셀의 1차원 배열로 변환합니다.
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  #? Flatten을 하는 이유? 이미지 데이터를 1차원 배열로 변환하여 신경망에 입력하기 위함
  #? 1차원 배열이라면 선을 말하는 것인가? 1차원 배열은 선이라기보다는 벡터라고 부릅니다.
  #? 벡터와 선의 차이? 벡터는 크기와 방향을 가지는 양, 선은 방향만 가지는 양
  # Dense 층은 노드가 128개인 Dense 층을 추가합니다.
  tf.keras.layers.Dense(128, activation='relu'),
  #? Dense? 완전 연결 층, 층의 모든 노드가 이전 층의 모든 노드와 연결되어 있는 층
  #? 완전 연결 층? 층의 모든 노드가 이전 층의 모든 노드와 연결되어 있는 층
  #? Dense 단어 뜻? 밀집한, 조밀한, 밀집한 층이라는 뜻
  tf.keras.layers.Dropout(0.2),
  #? Dropout 층은 과대적합을 방지하기 위해 추가합니다.
  tf.keras.layers.Dense(10, activation='softmax')
  #? 마지막 Dense 층은 10개의 노드를 가진 softmax 층입니다.
])

# 모델을 컴파일합니다.
# 훈련에 사용할 옵티마이저(optimizer)와 손실 함수를 선택합니다
model.compile(optimizer='adam',
  #? 컴파일을 왜 할까? 모델을 훈련하기 전에 설정이 필요한 몇 가지 추가적인 매개변수를 모델에 추가합니다.
  #? 옵티마이저? 모델이 손실 함수를 최소화하기 위해 가중치를 업데이트하는 방법
              loss='sparse_categorical_crossentropy',
  #? 손실 함수? 모델의 예측이 실제 값과 얼마나 일치하는지 측정하는 함수
              metrics=['accuracy'])
  #? metrics? 모델을 평가하는 데 사용되는 지표

# 각 예시에서 모델은 각 클래스에 대해 하나씩, logits 또는 log-odds 스코어 벡터를 반환합니다.
predictions = model(x_train[:1]).numpy()
#? predictions 배열의 각 요소는 해당 클래스에 속할 확률을 나타냅니다.
predictions

# tf.nn.softmax 함수는 다음과 같이 이러한 로짓을 각 클래스에 대한 확률로 변환합니다.
# tf.nn.softmax 함수를 네트워크의 마지막 레이어에 대한 활성화 함수로 베이킹할 수 있습니다. 
# 이렇게 하면 모델 출력을 더 직접적으로 해석할 수 있지만 
# 이 접근법은 소프트맥스 출력을 사용할 경우 모든 모델에 대해 
# 정확하고 수치적으로 안정적인 손실 계산을 제공하는 것이 불가능하므로 권장하지 않습니다.
tf.nn.softmax(predictions).numpy()

# losses.SparseCategoricalCrossentropy를 사용하여 로짓의 벡터와 True 인덱스를 사용하고 
# 각 예시에 대해 스칼라 손실을 반환하는 훈련용 손실 함수를 정의합니다.
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# 이 손실은 실제 클래스의 음의 로그 확률과 같습니다. 모델이 올바른 클래스를 확신하는 경우 손실은 0입니다.
# 이 훈련되지 않은 모델은 무작위에 가까운 확률(각 클래스에 대해 1/10)을 
# 제공하므로 초기 손실은 -tf.math.log(1/10) ~= 2.3에 근접해야 합니다.
loss_fn(y_train[:1], predictions).numpy()

# 훈련을 시작하기 전에 Keras Model.compile을 사용하여 모델을 구성하고 컴파일합니다. 
# optimizer 클래스를 adam으로 설정하고 loss를 앞에서 정의한 loss_fn 함수로 설정합니다. 
# metrics 매개변수를 accuracy로 설정하여 모델에 대해 평가할 메트릭을 지정합니다.
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# 모델을 훈련하고 평가합니다
model.fit(x_train, y_train, epochs=5)
#? model.fit? 모델을 학습시키는 함수
# Model.evaluate 메서드는 일반적으로 "Validation-set" 또는 "Test-set"에서 모델 성능을 확인합니다.
model.evaluate(x_test,  y_test, verbose=2)
#? model.evaluate? 모델을 평가하는 함수
# 훈련된 이미지 분류기는 이 데이터셋에서 약 98%의 정확도를 달성합니다. 더 자세한 내용은 TensorFlow 튜토리얼을 참고하세요.

# 모델이 확률을 반환하도록 하려면 다음과 같이 훈련된 모델을 래핑하고 여기에 소프트맥스를 첨부할 수 있습니다.
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

# 이제 이 모델은 로그-확률을 반환하며, 이 로그-확률은 tf.nn.softmax를 통해 확률로 변환됩니다.
probability_model(x_test[:5])

# 축하합니다! Keras API를 사용하는 사전에 빌드한 데이터세트를 사용하여 머신 러닝 모델을 훈련했습니다.