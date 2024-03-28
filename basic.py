# /usr/bin/env python3

#! 텐서플로우 기본 사용법
import tensorflow as tf
#? numpy? 수학적인 연산을 위한 라이브러리
import numpy as np
#? scipy? 과학기술 계산을 위한 라이브러리
from scipy.interpolate import interp1d
#? interp1d? 1차원 보간법을 사용하여 주어진 데이터를 보간하는 함수

# 텐서플로우 버전 확인
print("TensorFlow version:", tf.__version__)

# 분류를 위한 데이터 생성
#? 분류? 데이터를 여러 그룹으로 나누는 것
#? 왜 분류를 하는가? 데이터의 특징을 파악하고, 그룹을 나누어 분석하기 위함
X_classification = tf.convert_to_tensor(np.random.randn(1000, 20), dtype=tf.float32)
#? convert_to_tensor? numpy 배열을 텐서로 변환하는 함수 (텐서플로우에서 사용하는 데이터 형식)
y_classification = tf.convert_to_tensor(np.random.randint(0, 2, size=(1000,)), dtype=tf.int32)
#? dtype? 데이터의 타입을 지정하는 매개변수

# 회귀를 위한 데이터 생성
#? 회귀? 데이터를 통해 연속적인 값을 예측하는 것
#? 분류와 차이점? 분류는 미리 정해진 그룹에 데이터를 할당하는 것, 회귀는 연속적인 값을 예측하는 것
X_regression = tf.convert_to_tensor(np.random.randn(1000, 20), dtype=tf.float32)
#? random.randn(?,?)? 정규분포를 따르는 랜덤한 값을 생성하는 함수
#? 정규분포를 따르는? 데이터의 분포가 정규분포와 비슷한 형태를 띄는 것
y_regression = tf.convert_to_tensor(np.random.randn(1000,), dtype=tf.float32)


# 군집화를 위한 데이터 생성
#? 군집화? 데이터를 여러 그룹으로 나누는 것
#? 분류와 차이점? 분류는 미리 정해진 그룹에 데이터를 할당하는 것, 군집화는 미리 정해진 그룹이 없는 데이터를 그룹화하는 것
X_clustering = tf.convert_to_tensor(np.random.randn(1000, 20), dtype=tf.float32)

# 차원 축소를 위한 PCA 적용
#? 차원 축소? 데이터의 차원을 줄이는 것
#? PCA? 주성분 분석, 데이터의 차원을 줄이는 기법
#? 왜 차원을 줄이는가? 데이터의 특징을 잘 나타내는 주성분을 찾아내기 위함
#? 어떻게 차원을 줄이는가? 데이터의 분산을 최대한 보존하는 방향으로 차원을 줄임
pca = tf.keras.layers.Dense(2)
X_pca = pca(X_clustering)

# 유사도 계산 (코사인 유사도)
#? 유사도? 두 데이터가 얼마나 비슷한지를 나타내는 값
#? 코사인? 두 벡터 사이의 각도를 이용하여 유사도를 계산하는 방법
cosine_similarity = tf.keras.losses.cosine_similarity(X_clustering[:1], X_clustering).numpy()

# 보간을 위한 데이터 생성
#? 보간? 주어진 데이터를 기반으로 주어진 구간에서의 값을 추정하는 것
x_interp = np.linspace(0, 10, 10)
#? linspace? 주어진 구간에서 일정한 간격으로 값을 생성하는 함수
y_interp = np.sin(x_interp)
f = interp1d(x_interp, y_interp, kind='linear')

# 근사를 위한 다항식 학습
poly_coeffs = np.polyfit(x_interp, y_interp, 3)

print("Classification data shape:", X_classification.shape, y_classification.shape)
print("Regression data shape:", X_regression.shape, y_regression.shape)
print("Clustering data shape:", X_clustering.shape)
print("PCA data shape:", X_pca.shape)
print("Cosine similarity:", cosine_similarity)
print("Interpolated value at x=2:", f(2))
print("Polynomial coefficients:", poly_coeffs)

#? 이 코드에서 컴파일 하는 부분이 어디인가? 없음
#? 컴파일을 하는 경우와 아닌경우는 어떤 차이가 있는가? 
# 컴파일을 하는 경우에는 코드를 실행하기 전에 코드를 미리 변환하는 작업을 수행함
#? 모델을 학습하기 전에 컴파일을 해야하는것 아닌가? 모델을 학습하기 전에 컴파일을 해야함
#? 그러면 왜 여기서는 하지 않았지? 모델을 학습하는 코드가 없음