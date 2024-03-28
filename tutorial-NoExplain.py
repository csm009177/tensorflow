
#! 주석 없는 tutorial.py 코드
import tensorflow as tf

print("TensorFlow 버전:", tf.__version__)

mnist = tf.keras.datasets.mnist

#? MNIST 데이터세트는 0-9까지의 손으로 쓴 
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#? x_train은 학습 데이터, 
#? y_train은 학습 데이터의 레이블,
#? x_test는 테스트 데이터,
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

predictions = model(x_train[:1]).numpy()

tf.nn.softmax(predictions).numpy()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn(y_train[:1], predictions).numpy()

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)

model.evaluate(x_test,  y_test, verbose=2)

probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

probability_model(x_test[:5])
