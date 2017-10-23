from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential([
	Input(2),
    Dense(2),
    Dense(3),
])

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

