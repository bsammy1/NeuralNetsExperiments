from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np 

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

model = Sequential()
model.add(Dense(8, input_dim=2))
model.add(Activation('sigmoid'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# define the LSTM model
modelLstm = Sequential()
modelLstm.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
modelLstm.add(Dropout(0.2))
modelLstm.add(Dense(y.shape[1], activation='softmax'))

# load the network weights
filename = "weights-improvement-10-1.7558.hdf5"
modelLstm.load_weights(filename)
modelLstm.compile(loss='categorical_crossentropy', optimizer='adam')

# pick a random seed
start = numpy.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print("Seed:")
print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")

sgd = SGD(lr=0.1)
modelLstm.compile(loss='binary_crossentropy', optimizer=sgd)

model.fit(X, y, batch_size=1, nb_epoch=1000)
print(model.predict_proba(X))


# generate characters
for i in range(5):
	x = numpy.reshape(pattern, (1, len(pattern), 1))
	x = x / float(n_vocab)
	prediction = model.predict(x, verbose=0)
	index = numpy.argmax(prediction)
	result = int_to_char[index]
	seq_in = [int_to_char[value] for value in pattern]
	sys.stdout.write(result)
	pattern.append(index)
	pattern = pattern[1:len(pattern)]
print("\nDone.")



#to transform model into coreml
coreml_model = coremltools.converters.keras.convert(model)
coreml_model.save('notesPredictionModel.mlmodel')