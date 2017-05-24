import numpy,sys
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

# load ascii text and covert to lowercase
filename = "pg10657.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()

# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)

# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
X = []
Y = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    X.append([char_to_int[char] for char in seq_in])
    Y.append(char_to_int[seq_out])
n_patterns = len(X)
print("Total Patterns: ", n_patterns)

# reshape X to be [samples, time steps, features]
X = numpy.reshape(X, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(Y)

# define the LSTM model
model = Sequential()
#model.add(LSTM(128, input_shape=(X.shape[1], X.shape[2]),return_sequences=True))
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# define the checkpoint
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit(X, y, epochs=50, batch_size=32,callbacks=callbacks_list)

# load the network weights
#filename = "weights-improvement-93-0.5805.hdf5"
#model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')
# pick a random seed
start = numpy.random.randint(0, len(X)-1)
pattern = X[start]
print("Seed:")
print("\"", ''.join([int_to_char[value*n_vocab] for value in pattern.flatten()]), "\"")

# generate characters
pattern=pattern.flatten().tolist()
for i in range(1000):
    x = numpy.reshape(pattern, (1, len(pattern), 1))
    #x = x / float(n_vocab)
    prediction = model.predict(x, verbose=0)
    #index = numpy.argmax(prediction)
    index = numpy.random.choice(len(prediction[0]),p=prediction[0]) # SAMPLE RANDOMLY WITH PROBABILITIES FROM PREDICTOR
    result = int_to_char[index]
    seq_in = [int_to_char[value*n_vocab] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index/n_vocab)
    pattern = pattern[1:len(pattern)]
print("\nDone.")

