import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input
from tensorflow.keras.layers import Dense, Input, Embedding, Dropout, Activation, Conv1D, Softmax
from tensorflow.keras.layers import SpatialDropout1D
from tensorflow.keras.layers import LSTM, GRU, Conv1D, MaxPool1D, Activation

def image_feature_extractor(inputShape, embeddingDim=1024):

	img_input = Input(inputShape)

	model = Sequential()
	model.add(Conv2D(64, (10,10), activation='relu', input_shape=inputShape))
	model.add(MaxPool2D())

	model.add(Conv2D(128, (7,7), activation='relu'))
	model.add(MaxPool2D())

	model.add(Conv2D(128, (4,4), activation='relu'))
	model.add(MaxPool2D())

	model.add(Conv2D(256, (4,4), activation='relu'))
	model.add(Flatten())

	model.add(Dense(2*embeddingDim, activation='relu'))

	model.add(Dense(embeddingDim, activation='relu'))

	emb_output = model(img_input)

	imageModel = Model(inputs=img_input, outputs=emb_output)

	return imageModel

def text_feature_extractor(inputShape, embeddingDim=256):

	# The maximum number of words to be used. (most frequent)
	MAX_NB_WORDS = 50000
	# Max number of words.
	MAX_SEQUENCE_LENGTH = 500
	# This is fixed.
	EMBEDDING_DIM = 100

	text = Input(inputShape)

	model = Sequential()

	model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=inputShape))
	
	model.add(SpatialDropout1D(0.2))
	
	model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
	
	model.add(Dense(embeddingDim, activation='softmax'))

	emb_output = model(text)

	txtModel = Model(inputs=text, outputs=emb_output)

	return txtModel
