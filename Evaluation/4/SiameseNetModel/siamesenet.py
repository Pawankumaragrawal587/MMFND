import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input

def image_feature_extractor(inputShape, embeddingDim=1024):

	img_input = Input(inputShape)

	model = Sequential()
	
	model.add(Flatten())

	model.add(Dense(2*embeddingDim, activation='relu', input_shape=inputShape))

	model.add(Dense(embeddingDim, activation='relu'))

	emb_output = model(img_input)

	imageModel = Model(inputs=img_input, outputs=emb_output)

	return imageModel

def text_feature_extractor(inputShape, embeddingDim=1024):

	txt_embeddings = Input(inputShape)

	model = Sequential()

	model.add(Flatten())

	model.add(Dense(2*embeddingDim, activation='relu', input_shape=inputShape))

	model.add(Dense(embeddingDim, activation='relu'))

	emb_output = model(txt_embeddings)

	txtModel = Model(inputs=txt_embeddings, outputs=emb_output)

	return txtModel