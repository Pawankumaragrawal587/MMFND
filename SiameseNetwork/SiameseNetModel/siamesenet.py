import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input

def image_feature_extractor(inputShape, embeddingDim=4096):

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

	model.add(Dense(embeddingDim, activation='relu'))

	model.add(Dense(embeddingDim, activation='relu'))

	emb_output = model(img_input)

	imageModel = Model(inputs=img_input, outputs=emb_output)

	return imageModel

def text_feature_extractor(inputShape, embeddingDim=4096):

	txt_embeddings = Input(inputShape)

	model = Sequential()

	model.add(Dense(embeddingDim, activation='relu', input_shape=inputShape))

	model.add(Flatten())

	emb_output = model(txt_embeddings)

	txtModel = Model(inputs=txt_embeddings, outputs=emb_output)

	return txtModel