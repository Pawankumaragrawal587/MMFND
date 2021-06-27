# import the necessary packages
from SiameseNetModel.siamesenet import build_siamese_model
from SiameseNetModel import config
from SiameseNetModel import utils
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from sklearn.model_selection import train_test_split
import numpy as np

# prepare the positive and negative pairs
print("[INFO] preparing positive and negative pairs...")

(img_pair,label) = utils.make_pairs()

pairTrain, pairTest, labelTrain, labelTest = train_test_split(img_pair, label, 
	test_size = 0.25, random_state=42, stratify=label)

# configure the siamese network
print("[INFO] building siamese network...")

imgA = Input(shape=config.IMG_SHAPE)
imgB = Input(shape=config.IMG_SHAPE)

featureExtractor = build_siamese_model(config.IMG_SHAPE)

featsA = featureExtractor(imgA)
featsB = featureExtractor(imgB)

# finally, construct the siamese network
distance = Lambda(utils.euclidean_distance)([featsA, featsB])

outputs = Dense(1, activation="sigmoid")(distance)

model = Model(inputs=[imgA, imgB], outputs=outputs)

# compile the model
print("[INFO] compiling model...")

model.compile(loss="binary_crossentropy", optimizer="adam",metrics=["accuracy"])

# train the model
print("[INFO] training model...")

history = model.fit([pairTrain[:, 0], pairTrain[:, 1]], labelTrain[:],
	validation_data=([pairTest[:, 0], pairTest[:, 1]], labelTest[:]),
	batch_size=config.BATCH_SIZE, 
	epochs=config.EPOCHS)

# serialize the model to disk
print("[INFO] saving siamese model...")
model.save(config.MODEL_PATH)

# plot the training history
print("[INFO] plotting training history...")
utils.plot_training(history, config.PLOT_PATH)

