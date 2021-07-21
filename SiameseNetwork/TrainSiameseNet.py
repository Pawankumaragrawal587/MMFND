import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "7"  # specify which GPU(s) to be used

# import the necessary packages
from SiameseNetModel.siamesenet import image_feature_extractor, text_feature_extractor
from SiameseNetModel import config
from SiameseNetModel import utils
import tensorflow
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.optimizers import Adam

# prepare the positive and negative pairs
print("[INFO] preparing positive and negative pairs...")

(img_pair, txt_pair, label) = utils.make_pairs()

print('Img Shape: ', img_pair.shape)
print('Text Shape: ', txt_pair.shape)
print('Label Shape: ', label.shape)

txtTrain, txtTest, imgTrain, imgTest, labelTrain, labelTest = train_test_split(txt_pair, img_pair, label, 
	test_size = 0.25, random_state=42)


# configure the siamese network
print("[INFO] building siamese network...")

imgA = Input(shape=config.IMG_SHAPE, name='ImageT')
imgB = Input(shape=config.IMG_SHAPE, name='ImageS')
textA = Input(shape=config.TXT_SHAPE, name='TextT')
textB = Input(shape=config.TXT_SHAPE, name='TextS')

featureExtractor_img = image_feature_extractor(config.IMG_SHAPE)
featureExtractor_txt = text_feature_extractor(config.TXT_SHAPE)

imgfeatsA = featureExtractor_img(imgA)
imgfeatsB = featureExtractor_img(imgB)

txtfeatsA = featureExtractor_txt(textA)
txtfeatsB = featureExtractor_txt(textB)

concatanatedA = tensorflow.keras.layers.Concatenate()([imgfeatsA,txtfeatsA])
concatanatedB = tensorflow.keras.layers.Concatenate()([imgfeatsB,txtfeatsB])

# finally, construct the siamese network
Dis_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
distance = Dis_layer([concatanatedA, concatanatedB])

outputs = Dense(1, activation="sigmoid")(distance)

model = Model(inputs=[imgA, imgB, textA, textB], outputs=outputs)

print(model.summary())

# compile the model
print("[INFO] compiling model...")

metricss = [
	tensorflow.keras.metrics.FalseNegatives(name="fn"),
	tensorflow.keras.metrics.FalsePositives(name="fp"),
	tensorflow.keras.metrics.TrueNegatives(name="tn"),
	tensorflow.keras.metrics.TruePositives(name="tp"),
	"acc",
	tensorflow.keras.metrics.Precision(name="precision"),
	tensorflow.keras.metrics.Recall(name="recall"),
]

# optimizer = Adam(lr = 0.00006)

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=metricss)

# train the model
print("[INFO] training model...")

history = model.fit([imgTrain[:, 0], imgTrain[:, 1], txtTrain[:, 0], txtTrain[:, 1]], labelTrain[:],
	validation_data=([imgTest[:, 0], imgTest[:, 1], txtTest[:, 0], txtTest[:, 1]], labelTest[:]),
	batch_size=config.BATCH_SIZE, 
	epochs=config.EPOCHS)


# serialize the model to disk
print("[INFO] saving siamese model...")
model.save(config.MODEL_PATH)

# plot the training history
print("[INFO] plotting training history...")
utils.plot_training(history, config.PLOT_PATH)

