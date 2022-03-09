import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0"  # specify which GPU(s) to be used

# import the necessary packages
from SiameseNetModel.siamesenet import image_feature_extractor, text_feature_extractor
from SiameseNetModel import config
from SiameseNetModel import utils
from SiameseNetModel import metrics
import tensorflow
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.optimizers import Adam
import pandas as pd

# prepare the positive and negative pairs
print("[INFO] preparing positive and negative pairs...")

(img_pair, txt_pair, label) = utils.make_pairs()

print('Img Shape: ', img_pair.shape)
print('Text Shape: ', txt_pair.shape)
print('Label Shape: ', label.shape)

txtTrain, txtTest, imgTrain, imgTest, labelTrain, labelTest = train_test_split(txt_pair, img_pair, label, 
	test_size = 0.2, random_state=42, stratify=label)


# configure the siamese network
print("[INFO] building siamese network...")

imgA = Input(shape=config.IMG_SHAPE, name='ImageT')
imgB = Input(shape=config.IMG_SHAPE, name='ImageS')
textA = Input(shape=config.TXT_SHAPE, name='TextT')
textB = Input(shape=config.TXT_SHAPE, name='TextS')

featureExtractor_img1 = image_feature_extractor(config.IMG_SHAPE)
featureExtractor_txt1 = text_feature_extractor(config.TXT_SHAPE)
featureExtractor_img2 = image_feature_extractor(config.IMG_SHAPE)
featureExtractor_txt2 = text_feature_extractor(config.TXT_SHAPE)

imgfeatsA = featureExtractor_img1(imgA)
imgfeatsB = featureExtractor_img2(imgB)

txtfeatsA = featureExtractor_txt1(textA)
txtfeatsB = featureExtractor_txt2(textB)


concatanatedA = tensorflow.keras.layers.Concatenate()([imgfeatsA,txtfeatsA])
concatanatedB = tensorflow.keras.layers.Concatenate()([imgfeatsB,txtfeatsB])

# finally, construct the siamese network

Dis_layer1 = Lambda(lambda tensors:K.abs(tensors[0] + tensors[1]))
distance1 = Dis_layer1([concatanatedA, concatanatedB])

Dis_layer2 = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
distance2 = Dis_layer2([concatanatedA, concatanatedB])

Dis_layer3 = Lambda(lambda tensors:K.abs(tensors[0] * tensors[1]))
distance3 = Dis_layer3([concatanatedA, concatanatedB])

distance4 = tensorflow.keras.layers.Concatenate()([distance1, distance2, distance3])

distance = Dense(2048, activation="relu")(distance4)

outputs = Dense(1, activation="sigmoid")(distance)

model = Model(inputs=[imgA, imgB, textA, textB], outputs=outputs)

# Saving model summary
from contextlib import redirect_stdout

with open('Output/modelsummary.txt', 'w') as f:
	with redirect_stdout(f):
		model.summary()


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

optimizer = Adam(lr = 0.00006)

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


hist_df = pd.DataFrame(history.history)

# save to csv: 
hist_csv_file = 'Output/history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

from sklearn.metrics import classification_report

y_preds = model.predict([imgTest[:, 0], imgTest[:, 1], txtTest[:, 0], txtTest[:, 1]])

y_preds = y_preds>=0.5

clsf_report = pd.DataFrame(classification_report(y_true = labelTest, y_pred = y_preds, output_dict=True)).transpose()
clsf_report.to_csv('Output/ClassificationReport.csv', index= True)
