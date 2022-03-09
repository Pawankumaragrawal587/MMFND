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
textA = Input(shape=config.TXT_SHAPE, name='TextT')

featureExtractor_img = image_feature_extractor(config.IMG_SHAPE)
featureExtractor_txt = text_feature_extractor(config.TXT_SHAPE)

imgfeatsA = featureExtractor_img(imgA)

txtfeatsA = featureExtractor_txt(textA)

concatanatedA = tensorflow.keras.layers.Concatenate()([imgfeatsA,txtfeatsA])

outputs = Dense(1, activation="sigmoid")(concatanatedA)

model = Model(inputs=[imgA, textA], outputs=outputs)

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

model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=metricss)

# train the model
print("[INFO] training model...")

history = model.fit([imgTrain[:, 0], txtTrain[:, 0]], labelTrain[:],
	validation_data=([imgTest[:, 0], txtTest[:, 0]], labelTest[:]),
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

y_preds = model.predict([imgTest[:, 0], txtTest[:, 0]])

y_preds = y_preds>=0.5

clsf_report = pd.DataFrame(classification_report(y_true = labelTest, y_pred = y_preds, output_dict=True)).transpose()
clsf_report.to_csv('Output/ClassificationReport.csv', index= True)
