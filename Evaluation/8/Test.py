import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "4"  # specify which GPU(s) to be used

# import the necessary packages
from SiameseNetModel.siamesenet import image_feature_extractor, text_feature_extractor
from SiameseNetModel import config
from SiameseNetModel import utils
from SiameseNetModel import metrics
import tensorflow
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, load_model
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

print('Getting IDs')

ID = utils.get_ID()

print('Img Shape: ', img_pair.shape)
print('Text Shape: ', txt_pair.shape)
print('Label Shape: ', label.shape)
print('ID Shape: ', ID.shape)

txtTrain, txtTest, imgTrain, imgTest, labelTrain, labelTest, IDTrain, IDTest = train_test_split(txt_pair, img_pair, label, ID,
	test_size = 0.2, random_state=42, stratify=label)


print('Loading Model')
model = tensorflow.keras.models.load_model(config.MODEL_PATH)

print('Getting Predictions')
y_preds = model.predict([imgTest[:, 0], txtTest[:, 0]])

y_preds = y_preds>=0.5

final_data = []

for i in range(len(IDTest)):
	temp=[]
	temp.append(IDTest[i])
	if labelTest[i]==1:
		temp.append('REAL')
	else:
		temp.append('FAKE')
	if y_preds[i]==True:
		temp.append('REAL')
	else:
		temp.append('FAKE')
	final_data.append(temp)

df = pd.DataFrame(final_data, columns =['IDTest', 'labelTest', 'y_preds'])

df.to_csv('Output/Predictions.csv')
