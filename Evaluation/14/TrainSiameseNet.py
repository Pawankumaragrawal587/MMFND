import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "3"  # specify which GPU(s) to be used

# import the necessary packages
from SiameseNetModel.siamesenet import image_feature_extractor, text_feature_extractor
from SiameseNetModel import config
from SiameseNetModel import utils
# from SiameseNetModel import metrics
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
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Input, Embedding, Dropout, Activation, Conv1D, Softmax



# All general imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer 
from sklearn.model_selection import train_test_split
import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, Reshape, Conv2D, MaxPool2D, Concatenate, Flatten, Dropout, Dense, Bidirectional, GlobalAveragePooling1D, GRU, GlobalMaxPooling1D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, GRU, Conv1D, MaxPool1D, Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import SpatialDropout1D
from tensorflow.keras.layers import Dense, Input, Embedding, Dropout, Activation, Conv1D, Softmax
from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
import io, os, gc




# prepare the positive and negative pairs
print("[INFO] preparing positive and negative pairs...")

(img_pair, txt_pair, label, ID) = utils.make_pairs()

print('Img Shape: ', img_pair.shape)
print('Text Shape: ', txt_pair.shape)
print('Label Shape: ', label.shape)
print('ID Shape: ', ID.shape)

# Combining all text instances (# Considering only snoopes for now)
t1 = txt_pair[:,0].tolist()
t2 = txt_pair[:,1].tolist()
uq_tr_1 = list(set(t1))
uq_tr_2 = list(set(t2))
merged = uq_tr_1 + uq_tr_2
total_dataset = merged

# Defining the tokenizer
def get_tokenizer(vocabulary_size):
	print('Training tokenizer...')
	tokenizer = Tokenizer(num_words= vocabulary_size)
	tweet_text = []
	print('Read {} Sentences'.format(len(total_dataset)))
	tokenizer.fit_on_texts(total_dataset)
	return tokenizer

# For getting the embedding matrix
def get_embeddings():
	print('Generating embeddings matrix...')
	# PUt the link to embeddings file here
	embeddings_file = 'glove.42B.300d.txt'
	embeddings_index = dict()
	with open(embeddings_file, 'r', encoding="utf-8") as infile:
		for line in infile:
			values = line.split()
			word = values[0]
			vector = np.asarray(values[1:], "float32")
			embeddings_index[word] = vector
	# create a weight matrix for words in training docs
	vocabulary_size = len(embeddings_index)
	embeddinds_size = list(embeddings_index.values())[0].shape[0]
	print('Vocabulary = {}, embeddings = {}'.format(vocabulary_size, embeddinds_size))
	tokenizer = get_tokenizer(vocabulary_size)
	embedding_matrix = np.zeros((vocabulary_size, embeddinds_size))
	considered = 0
	total = len(tokenizer.word_index.items())
	for word, index in tokenizer.word_index.items():
		if index > vocabulary_size - 1:
			print(word, index)
			continue
		else:
			embedding_vector = embeddings_index.get(word)
			if embedding_vector is not None:
				embedding_matrix[index] = embedding_vector
				considered += 1
	print('Considered ', considered, 'Left ', total - considered)			
	return embedding_matrix, tokenizer, vocabulary_size, embeddinds_size

# Need to make change
def get_data(tokenizer, MAX_LENGTH, txttgt, txtsrc, label):
	print('Loading data')
	X1, X2, Y = [], [], []
	X1 = txttgt.tolist()
	X2 = txtsrc.tolist()
	Y = label.tolist()
	assert len(X1) == len(X2) == len(Y)
	sequences_1 = tokenizer.texts_to_sequences(X1)
	sequences_2 = tokenizer.texts_to_sequences(X2)
	# for i, s in enumerate(sequences):
	# 	sequences[i] = sequences[i][-250:]
	X1 = pad_sequences(sequences_1, maxlen=MAX_LENGTH)
	X2 = pad_sequences(sequences_2, maxlen=MAX_LENGTH)
	Y = np.array(Y)
	return X1, X2, Y

embedding_matrix, tokenizer, Vocabulary, embeddings = get_embeddings()

MAX_LENGTH = 100
# read ml data
X1, X2, Y = get_data(tokenizer, MAX_LENGTH, txt_pair[:,0], txt_pair[:,1], label)

encoder = LabelBinarizer() #convertes into one hot form
encoder.fit(Y)
Y_enc = encoder.transform(Y)

y_train_unsplit = tensorflow.keras.utils.to_categorical(Y_enc)
print(y_train_unsplit)

X1Train, X1Test, X2Train, X2Test, txtTrain, txtTest, imgTrain, imgTest, labelTrain, labelTest, IDTrain, IDTest = train_test_split(X1, X2, txt_pair, img_pair, label, ID,
	test_size = 0.2, random_state=42, stratify=label)

# configure the siamese network
print("[INFO] building siamese network...")

# The maximum number of words to be used. (most frequent)
MAX_NUM_WORDS = embedding_matrix.shape[0]
# Max number of words.
MAX_SEQUENCE_LENGTH = 100
# This is fixed.
NUM_EMBEDDING_DIM = embedding_matrix.shape[1] 
NUM_LSTM_UNITS = 64

# LSTM based model
target_input = Input(shape=(MAX_SEQUENCE_LENGTH, ), dtype='int32')

embedding_layer = Embedding(MAX_NUM_WORDS, NUM_EMBEDDING_DIM, weights = [embedding_matrix], trainable=True)
target_embed = embedding_layer(target_input)
lstm_target = Bidirectional(LSTM(NUM_LSTM_UNITS))

txtfeatsA1 = lstm_target(target_embed)
txtfeatsA = Dense(1024,activation="relu")(txtfeatsA1)

imgA = Input(shape=config.IMG_SHAPE, name='ImageT')

featureExtractor_img1 = image_feature_extractor(config.IMG_SHAPE)

imgfeatsA = featureExtractor_img1(imgA)

concatanatedA = tensorflow.keras.layers.Concatenate()([imgfeatsA,txtfeatsA])

outputs = Dense(1, activation="sigmoid")(concatanatedA)

model = Model(inputs=[imgA, target_input], outputs=outputs)


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

history = model.fit([imgTrain[:, 0], X1Train], labelTrain[:],
	validation_data=([imgTest[:, 0], X1Test], labelTest[:]),
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

y_preds = model.predict([imgTest[:, 0], X1Test])

y_preds = y_preds>=0.5

clsf_report = pd.DataFrame(classification_report(y_true = labelTest, y_pred = y_preds, output_dict=True)).transpose()
clsf_report.to_csv('Output/ClassificationReport.csv', index= True)

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

