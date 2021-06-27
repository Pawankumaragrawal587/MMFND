# import the necessary packages
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ast import literal_eval
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array

def get_image(imagepath):

	img = Image.open(imagepath)
	img = img.resize((224,224))

	img = img_to_array(img)

	if img.shape[2]==1:
		img = np.stack([img,img,img],axis=2)
		img = img.reshape(img.shape[0],img.shape[1],3)

	img/=255.0

	return img

def make_pairs():

	# initialize two empty lists to hold the (image, image) pairs and
	# labels to indicate if a pair is positive or negative
	pairImages = []
	pairLabels = []

	print('Loading Datasets...')
	
	data = []
	
	temp = pd.read_csv('../Dataset/Initial_Data/ImageSimilarities_Snopes.csv')
	data.append(temp)
	temp = pd.read_csv('../Dataset/Initial_Data/ImageSimilarities_Reuters.csv')
	data.append(temp)
	temp = pd.read_csv('../Dataset/Initial_Data/ImageSimilarities_ReCovery.csv')
	data.append(temp)
	temp = pd.read_csv('../Dataset/Initial_Data/ImageSimilarities_TICNN.csv')
	data.append(temp)

	data = pd.concat(data, ignore_index=True, sort=False)

	data = data.iloc[:,:].values

	print('Shape of dataset after concatanation: ', data.shape)
	
	for i in range(len(data)):

		target_img = ''

		try:
			target_img = get_image('../Dataset/Initial_Data/TargetImages/' + data[i][1] + '.jpg')

		except:
			continue

		dataset = ''

		if data[i][1][0:3]=='Snp':
			dataset = 'Snopes'
		elif data[i][1][0:3]=='Rtr':
			dataset = 'Reuters'
		elif data[i][1][0:3]=='Rcv':
			dataset = 'ReCovery'
		else:
			dataset = 'TICNN'
		
		for j in range(2,6):

			data[i][j] = literal_eval(data[i][j])
			
			for k in range(min(6,len(data[i][j]))):

				name = '../Dataset/Initial_Data/SourceImages/' + dataset + '/' + data[i][1] + '_' + str(j-1) + '_' + str(data[i][j][k]['Index']) + '.jpg'
				
				if data[i][j][k]['Similarity']>0.5:
					try:
						src_img = get_image(name)
						
						pairImages.append([target_img,src_img])
						pairLabels.append([1])

					except:
						continue

				else:
					try:
						src_img = get_image(name)
						
						pairImages.append([target_img,src_img])
						pairLabels.append([0])

					except:
						continue

	# return a 2-tuple of our image pairs and labels
	return (np.array(pairImages), np.array(pairLabels))

def euclidean_distance(vectors):

	# unpack the vectors into separate lists
	(featsA, featsB) = vectors
	
	# compute the sum of squared distances between the vectors
	sumSquared = K.sum(K.square(featsA - featsB), axis=1,keepdims=True)
	
	# return the euclidean distance between the vectors
	return K.sqrt(K.maximum(sumSquared, K.epsilon()))

def plot_training(H, plotPath):

	# construct a plot that plots and saves the training history
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(H.history["loss"], label="train_loss")
	plt.plot(H.history["val_loss"], label="val_loss")
	plt.plot(H.history["accuracy"], label="train_acc")
	plt.plot(H.history["val_accuracy"], label="val_acc")
	plt.title("Training Loss and Accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	plt.savefig(plotPath)
