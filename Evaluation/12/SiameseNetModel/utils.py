# import the necessary packages
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ast import literal_eval
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import re
from SiameseNetModel import config
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications import vgg16
from tensorflow.keras import models, Model
from numpy import save
from numpy import load

def get_image(imagepath):

	img = Image.open(imagepath)
	img = img.resize((224,224))

	img = img_to_array(img)

	if img.shape[2]==1:
		img = np.stack([img,img,img],axis=2)
		img = img.reshape(img.shape[0],img.shape[1],3)

	return img

def cleaned_text(x):

	x = str(x)
	val =  [re.sub(r'[^a-zA-Z]',' ',w).lower() for w in x.split() ]  # if re.sub(r'[^a-zA-Z]',' ',w).lower() not in stop_words

	res = ""

	for i in range(min(300,len(val))):
	    res = res + val[i] + ' '
	    
	res = res.split()
	res = ' '.join(res)

	return res

def make_pairs():

	# initialize two empty lists to hold the (image, image) pairs and
	# labels to indicate if a pair is positive or negative

	pairImages = []
	pairTexts = []
	pairLabels = []
	ID = []

	print('[INFO] Loading and Processing Dataset...')
	
	source = []
	
	# temp = pd.read_csv('../Dataset/Final_Data/Source/Snopes.csv')
	# source.append(temp)
	# temp = pd.read_csv('../Dataset/Final_Data/Source/Reuters.csv')
	# source.append(temp)
	# temp = pd.read_csv('../Dataset/Final_Data/Source/ReCovery.csv')
	# source.append(temp)
	temp = pd.read_csv('../Dataset/Final_Data/Source/TICNN.csv')
	source.append(temp)

	source = pd.concat(source, ignore_index=True, sort=False)

	target = []
	
	# temp = pd.read_csv('../Dataset/Final_Data/Target/Snopes.csv')
	# target.append(temp)
	# temp = pd.read_csv('../Dataset/Final_Data/Target/Reuters.csv')
	# target.append(temp)
	# temp = pd.read_csv('../Dataset/Final_Data/Target/ReCovery.csv')
	# target.append(temp)
	temp = pd.read_csv('../Dataset/Final_Data/Target/TICNN.csv')
	target.append(temp)

	target = pd.concat(target, ignore_index=True, sort=False)

	source = source.iloc[:,:].values
	target = target.iloc[:,:].values

	print('Shape of Source after concatanation: ', source.shape)
	print('Shape of Target after concatanation: ', target.shape)

	
	for i in range(len(source)):
		
		if i%100==0:
			print('Loading Data...',i)

		target_img = ''

		try:
			target_img = get_image('../Dataset/Initial_Data/TargetImages/' + source[i][1] + '.jpg')

		except:
			continue

		target_txt = cleaned_text(target[i][3])

		dataset = ''
		if source[i][1][0:3]=='Snp':
			dataset = 'Snopes'
		elif source[i][1][0:3]=='Rtr':
			dataset = 'Reuters'
		elif source[i][1][0:3]=='Rcv':
			dataset = 'ReCovery'
		else:
			dataset = 'TICNN'

		for j in range(3,3+4*source[i][2],4):
			
			source[i][j+2] = literal_eval(source[i][j+2])

			src_txt = cleaned_text(source[i][j+1])

			try:
				src_img = get_image('../Dataset/Initial_Data/SourceImages/' + dataset + '/' + source[i][j+2]['image_name'])

				pairImages.append([target_img, src_img])
				pairTexts.append([target_txt, src_txt])	
				ID.append([source[i][j+2]['image_name']])			
				if target[i][5]=='FAKE':
					pairLabels.append([0])
				else:
					pairLabels.append([1])
			except:
				continue
	
	# return a 2-tuple of our pairs and labels
	return (np.array(pairImages), np.array(pairTexts), np.array(pairLabels), np.array(ID))

def plot_training(H, plotPath):

	# construct a plot that plots and saves the training history
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(H.history["loss"], label="train_loss")
	plt.plot(H.history["val_loss"], label="val_loss")
	plt.plot(H.history["acc"], label="train_acc")
	plt.plot(H.history["val_acc"], label="val_acc")
	plt.title("Training Loss and Accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="upper right")
	plt.savefig(plotPath)
