# USAGE
# python lenet_mnist.py --save-model 1 --weights output/lenet_weights.hdf5
# python lenet_mnist.py --load-model 1 --weights output/lenet_weights.hdf5

# import the necessary packages
from pyimagesearch.cnn.networks.lenet import LeNet
from sklearn.cross_validation import train_test_split
from sklearn import datasets
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import argparse
import cv2
import pickle
import matplotlib.pyplot as plt
import random

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--save-model", type=int, default=-1,
	help="(optional) whether or not model should be saved to disk")
ap.add_argument("-l", "--load-model", type=int, default=-1,
	help="(optional) whether or not pre-trained model should be loaded")
ap.add_argument("-w", "--weights", type=str,
	help="(optional) path to weights file")
args = vars(ap.parse_args())

# grab the MNIST dataset (if this is your first time running this
# script, the download may take a minute -- the 55MB MNIST dataset
# will be downloaded)
# print("[INFO] downloading MNIST...")
# dataset = datasets.fetch_mldata("MNIST (Original)")

# reshape the MNIST dataset from a flat list of 784-dim vectors, to
# 28 x 28 pixel images, then scale the data to the range [0, 1.0]
# and construct the training and testing splits
# data = dataset.data.reshape((dataset.data.shape[0], 28, 28))
# data = data[:, :, :, np.newaxis]


#read file
try:   
    with open('Xdata.txt','rb') as x_file:  
        X_load=pickle.load(x_file)  
    with open('Ydata.txt','rb') as y_file:  
        y_load=pickle.load(y_file)  
    with open('label.txt','rb') as label_file:  
        label_load=pickle.load(label_file)
except IOError as err:  
    print('File error: ' + str(err))  
except pickle.PickleError as perr:  
    print('Pickling error: ' + str(perr))  

#数据集的类别数
# numClass = int(y_load.size/1000)
numClass = 2

#数据集的图片大小
datasize = 48

#将x，y合并 并打乱顺序
# y_load = y_load[:,np.newaxis]
# dataset = np.concatenate((y_load,X_load), axis = 1)
# np.random.shuffle(dataset)
# #将x，y分开
# X_load = dataset[:,1:]
# y_load = dataset[:,0]

data = X_load.reshape((X_load.data.shape[0],datasize,datasize))
data = data[:,:,:,np.newaxis]

(trainData, testData, trainLabels, testLabels) = train_test_split(
	data / 255.0, y_load.astype("int"), test_size=0.33)

# transform the training and testing labels into vectors in the
# range [0, classes] -- this generates a vector for each label,
# where the index of the label is set to `1` and all other entries
# to `0`; in the case of MNIST, there are 10 class labels
trainLabels = np_utils.to_categorical(trainLabels, numClass)
testLabels = np_utils.to_categorical(testLabels, numClass)
epochs = 20


datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(trainData)

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.01)
model = LeNet.build(width=datasize, height=datasize, depth=1, classes=numClass,
	weightsPath=args["weights"] if args["load_model"] > 0 else None)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# only train and evaluate the model if we *are not* loading a
# pre-existing model
if args["load_model"] < 0:
	print("[INFO] training...")
	# model.fit(trainData, trainLabels, batch_size=128, nb_epoch=epochs,
	# 	verbose=1)

	# fits the model on batches with real-time data augmentation:
	model.fit_generator(datagen.flow(trainData, trainLabels, batch_size=128, shuffle=True),
						 epochs=epochs)

	# show the accuracy on the testing set
	print("[INFO] evaluating...")
	(loss, accuracy) = model.evaluate(testData, testLabels,
		batch_size=128, verbose=1)
	print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))

# check to see if the model should be saved to file
if args["save_model"] > 0:
	print("[INFO] dumping weights to file...")
	model.save_weights(args["weights"], overwrite=True)



# # randomly select a few testing digits
# for i in np.random.choice(np.arange(0, len(testLabels)), size=(10,)):
# 	# classify the character
# 	probs = model.predict(testData[np.newaxis, i])
# 	prediction = probs.argmax(axis=1)

# 	# resize the image from a 28 x 28 image to a 96 x 96 image so we
# 	# can better see it
# 	image = (testData[i] * 255).astype("uint8")
# 	# image = cv2.merge([image] * 3)
# 	image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
# 	cv2.putText(image, str(prediction[0]), (5, 20),
# 		cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

# 	# show the image and prediction
# 	print("[INFO] Predicted: {}, Actual: {},Predicted chinese: {}".format(prediction[0],
# 		np.argmax(testLabels[i]),label_load[prediction[0]]))
# 	cv2.imshow("Chinese", image)
# 	cv2.waitKey(0)

#找到所有判断错误的测试样本
for i in range(len(testLabels)):
	probs = model.predict(testData[np.newaxis, i])
	prediction = probs.argmax(axis=1)
	if prediction[0]!=np.argmax(testLabels[i]):
		image = (testData[i] * 255).astype("uint8")
		image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
		cv2.putText(image, str(prediction[0]), (5, 20),
			cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
		# show the image and prediction
		print("[ERROR PREDICTION]Predicted: {}, Actual: {},Predicted chinese: {}".format(prediction[0],
			np.argmax(testLabels[i]),label_load[prediction[0]]))
		cv2.imshow("Chinese", image)
		cv2.waitKey(0)


def gaussianNoisy(im, mean, sigma):
	"""
    对图像做高斯噪音处理
    :param im: 单通道图像
    :param mean: 偏移量
    :param sigma: 标准差
    :return:
    """
	for _i in range(len(im)):
		im[_i] += random.gauss(mean, sigma)
	return im