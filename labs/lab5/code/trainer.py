'''File contains the trainer class

Complete the functions train() which will train the network given the dataset and hyperparams, and the function __init__ to set your network topology for each dataset
'''
import numpy as np
import sys
import pickle

import nn

from util import *
from layers import *

class Trainer:
	def __init__(self,dataset_name):
		self.save_model = False
		if dataset_name == 'MNIST':
			self.XTrain, self.YTrain, self.XVal, self.YVal, self.XTest, self.YTest = readMNIST()
			# Add your network topology along with other hyperparameters here
			self.batch_size = 50
			self.epochs = 10
			self.lr = 0.1
			self.out_nodes = 10
			self.nn = nn.NeuralNetwork(self.out_nodes, self.lr)
			self.nn.addLayer(FullyConnectedLayer(784, 20, 'relu'))
			self.nn.addLayer(FullyConnectedLayer(20, 40, 'relu'))
			self.nn.addLayer(FullyConnectedLayer(40, self.out_nodes, 'softmax'))


		if dataset_name == 'CIFAR10':
			self.XTrain, self.YTrain, self.XVal, self.YVal, self.XTest, self.YTest = readCIFAR10()
			self.XTrain = self.XTrain[0:5000,:,:,:]
			self.XVal = self.XVal[0:1000,:,:,:]
			self.XTest = self.XTest[0:1000,:,:,:]
			self.YVal = self.YVal[0:1000,:]
			self.YTest = self.YTest[0:1000,:]
			self.YTrain = self.YTrain[0:5000,:]

			self.save_model = True
			self.model_name = "model_.p"

			# Add your network topology along with other hyperparameters here
			self.batch_size = 100
			self.epochs = 20
			self.lr = 0.1
			self.out_nodes = 10
			self.nn = nn.NeuralNetwork(self.out_nodes, self.lr)
			# self.nn.addLayer(ConvolutionLayer([3, 32, 32], [3, 3], 16, 1, 'relu'))
			# self.nn.addLayer(MaxPoolingLayer([16, 30, 30], [2, 2], 2))
			# self.nn.addLayer(ConvolutionLayer([16, 15, 15], [3, 3], 32, 1, 'relu'))
			# self.nn.addLayer(AvgPoolingLayer([32, 13, 13], [2, 2], 2))
			# self.nn.addLayer(FlattenLayer())
			# self.nn.addLayer(FullyConnectedLayer(1152, 512, 'relu'))
			# self.nn.addLayer(FullyConnectedLayer(512, 10, 'softmax'))
			self.nn.addLayer(ConvolutionLayer([3, 32, 32], [5, 5], 16, 3, 'relu'))
			self.nn.addLayer(MaxPoolingLayer([16, 10, 10], [2, 2], 2))
			self.nn.addLayer(FlattenLayer())
			self.nn.addLayer(FullyConnectedLayer(400, 256, 'relu'))
			self.nn.addLayer(FullyConnectedLayer(256, 256, 'relu'))
			self.nn.addLayer(FullyConnectedLayer(256, 10, 'softmax'))

		if dataset_name == 'XOR':
			self.XTrain, self.YTrain, self.XVal, self.YVal, self.XTest, self.YTest = readXOR()
			# Add your network topology along with other hyperparameters here
			self.batch_size = 20
			self.epochs = 30
			self.lr = 0.1
			self.out_nodes = 2
			self.nn = nn.NeuralNetwork(self.out_nodes, self.lr)
			self.nn.addLayer(FullyConnectedLayer(2, 3, 'relu'))
			self.nn.addLayer(FullyConnectedLayer(3, self.out_nodes, 'softmax'))


		if dataset_name == 'circle':
			self.XTrain, self.YTrain, self.XVal, self.YVal, self.XTest, self.YTest = readCircle()
			# Add your network topology along with other hyperparameters here
			self.batch_size = 20
			self.epochs = 30
			self.lr = 0.1
			self.out_nodes = 2
			self.nn = nn.NeuralNetwork(self.out_nodes, self.lr)
			self.nn.addLayer(FullyConnectedLayer(2, 3, 'relu'))
			self.nn.addLayer(FullyConnectedLayer(3, self.out_nodes, 'softmax'))
	def train(self, verbose=True):
		# Method for training the Neural Network
		# Input
		# trainX - A list of training input data to the neural network
		# trainY - Corresponding list of training data labels
		# validX - A list of validation input data to the neural network
		# validY - Corresponding list of validation data labels
		# printTrainStats - Print training loss and accuracy for each epoch
		# printValStats - Prints validation set accuracy after each epoch of training
		# saveModel - True -> Saves model in "modelName" file after each epoch of training
		# loadModel - True -> Loads model from "modelName" file before training
		# modelName - Name of the model from which the funtion loads and/or saves the neural net
		
		# The methods trains the weights and baises using the training data(trainX, trainY)
		# and evaluates the validation set accuracy after each epoch of training

		for epoch in range(self.epochs):
			# A Training Epoch
			if verbose:
				print("Epoch: ", epoch)

			# TODO
			# Shuffle the training data for the current epoch
			n_data = self.XTrain.shape[0]
			n_dims = self.XTrain.shape[1]
			indices = np.random.choice(n_data,n_data,replace = False)
			self.XTrain = self.XTrain[indices]
			self.YTrain = self.YTrain[indices]

			# Initializing training loss and accuracy
			trainLoss = 0
			trainAcc = 0

			# Divide the training data into mini-batches
			numBatches = n_data//self.batch_size
			for i in range(numBatches):
				offset = i*self.batch_size
				# Calculate the activations after the feedforward pass
				preds = self.nn.feedforward(self.XTrain[offset:offset+self.batch_size])
				# Compute the loss  
				curr_loss = self.nn.computeLoss(self.YTrain[offset:offset+self.batch_size],preds)
				trainLoss += curr_loss
				# Calculate the training accuracy for the current batch
				predlabels = oneHotEncodeY(np.argmax(preds[-1],1), self.out_nodes)
				trainAcc += self.nn.computeAccuracy(self.YTrain[offset:offset+self.batch_size], predlabels)
				# Backpropagation Pass to adjust weights and biases of the neural network
				self.nn.backpropagate(preds, self.YTrain[offset:offset+self.batch_size])

			# END TODO
			# Print Training loss and accuracy statistics
			trainAcc /= numBatches
			if verbose:
				print("Epoch ", epoch, " Training Loss=", trainLoss, " Training Accuracy=", trainAcc)
			
			if self.save_model:
				model = []
				for l in self.nn.layers:
					# print(type(l).__name__)
					if type(l).__name__ != "AvgPoolingLayer" and type(l).__name__ != "FlattenLayer" and type(l).__name__ != "MaxPoolingLayer": 
						model.append(l.weights) 
						model.append(l.biases)
				pickle.dump(model,open(self.model_name,"wb"))
				print("Model Saved... ")

			# Estimate the prediction accuracy over validation data set
			if self.XVal is not None and self.YVal is not None and verbose:
				_, validAcc = self.nn.validate(self.XVal, self.YVal)
				print("Validation Set Accuracy: ", validAcc, "%")

		pred, acc = self.nn.validate(self.XTest, self.YTest)
		print('Test Accuracy ',acc)

