#!/usr/bin/env python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)
# termcolor, do (pip install termcolor)

import tensorflow as tf
import cv2
import sys
import os
import glob
# import Misc.ImageUtils as iu
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from Network.Network import HomographyModel
# from Network.Resnet import HomographyModel
from Misc.MiscUtils import *
from Misc.DataUtils import *
import numpy as np
import time
import argparse
import shutil
from StringIO import StringIO
import string
from termcolor import colored, cprint
import math as m
from tqdm import tqdm
from Misc.TFSpatialTransformer import *
from Misc.DLT import *
# Don't generate pyc codes
sys.dont_write_bytecode = True


def GenerateBatch(BasePath, DirNamesTrain1, DirNamesTrain2, TrainLabels, ImageSize, MiniBatchSize, PerEpochCounter):
	"""
	Inputs: 
	BasePath - Path to COCO folder without "/" at the end
	DirNamesTrain - Variable with Subfolder paths to train files
	NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
	TrainLabels - Labels corresponding to Train
	NOTE that TrainLabels can be replaced by Val/TestLabels for generating batch corresponding to validation (held-out testing in this case)/testing
	ImageSize - Size of the Image
	MiniBatchSize is the size of the MiniBatch
	Outputs:
	I1Batch - Batch of images
	LabelBatch - Batch of one-hot encoded labels 
	"""
	I1Batch = []
	out_images = []
	homography_image = []
	LabelBatch = []
	ImageNum = 0
	img_size = 128
	for i in range(PerEpochCounter*MiniBatchSize, (PerEpochCounter+1)*MiniBatchSize):
		im1 = np.float32(cv2.imread(DirNamesTrain1[i], 0))
		#print(im1.shape)
		im2 = np.float32(cv2.imread(DirNamesTrain2[i], 0))
		#print("\n----"+str(im2.shape)+"\n---")

		ims = np.zeros((img_size, img_size,2))
		
		# im_for_homogrphay = np.zeros((img_size, img_size,1))
		# im_out = np.zeros((img_size, img_size,1))
		
		ims[:,:,0] = (im1 -127.0)/127.0
		ims[:,:,1] = (im2 - 127.0)/127.0
		tmp = TrainLabels[i]
		
		LabelBatch.append(tmp.reshape((8,1)))

		I1Batch.append(ims)
		homography_image.append(im2.reshape(img_size,img_size,1)) #### goes into transformer layer
		out_images.append(im1.reshape(img_size,img_size,1))   #### calculate loss on this
	return I1Batch, homography_image, out_images, LabelBatch

def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
	"""
	Prints all stats with all arguments
	"""
	print("--------------------------------------------------------")
	print('Number of Epochs Training will run for ' + str(NumEpochs))
	print('Factor of reduction in training data is ' + str(DivTrain))
	print('Mini Batch Size ' + str(MiniBatchSize))
	print('Number of Training Images ' + str(NumTrainSamples))
	if LatestFile is not None:
		print('Loading latest checkpoint with the name ' + LatestFile)              
	print("--------------------------------------------------------")

	
def TrainOperation(ImgPH, HomoPH, OutPH, LabelPH, DirNamesTrain1, DirNamesTrain2, TrainLabels, NumTrainSamples, ImageSize,
				   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
				   DivTrain, LatestFile, BasePath, LogsPath, ModelType, DirNamesValid1, DirNamesValid2, 
				   NumValidSamples, ValidLabels, lr, OptimizerHomography):
	"""
	Inputs: 
	ImgPH is the Input Image placeholder
	LabelPH is the one-hot encoded label placeholder
	DirNamesTrain - Variable with Subfolder paths to train files
	TrainLabels - Labels corresponding to Train/Test
	NumTrainSamples - length(Train)
	ImageSize - Size of the image
	NumEpochs - Number of passes through the Train data
	MiniBatchSize is the size of the MiniBatch
	SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
	CheckPointPath - Path to save checkpoints/model
	DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
	LatestFile - Latest checkpointfile to continue training
	BasePath - Path to COCO folder without "/" at the end
	LogsPath - Path to save Tensorboard Logs
	ModelType - Supervised or Unsupervised Model
	Outputs:
	Saves Trained network in CheckPointPath and Logs to LogsPath
	"""      
	# Predict output with forward pass

	# with tf.name_scope("Hmography_Net"):

	H4Pt = HomographyModel(ImgPH, ImageSize, MiniBatchSize)
	
	with tf.name_scope("Tensor_DLT"):
		H_inv = DLT(MiniBatchSize, H4Pt)
	
	out_size = tf.convert_to_tensor(np.array([128,128]),dtype=tf.int32)
	
	with tf.name_scope("Spatial_Transformer_Layer"):
		output, condition = transformer(OutPH, H_inv, out_size)
		img_summary = tf.summary.image('Warped_images', output)
		actual_out_summary = tf.summary.image('Match_with_this', OutPH)
		input_image_summary = tf.summary.image('Input_Image', HomoPH)

	initial_lr = 3*0.001

	############################## Loss functions ###################################
	if ModelType=="Unsup":
		with tf.name_scope('Loss'):
			# loss = tf.nn.l2_loss(prLogits-LabelPH)
			# loss = tf.abs()
			loss = tf.reduce_mean(tf.abs(output - HomoPH))*0.009
			lossSummary = tf.summary.scalar('LossEveryIter', loss)

		with tf.name_scope("ValidationLoss"):
			validation_loss = tf.reduce_mean(tf.abs(output-HomoPH))*0.009
			validationLossSummary = tf.summary.scalar("ValidationLoss ",validation_loss)
	
	else:
		with tf.name_scope('Loss'):
			# loss = tf.nn.l2_loss(prLogits-LabelPH)
			loss = tf.reduce_sum(tf.square(tf.reshape(H4Pt, [MiniBatchSize,8,1]) - LabelPH))/2.0
			lossSummary = tf.summary.scalar('LossEveryIter', loss)

		with tf.name_scope("ValidationLoss"):
			validation_loss = tf.reduce_sum(tf.square(tf.reshape(H4Pt, [MiniBatchSize,8,1]) - LabelPH))/2.0
			validationLossSummary = tf.summary.scalar("ValidationLoss ",validation_loss)
	##################################################################################

	#########################  Optimizers ############################
	if OptimizerHomography=="Adam":
		with tf.name_scope('Adam'):
			Optimizer = tf.train.AdamOptimizer(learning_rate = lr).minimize(loss)
	else:
		with tf.name_scope('SGD'):
			Optimizer = tf.keras.optimizers.SGD(learning_rate = lr, momentum=0.9).minimize(loss)
	###################################################################

	#########################  Erros ##################################
	with tf.name_scope('Error'):
		# mm = tf.reshape()
		msr = tf.abs(tf.reshape(H4Pt, [MiniBatchSize,8,1]) - LabelPH)
		Err = tf.reduce_mean(msr)
		errorSummary = tf.summary.scalar("Error ",Err)

	with tf.name_scope("ValidationError"):
		validation_Err = tf.reduce_mean(tf.abs(tf.reshape(H4Pt, [MiniBatchSize,8,1]) - LabelPH))		
		validationErrorSummary = tf.summary.scalar("ValidationError ",validation_Err)
	######################################################################

	
	#####################  Summaries ###############################
	TrainingSummary = tf.summary.merge([lossSummary,errorSummary])
	ValidationSummary = tf.summary.merge([validationLossSummary, validationErrorSummary])
	images_summary = tf.summary.merge([img_summary, actual_out_summary, input_image_summary	])
	##############################################################
	# Setup Saver
	Saver = tf.train.Saver()
	acc = []
	temp_error = []
	temp_loss = []
	temp_valid_loss = []
	loss_ = []
	with tf.Session() as sess:       
		if LatestFile is not None:
			Saver.restore(sess, CheckPointPath + LatestFile + '.ckpt')
			# Extract only numbers from the name
			StartEpoch = int(''.join(c for c in LatestFile.split('a')[0] if c.isdigit()))
			print('Loaded latest checkpoint with the name ' + LatestFile + '....')
		else:
			sess.run(tf.global_variables_initializer())
			StartEpoch = 0
			print('New model initialized....')

		# Tensorboard
		Writer = tf.summary.FileWriter(LogsPath, graph=tf.get_default_graph())
			
		for Epochs in tqdm(range(StartEpoch, NumEpochs)):
			NumIterationsPerEpoch = int(NumTrainSamples/MiniBatchSize/DivTrain)
			if Epochs%8==0:
				initial_lr = initial_lr/10.0
				print("------------------------\n setting new learning rate\n")
				print(initial_lr)
				# MiniBatchSize = MiniBatchSize*2
			for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
				I1Batch, Transform, output_images, labels = GenerateBatch(BasePath, DirNamesTrain1, DirNamesTrain2, TrainLabels, ImageSize, MiniBatchSize, PerEpochCounter)
				
				FeedDict = {ImgPH: I1Batch, HomoPH: Transform, OutPH: output_images , lr: initial_lr, LabelPH:labels }
				_, LossThisBatch, Summary, Im_Summary, ER = sess.run([Optimizer, loss, TrainingSummary, images_summary, Err], feed_dict=FeedDict)
				
				temp_loss.append(LossThisBatch)
				if PerEpochCounter % SaveCheckPoint == 0:
					# Save the Model learnt in this epoch
					# SaveName =  CheckPointPath + str(Epochs) + 'a' + str(PerEpochCounter) + 'model.ckpt'
					# Saver.save(sess,  save_path=SaveName)
					# print('\n' + SaveName + ' Model Saved...')
					# if (Epochs*NumIterationsPerEpoch + PerEpochCounter)>0:
					Writer.add_summary(Im_Summary, Epochs*NumIterationsPerEpoch + PerEpochCounter)
					print("Loss of model : "+str(LossThisBatch))
					# print("-----------------------------------------------------\n")
					# print("ctual values : ")
					# print(actual)
					# print("-----------------------------------------------------\n")
					# print("Predicted valuess : "+str(h4pt_pre))
					# print("-----------------------------------------------------")
					# print("Error valuess : ")
					# print(MSR)
					# print("-----------------------------------------------------")
					# print("Error:: : ")
					# print(ERR)
					# print("-----------------------------------------------------")
				# Tensorboard
				Writer.add_summary(Summary, Epochs*NumIterationsPerEpoch + PerEpochCounter)
				# If you don't flush the tensorboard doesn't update until a lot of iterations!
				Writer.flush()

			######################### Validation ################################
			NumIterationsPerEpoch = int(NumValidSamples/MiniBatchSize)
			for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
				I1Batch, Transform, output_images, labels = GenerateBatch(BasePath, DirNamesValid1, DirNamesValid2, ValidLabels, ImageSize, MiniBatchSize, PerEpochCounter)
				FeedDict = {ImgPH: I1Batch, HomoPH: Transform, OutPH: output_images , lr: initial_lr, LabelPH:labels}				
				valSummary, Im_Summary = sess.run([ValidationSummary, images_summary], feed_dict=FeedDict)
				# temp_valid_loss.append(LossThisBatchValidation)
				Writer.add_summary(valSummary, Epochs*NumIterationsPerEpoch + PerEpochCounter)
				# If you don't flush the tensorboard doesn't update until a lot of iterations!
				Writer.flush()

			# Save model every epoch
			SaveName = CheckPointPath + str(Epochs) + 'model.ckpt'
			Saver.save(sess, save_path=SaveName)
			print('\n' + SaveName + ' Model Saved...')
			print("----------------After epoch------------")
			print("Total loss = "+str(np.array(temp_loss).sum()))
			print("Validation loss = "+str(np.array(temp_valid_loss).sum()))
			print("--------------------------------------------")
			temp_loss = []
			temp_valid_loss = []

def main():
	"""
	Inputs: 
	None
	Outputs:
	Runs the Training and testing code based on the Flag
	"""
	# Parse Command Line arguments
	Parser = argparse.ArgumentParser()
	Parser.add_argument('--BasePath', default='..', help='Base path of images, Default:/media/nitin/Research/Homing/SpectralCompression/COCO')
	Parser.add_argument('--CheckPointPath', default='../Checkpoints/', help='Path to save Checkpoints, Default: ../Checkpoints/')
	Parser.add_argument('--ModelType', default='Unsup', help='Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup')
	Parser.add_argument('--NumEpochs', type=int, default=20, help='Number of Epochs to Train for, Default:50')
	Parser.add_argument('--DivTrain', type=int, default=1, help='Factor to reduce Train data by per epoch, Default:1')
	Parser.add_argument('--MiniBatchSize', type=int, default=8, help='Size of the MiniBatch to use, Default:1')
	Parser.add_argument('--LoadCheckPoint', type=int, default=0, help='Load Model from latest Checkpoint from CheckPointsPath?, Default:0')
	Parser.add_argument('--LogsPath', default='Logs/', help='Path to save Logs for Tensorboard, Default=Logs/')
	Parser.add_argument('--Optimizer', default='Adam', help='Select you want to choose SGD or Adam optimizer')

	Args = Parser.parse_args()
	NumEpochs = Args.NumEpochs
	BasePath = Args.BasePath
	DivTrain = float(Args.DivTrain)
	MiniBatchSize = Args.MiniBatchSize
	LoadCheckPoint = Args.LoadCheckPoint
	CheckPointPath = Args.CheckPointPath
	LogsPath = Args.LogsPath
	ModelType = Args.ModelType
	OptimizerHomography = Args.Optimizer
	# Setup all needed parameters including file reading
	DirNamesTrain1, DirNamesTrain2, SaveCheckPoint, ImageSize, NumTrainSamples, TrainLabels, NumClasses = SetupAll(BasePath, CheckPointPath)
	DirNamesValid1, DirNamesValid2, NumValidSamples, ValidLabels = setupValidation(BasePath)

	# Find Latest Checkpoint File
	if LoadCheckPoint==1:
		LatestFile = FindLatestModel(CheckPointPath)
	else:
		LatestFile = None
	
	# Pretty print stats
	PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

	# Define PlaceHolder variables for Input and Predicted output
	ImgPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, ImageSize[0], ImageSize[1], ImageSize[2]))
	HomoPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, ImageSize[0], ImageSize[1], 1)) # Output image
	OutPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, ImageSize[0], ImageSize[1], 1)) # Output image
	LabelPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, 8,1))
	learning_rate = tf.placeholder(tf.float32, shape=[])
	###############################################################
	TrainOperation(ImgPH, HomoPH, OutPH, LabelPH, DirNamesTrain1, DirNamesTrain2, TrainLabels, NumTrainSamples, ImageSize,
				   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
				   DivTrain, LatestFile, BasePath, LogsPath, ModelType, DirNamesValid1, DirNamesValid2, NumValidSamples, ValidLabels, learning_rate, OptimizerHomography)
		
	
if __name__ == '__main__':
	main()
 
