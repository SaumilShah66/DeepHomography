#!/usr/bin/evn python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project1: MyAutoPano: Phase 1 Starter Code

Author(s): 
Chahat Deep Singh (chahat@terpmail.umd.edu) 
PhD Student in Computer Science,
University of Maryland, College Park

Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

# Code starts here:

import numpy as np
import cv2
import tensorflow as tf
from Network.Network import HomographyModel
from Misc.MiscUtils import *
from Misc.TFSpatialTransformer import *
from Misc.DLT import *
import matplotlib.pyplot as plt
# Add any python libraries here
import random	
import argparse

path = "../Data/P1TestSet/Phase2/"


def generatePatches(image, patchSize, p):
	#  img ---- colored image
	img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# img = image
	h,w = img.shape
	
	x_start, y_start = (int(w/2) - patchSize/2), (int(h/2) - patchSize/2)
	# print("\n\n\n-----"+str([lim_x, lim_y])+"\n\n----")
	im1 = img[y_start:y_start + patchSize, x_start:x_start + patchSize]
	if (im1.shape != (patchSize,patchSize)):
		print("\n\n\n-----Image shape = "+str(im1.shape)+"\nBut start ="+str([lim_x, lim_y])+"\n original image size = "+str(img.shape) +"\n\n----")
	## generate random perturb points
	u = [random.randint(-p/2,p/2) for i in range(4)]  ### Perturb in x 
	v = [random.randint(-p/2,p/2) for i in range(4)]  ### Perturn in y
	### Find homography
	pa = np.array([[x_start, y_start],
					[x_start + patchSize, y_start],
					[x_start + patchSize, y_start + patchSize],
					[x_start, y_start + patchSize]], 
					dtype='f')
	pb = np.array([[x_start + u[0], y_start + v[0]],
					[x_start + patchSize + u[1], y_start + v[1]],
					[x_start + patchSize + u[2], y_start + patchSize + v[2]],
					[x_start + u[3], y_start + patchSize +v[3]]], 
					dtype='f')
	H = np.linalg.inv(cv2.getPerspectiveTransform(pa,pb))
	im2_ = cv2.warpPerspective(img,H,(w,h))
	image2 = cv2.warpPerspective(image,H,(w,h))
	color1 = (0, 255, 0)
	thickness = 2
	image_ = drawPatch(image2, [x_start, y_start],color1,thickness)

	cv2.imwrite("report.png",image_)
	im2 = im2_[y_start:y_start + patchSize, x_start:x_start + patchSize]
	
	output_homo = []
	for i in range(4):
		output_homo.append(u[i])
		output_homo.append(v[i])
	# output_homo = np.array(output_homo)
	return im1, im2, output_homo, [x_start, y_start]


def readImages(num, maxp):
	path = "../Data/P1TestSet/Phase2/"+str(num)+".jpg"
	image1 = cv2.imread(path)
	im1, im2, output_homo, start = generatePatches(image1, 128, maxp)
	im = np.zeros([1,128,128,2])
	im[0, :, :, 0] = (im1 - 127.0)/127.0
	im[0, :, :, 1] = (im2 - 127.0)/127.0
 	return im, image1, output_homo, start, im1

def homography(img1, img2, h):
    h1,w1,_ = img1.shape
    h2,w2,_ = img2.shape
   
    pts1 = np.array([[0,0],[0,h1],[w1,h1],[w1,0]], dtype='f').reshape(-1,1,2)
    pts2 = np.array([[0,0],[0,h2],[w2,h2],[w2,0]], dtype='f').reshape(-1,1,2)
    pts1_ = cv2.perspectiveTransform(pts1, h)
    print(pts1_.shape)
    print(pts2.shape)
    pts = np.concatenate((pts1_, pts2), axis=0)
    xmin, ymin = (pts.min(axis=0).ravel() - 0.5)
    xmax, ymax = (pts.max(axis=0).ravel() + 0.5)
    xmin, xmax, ymin, ymax= int(xmin), int(xmax), int(ymin), int(ymax)
    t = [-xmin,-ymin]
#     t=[0,0]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate
#     mergedImg= np.zeros([ymax-ymin,xmax-xmin,3])
    mergedImg= np.zeros([ymax-ymin,xmax-xmin,3], dtype= 'uint8')
#     mergedImg[-ymin:h2-ymin,-xmin:w2-xmin]= img2
    mergedImg = cv2.warpPerspective(img1,Ht.dot(h),((xmax-xmin, ymax-ymin)))
    
    return mergedImg

def model(im, im1, modelPath):
	ImageSize = [128,128,2]
	ImgPH = tf.placeholder(tf.float32, shape=(1, ImageSize[0], ImageSize[1], ImageSize[2]))
	HomoPH = tf.placeholder(tf.float32, shape=(1, ImageSize[0], ImageSize[1], 1)) # Output image
	OutPH = tf.placeholder(tf.float32, shape=(1, ImageSize[0], ImageSize[1], 1)) # Output image
	LabelPH = tf.placeholder(tf.float32, shape=(1, 8,1))
	learning_rate = tf.placeholder(tf.float32, shape=[])

	H4Pt = HomographyModel(ImgPH, ImageSize, 1)
	H_inv = DLT(1, H4Pt)
	out_size = tf.convert_to_tensor(np.array([128,128]),dtype=tf.int32)
	output, condition = transformer(OutPH, H_inv, out_size)

	Saver = tf.train.Saver()
	
	with tf.Session() as sess:
		Saver.restore(sess, modelPath)
		FeedDict = {ImgPH: im, OutPH:im1 }
		H4pt_pred, H_pred, out_image = sess.run([H4Pt, H_inv, output], FeedDict)
	print(H_pred)
	print(H_pred.shape)
	print(H4pt_pred)
	# cv2.imwrite("pred.jpg", out_image)
	return out_image,H4pt_pred, H_pred

def drawPatch(image1, start,color1,thickness):
	pt1 = (start[0], start[1])
	pt2 = (start[0]+128, start[1])
	pt3 = (start[0]+128, start[1] + 128)
	pt4 = (start[0], start[1] + 128)

	image = cv2.line(image1, pt1, pt2, color1, thickness)
	image = cv2.line(image1, pt2, pt3, color1, thickness)
	image = cv2.line(image1, pt3, pt4, color1, thickness)
	image = cv2.line(image1, pt4, pt1, color1, thickness) 
	return image

def drawPerturb(image1, start, output_homo, color2, thickness):
	pt1 = (start[0] + output_homo[0], start[1] + output_homo[1])
	pt2 = (start[0] + 128 + output_homo[2], start[1] + output_homo[3])
	pt3 = (start[0] + 128 + output_homo[4], start[1] + 128 + output_homo[5])
	pt4 = (start[0] + output_homo[6], start[1] + 128 + output_homo[7])

	image = cv2.line(image1, pt1, pt2, color2, thickness)
	image = cv2.line(image1, pt2, pt3, color2, thickness)
	image = cv2.line(image1, pt3, pt4, color2, thickness)
	image = cv2.line(image1, pt4, pt1, color2, thickness) 
	return image

def drawPrediction(image, start, h4, color3, thickness):
	pt1 = (start[0] + int(h4[0,0]), start[1] + int(h4[0,1]))
	pt2 = (start[0] + 128 + int(h4[0,2]), start[1] + int(h4[0,3]))
	pt3 = (start[0] + 128 + int(h4[0,4]), start[1] + 128 + int(h4[0,5]))
	pt4 = (start[0] + int(h4[0,6]), start[1] + 128 + int(h4[0,7]))

	image = cv2.line(image, pt1, pt2, color3, thickness)
	image = cv2.line(image, pt2, pt3, color3, thickness)
	image = cv2.line(image, pt3, pt4, color3, thickness)
	image = cv2.line(image, pt4, pt1, color3, thickness) 
	return image

def main():
# Add any Command Line arguments here
	Parser = argparse.ArgumentParser()
	Parser.add_argument('--Model', default="Unsup", help="Supervised or Unsupervised model")
	Parser.add_argument('--TestNumber', default="1", help="Number of image from test cases")
	Parser.add_argument('--MaxPerturb', default=32, help="Maximum perturb size")
	Args = Parser.parse_args()
	Model = Args.Model
	num = Args.TestNumber
	MaxPerturb = int(Args.MaxPerturb)

	if Model=="Sup":
		modelPath = "../SupCheckpoints/19model.ckpt"
		image_name = "Test_out_Sup_"+str(num)+".jpg"
	else:
		modelPath = "../UnsupCheckpoints/19model.ckpt"
		image_name = "Test_out_Unsup_"+str(num)+".jpg"

	im, image1, output_homo, start, im1  = readImages(num, MaxPerturb)
	_,h4,_ =  model(im, im1.reshape([1,128,128,1]), modelPath)

	color1 = (0, 255, 0)
	color2 = (0, 0, 255)
	color3 = (255, 0, 0)
	thickness = 2

	image = drawPatch(image1, start, color1, thickness)
	image = drawPerturb(image, start, output_homo, color2, thickness)
	# image = drawPrediction(image, start, h4, color3, thickness)

	cv2.imwrite(image_name, image)
		
if __name__ == '__main__':
	main()
 
