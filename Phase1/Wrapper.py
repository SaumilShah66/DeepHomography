#!/usr/bin/evn python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project1: MyAutoPano: Phase 1 Starter Code

Author(s): 
Varun Asthana (vasthana@umd.edu) 
Graduate Student in Robotics,
University of Maryland, College Park

Saumil Shah (sshah293@terpmail.umd.edu)
Graduate Student in Robotics,
University of Maryland, College Park
"""


import sys
# sys.path.remove(sys.path[1])
import cv2
import numpy as np
import glob
import copy
import matplotlib.pyplot as plt
import argparse

def imgCorners(img, imageName):
	imgCopy=np.copy(img)
	imgCopy2=np.copy(img)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	gray = np.float32(gray)
	features = cv2.goodFeaturesToTrack(gray, 1500, 0.02,10)
	h,w = img.shape[0],img.shape[1]
	Nstrong = features.shape[0]
	for i in range(Nstrong):
		w_,h_ = int(features[i][0][0]),int(features[i][0][1])
		cv2.circle(imgCopy, (w_,h_), 3, 255, -1)
	cv2.imwrite('corners'+imageName+'.png',imgCopy)
	r = np.ones((Nstrong,1))*(float('inf'))
	print("N strong = "+str(Nstrong))
	if(Nstrong>1000):
		Nbest=600
		print(" Setting 600 best points")
	elif(Nstrong>600):
		Nbest=300
		print(" Setting 300 best points ")
	else:
		Nbest=Nstrong
		print(" Setting same best points as Nstrong ")

	for i in range(Nstrong):
		if(i%500==0):
			print('Nstrong= '+str(Nstrong)+' and i= '+ str(i))
		for j in range(0,Nstrong):
			yi,xi = int(features[i][0][0]),int(features[i][0][1])
			yj,xj = int(features[j][0][0]),int(features[j][0][1])
			if(gray[xj,yj] > gray[xi,yi]):
				ed= (xj - xi)**2 + (yj - yi)**2
			else:
				ed= float('inf')
			if(ed<r[i]):
				r[i]=ed
	points=np.zeros((Nstrong,3), dtype='f')

	for i in range(Nstrong):
		points[i][0]= r[i][0]
		points[i][1]= features[i][0][0]
		points[i][2]= features[i][0][1]

	pointsSorted = points[points[:,0].argsort()[::-1]]
	temp=pointsSorted[0:Nbest,1:3]

	for i in range(Nbest):
		w_, h_ = points[i][1], points[i][2]
		cv2.circle(imgCopy2, (w_,h_), 3, 255, -1)
	cv2.imwrite('anms'+imageName+'.png',imgCopy2)
	return np.array(points[:Nbest,1:3], dtype=int)

def featureMap(img, img_cords, img_name):
	imgFM=[]
	h, w = img.shape[0], img.shape[1]
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	gray = np.float32(gray)
	patch_size = 40
	patch_pad = int(patch_size/2)
	best_w = img_cords[:,0] + patch_pad
	best_h = img_cords[:,1] + patch_pad
	image_patch = np.zeros((int(h+patch_size), int(w+patch_size)))
	image_patch[patch_pad:patch_pad+h, patch_pad:patch_pad+w] = gray
	for i in range(best_h.shape[0]):
		tempH= best_h[i]
		tempW= best_w[i]
		tempImg= image_patch[tempH-patch_pad:tempH+patch_pad,tempW-patch_pad:tempW+patch_pad]
		tempBlurr= cv2.GaussianBlur(tempImg,(3,3),1.5)
		image_patch[tempH-patch_pad:tempH+patch_pad,tempW-patch_pad:tempW+patch_pad]= tempBlurr
		tempVec= (cv2.resize(tempBlurr,(8,8))).reshape(-1,1)
		imgFM.append(tempVec)
	cv2.imwrite('FD'+img_name+'.png',image_patch)
	return np.array(imgFM)

def imageFeatureMatch(img1_fmap, img2_fmap, thresh):
	matchedPairs=[]
	pairFound= False
	for i in range(len(img1_fmap)):
		img1_vec= (img1_fmap[i] - np.mean(img1_fmap[i]))/np.std(img1_fmap[i])
		min1= float('inf')
		min2= float('inf')
		matchedIndx=0
		for j in range(len(img2_fmap)):
			img2_vec= (img2_fmap[j] - np.mean(img2_fmap[j]))/np.std(img2_fmap[j])
			dist= ((img2_vec - img1_vec)**2).sum()
			if(min1>dist):
				min2= min1
				min1= dist
				matchedIndx=j # with i of image1
		if((min1/min2)<thresh):
			matchedPairs.append(np.array([i,matchedIndx]))

	if(len(matchedPairs)>30):
		pairFound= True
	else:
		print('2 images did not match as only matches found are ', len(matchedPairs))
		matchedPairs=[]
	return pairFound, np.array(matchedPairs)

def plotMatch(img, img1_cords, img2_cords, w1, name):
	imgNew= np.copy(img)
	for i in range(len(img1_cords)-1):
		cv2.line(imgNew,(img1_cords[i][0], img1_cords[i][1]),(img2_cords[i][0]+w1, img2_cords[i][1]),[0,255,0],1)
	cv2.imwrite('matching'+ name, imgNew)
	return


def ransac(img1_matched_cords, img2_matched_cords):
	Nmax=10000
	img1_trans= np.ones((img1_matched_cords.shape[0],3))
	img1_trans[:,0:2]= img1_matched_cords
	inlier_max= 0
	randomPairs= []
	for itr in range(Nmax):
		totMatch=np.arange(img1_matched_cords.shape[0])
		select=np.random.choice(totMatch, size=4)
		
		select.sort()
		flag= True
		while(flag):
			flag= False
			for x in range(len(randomPairs)):
				comp= select== randomPairs[x]
				if(comp.all()):
					select=np.random.choice(totMatch, size=4)
					select.sort()
					flag= True
					break
		randomPairs.append(select)
		homo_img1=[]
		homo_img2=[]
		for val in select:
			homo_img1.append(img1_matched_cords[val]) #[[w1,h1], [w2,h2], [w3,h3], [w4,h4]]
			homo_img2.append(img2_matched_cords[val])
		homo_img1= np.array(homo_img1, dtype='f')
		homo_img2= np.array(homo_img2, dtype='f')
		h = cv2.getPerspectiveTransform(homo_img1, homo_img2)
		img1_trans = np.matmul(h,img1_trans.T)
		img1_trans=(img1_trans/img1_trans[2,:]).T
		ssd= ((img2_matched_cords-img1_trans[:,:2])**2).sum(axis=1)
		if(itr%(Nmax/10)==0):
			print(itr)
		ssd_thresh=10**(np.log10(ssd.mean())/2.5)
		ssd_imp= (ssd<ssd_thresh)
		inlier_ratio= float(ssd_imp.sum())/ssd_imp.shape[0]
		if(inlier_max<inlier_ratio):
			inlier_max= inlier_ratio
			h_best= h
			ssd_fin= ssd_imp
		if(inlier_ratio>0.85):
			h_best= h
			ssd_fin= ssd_imp
			break
	inliers_img1_cords= []
	inliers_img2_cords= []
	inliers_img1_cords, inliers_img2_cords = img1_matched_cords[ssd_fin], img2_matched_cords[ssd_fin]
	return inlier_max, inliers_img1_cords, inliers_img2_cords, h_best


def homography(img1, img2, h):
	h1,w1,_ = img1.shape
	h2,w2,_ = img2.shape
	pts1 = np.array([[0,0],[0,h1],[w1,h1],[w1,0]], dtype='f').reshape(-1,1,2)
	pts2 = np.array([[0,0],[0,h2],[w2,h2],[w2,0]], dtype='f').reshape(-1,1,2)
	pts1_ = cv2.perspectiveTransform(pts1, h)
	pts = np.concatenate((pts1_, pts2), axis=0)
	xmin, ymin = (pts.min(axis=0).ravel() - 0.5)
	xmax, ymax = (pts.max(axis=0).ravel() + 0.5)
	xmin, xmax, ymin, ymax= int(xmin), int(xmax), int(ymin), int(ymax)
	t = [-xmin,-ymin]
	Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate
	mergedImg= np.zeros([ymax-ymin,xmax-xmin,3], dtype= 'uint8')
	mergedImg = cv2.warpPerspective(img1,Ht.dot(h),((xmax-xmin, ymax-ymin)))
	mergedImg_gray = cv2.cvtColor(mergedImg, cv2.COLOR_BGR2GRAY)
	commonMerg=[]
	commonImg=[]
	imgSrc= img2.copy()
	i=0
	for hi in range(-ymin,h2-ymin):
		j=0
		for wi in range(-xmin, w2-xmin):
			if(mergedImg_gray[hi,wi]>0):
				imgSrc[i,j]= mergedImg[hi,wi]
			j+=1
		i+=1
	src_mask = np.full(imgSrc.shape, 255, dtype = np.uint8)
	center = ((w2)/2,(h2)/2)

	# Clone seamlessly.
	imgNew = cv2.seamlessClone(img2, imgSrc, src_mask, center, cv2.NORMAL_CLONE)
	mergedImg[-ymin:h2-ymin,-xmin:w2-xmin]= imgNew    
	return mergedImg

def main(path):
	filenames= glob.glob(path)
	imageNames= []
	imgs=[]
	allMaps= []
	allCords= []
	if(len(filenames)<=1):
		print('Not enough images to check for Panoroma')
		return
	for i in range(len(filenames)):
		name=filenames[i].split('/')
		tempName=name[len(name)-1]
		tempName= tempName.split('.')[0]
		img= cv2.imread(filenames[i])
		f_cord= imgCorners(img, tempName)
		f_map= featureMap(img, f_cord, tempName)
		imageNames.append(tempName)
		imgs.append(img)
		allCords.append(f_cord)
		allMaps.append(f_map)

	finalImg= copy.deepcopy(imgs[0])
	del imgs[0]
	finalCord= copy.deepcopy(allCords[0])
	del allCords[0]
	finalMap= copy.deepcopy(allMaps[0])
	del allMaps[0]
	finalName= copy.deepcopy(imageNames[0])
	del imageNames[0]
	i=0
	imgsPaired= []
	Homo_best=[]
	imgLeft= len(imgs)
	j=0
	while(i<len(imgs)):
		imgLeft= len(imgs)
		pairFound= False
		image_1= finalImg
		image_2= imgs[i]
		pairFound, matchedPairs= imageFeatureMatch(allMaps[i], finalMap, 0.5)
		if(pairFound):
			j=0
			print('In img '+ str(imageNames[i]) +' and ' + str(finalName))
			print('Pairs matched = ', len(matchedPairs))
			h1,w1,_ = image_1.shape
			h2,w2,_ = image_2.shape    
			new = np.zeros((max(h1,h2),w1+w2,3),dtype='uint8')
			new[:h1, :w1,:] = image_1
			new[:h2, w1:,:] = image_2
			img1_matched_cords= []
			img2_matched_cords= []
			for a in range(len(matchedPairs)):
				img1_matched_cords.append(finalCord[matchedPairs[a][1]])
				img2_matched_cords.append(allCords[i][matchedPairs[a][0]])
			img1_mc=np.array(img1_matched_cords)
			img2_mc=np.array(img2_matched_cords)
			plotMatch(new, img1_mc, img2_mc, w1, str(finalName)+'_'+str(imageNames[i])+'.png')
			count= 1
			inlier_max=0
			while(count<=8):
				inlier_ratio, inliers_img1, inliers_img2, h = ransac(img1_mc, img2_mc)
				print('Inlier Ratio ', inlier_ratio)
				if(inlier_max<inlier_ratio):
					inlier_max= inlier_ratio
					h_best= h
				if(inlier_max> 0.5):
					break
				else:
					count+=1
			print('Inlier Max ', inlier_max)
			h_ls= cv2.findHomography(inliers_img1, inliers_img2, method=0)
			if(h_ls[0].any()):
				h_best= h_ls[0]
			Homo_best.append(h_best)
			plotMatch(new, inliers_img1, inliers_img2, w1, '_ransac_'+str(imageNames[i])+'_'+str(finalName)+'.png')
			mergeImg= homography(image_1, image_2, h_best)
			cv2.imwrite('mypano'+str(imageNames[i])+'_'+str(finalName)+'.png', mergeImg)
			plt.imshow(mergeImg)
			del imgs[i]
			del allCords[i]
			del allMaps[i]
			del imageNames[i]
			i-= 1
			finalImg= mergeImg
			finalName= 'merge'
			finalCord= imgCorners(finalImg, finalName)
			finalMap= featureMap(finalImg, finalCord, finalName)
		else:
			j+=1
			if(j==imgLeft):
				break
			else:
				imgs.append(imgs[i])
				allCords.append(allCords[i])
				allMaps.append(allMaps[i])
				imageNames.append(imageNames[i])
				del imgs[i]
				del allCords[i]
				del allMaps[i]
				del imageNames[i]
				i-= 1
		i+=1
	return

if __name__=='__main__':
	Parser = argparse.ArgumentParser()
	Parser.add_argument('--ImageDirectory', default="../Data/Train/Set2", help='Give the path to directory containing all the images')
	Args = Parser.parse_args()
	ImagesPath = Args.ImageDirectory
	path = ImagesPath + "/*.jpg"
	main(path)

