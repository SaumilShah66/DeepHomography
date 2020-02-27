import glob
import cv2
import numpy as np 
import random
import os

def imgCorners(img):
	features = cv2.goodFeaturesToTrack(img, 1500, 0.03,10)
	try:
		Nstrong = features.shape[0]
		if(Nstrong >= 15):
			return True
		else:
			return False
	except:
		return False

def randomFlip(img):
	if(random.randint(0,1)):
		return cv2.flip(img,1)
	else:
		return img

def randomBrightness(img):
	return img + random.randint(-10,10)

def getImage(name):
	im = np.float32(cv2.imread(name,0))
	im_ = randomFlip(im)
	return randomBrightness(im_)

def generatePatches(img, patchSize, p):
	h,w = img.shape
	lim_x, lim_y = int(3*w/5 - patchSize - p), int(3*h/5 - patchSize - p)
	if(lim_x>p/2 and lim_y>p/2):
		x_start, y_start = (int(w/5) + random.randint(int(p/2), lim_x)), (int(h/5)+random.randint(int(p/2), lim_y))
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
		im2 = im2_[y_start:y_start + patchSize, x_start:x_start + patchSize]
		#### generate input array	
		# im_in = np.zeros((patchSize,patchSize,2))				
		# im_in[:,:,0] = (im1 - 127.0)/127.0
		# im_in[:,:,1] = (im2 - 127.0)/127.0
		output_homo = []
		for i in range(4):
			output_homo.append(u[i])
			output_homo.append(v[i])
		output_homo = np.array(output_homo)
		if imgCorners(im1):
			return im1,im2,output_homo
		else:
			return 0,0,0
	else:
		return 0,0,0

path = "../Data/Train/*.jpg"
newPath = "../Data/AugTrain/"
path_test = "../Data/Val/*.jpg"
newPath_test = "../Data/AugVal/"
try:
	os.mkdir(newPath)
except:
	pass

try:
	os.mkdir(newPath_test)
except:
	pass

imgNames = glob.glob(path)
imgNames_test = glob.glob(path_test)

patchSize = 128
perturbSize = patchSize/8
i = 0
im1_names = []
im1_filename = "TxtFiles/data1.txt"
im2_names = []
im2_filename = "TxtFiles/data2.txt"
all_perturbs = []
all_perturbs_file = "TxtFiles/perturbs.csv"

im1_test_names = []
im1_test_filename = "TxtFiles/val_data1.txt"
im2_test_names = []
im2_test_filename = "TxtFiles/val_data2.txt"
all_test_perturbs = []
all_test_perturbs_file = "TxtFiles/val_perturbs.csv"

############################ Train Data ############################
###################### Generating training images ##################
for imageName in imgNames:
	img = getImage(imageName)
	for j in range(10):
		im1,im2,homo = generatePatches(img, patchSize, perturbSize)
		try:
			if im1==0:
				print("----------\n\n\nMissed  "+str(imageName)+"\n\n\n-------")
		except:
			cv2.imwrite(newPath+str(i+1)+"_1.jpg", im1)
			cv2.imwrite(newPath+str(i+1)+"_2.jpg", im2)
			im1_names.append(newPath+str(i+1)+"_1.jpg")
			im2_names.append(newPath+str(i+1)+"_2.jpg")
			all_perturbs.append(homo)
			i=i+1
			print(str(i+1)+" writter !!")
			pass
######################################################################
################## Shuffling train data ##############################
print("Suffling data....")
for j in range(10):
	ran = random.randint(1,10)
	random.Random(ran).shuffle(im1_names)
	random.Random(ran).shuffle(im2_names)
	random.Random(ran).shuffle(all_perturbs)
######################################################################
################# Writing data into files ############################
print("Writing Data 1 names")
with open(im1_filename, 'w') as output:
    for row in im1_names:
        output.write(str(row) + '\n')
print("Writing Data 2 names")
with open(im2_filename, 'w') as output:
    for row in im2_names:
        output.write(str(row) + '\n')

print("Writing outputs")
np.savetxt(all_perturbs_file, np.array(all_perturbs), delimiter=",")
#####################################################################
#####################################################################

######################### Test data ##################################

###################### Generating training images ####################
for imageName in imgNames_test:
	img = getImage(imageName)
	for j in range(5):
		im1,im2,homo = generatePatches(img, patchSize, perturbSize)
		try:
			if im1==0:
				print("----------\n\n\nMissed  "+str(imageName)+"\n\n\n-------")
		except:
			cv2.imwrite(newPath_test+str(i+1)+"_1.jpg", im1)
			cv2.imwrite(newPath_test+str(i+1)+"_2.jpg", im2)
			im1_test_names.append(newPath_test+str(i+1)+"_1.jpg")
			im2_test_names.append(newPath_test+str(i+1)+"_2.jpg")
			all_test_perturbs.append(homo)
			i=i+1
			print(str(i+1)+" writter !!")
			pass
######################################################################
################## Shuffling train data ##############################
print("Suffling data....")
for j in range(10):
	ran = random.randint(1,10)
	random.Random(ran).shuffle(im1_test_names)
	random.Random(ran).shuffle(im2_test_names)
	random.Random(ran).shuffle(all_test_perturbs)
######################################################################
################# Writing data into files ############################
print("Writing Data 1 names")
with open(im1_test_filename, 'w') as output:
    for row in im1_test_names:
        output.write(str(row) + '\n')
print("Writing Data 2 names")
with open(im2_test_filename, 'w') as output:
    for row in im2_test_names:
        output.write(str(row) + '\n')

print("Writing outputs")
np.savetxt(all_test_perturbs_file, np.array(all_test_perturbs), delimiter=",")
######################################################################
######################################################################