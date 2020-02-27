"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

import os
import cv2
import numpy as np
import random
import skimage
import PIL
import sys
# Don't generate pyc codes
sys.dont_write_bytecode = True

def SetupAll(BasePath, CheckPointPath):
    """
    Inputs: 
    BasePath is the base path where Images are saved without "/" at the end
    CheckPointPath - Path to save checkpoints/model
    Outputs:
    DirNamesTrain - Variable with Subfolder paths to train files
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    ImageSize - Size of the image
    NumTrainSamples - length(Train)
    NumTestRunsPerEpoch - Number of passes of Val data with MiniBatchSize 
    Trainabels - Labels corresponding to Train
    NumClasses - Number of classes
    """
    # Setup DirNames
    DirNamesTrain1, DirNamesTrain2 = SetupDirNames(BasePath)

    # Read and Setup Labels
    LabelsPathTrain = './TxtFiles/perturbs.csv'
    TrainLabels = ReadLabels(LabelsPathTrain)

    # If CheckPointPath doesn't exist make the path
    if(not (os.path.isdir(CheckPointPath))):
       os.makedirs(CheckPointPath)
        
    # Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    SaveCheckPoint = 100 
    # Number of passes of Val data with MiniBatchSize 
    NumTestRunsPerEpoch = 5
    
    # Image Input Shape
    ImageSize = [128, 128, 2]
    NumTrainSamples = len(DirNamesTrain1)

    # Number of classes
    NumClasses = 8

    return DirNamesTrain1, DirNamesTrain2, SaveCheckPoint, ImageSize, NumTrainSamples, TrainLabels, NumClasses

def setupValidation(BasePath):
    DirNamesValid1, DirNamesValid2 = SetupValidDirNames(BasePath)
    LabelsPathValid = './TxtFiles/val_perturbs.csv'
    ValidLabels = ReadValidLabels(LabelsPathValid)
    NumValidSamples = len(DirNamesValid1)
    return DirNamesValid1, DirNamesValid2, NumValidSamples, ValidLabels

def ReadLabels(LabelsPathTrain):
    if(not (os.path.isfile(LabelsPathTrain))):
        print('ERROR: Train Labels do not exist in '+LabelsPathTrain)
        sys.exit()
    else:
        TrainLabels = open(LabelsPathTrain, 'r')
        TrainLabels = TrainLabels.read()
        TrainLabels = TrainLabels.split()
        allLabels = [np.array(label.split(","),dtype="f") for label in TrainLabels]
    return allLabels
    
def ReadValidLabels(LabelsPathValid):
    if(not (os.path.isfile(LabelsPathValid))):
        print('ERROR: Validation Labels do not exist in '+LabelsPathValid)
        sys.exit()
    else:
        ValidLabels = open(LabelsPathValid, 'r')
        ValidLabels = ValidLabels.read()
        ValidLabels = ValidLabels.split()
        allLabels = [np.array(label.split(","),dtype="f") for label in ValidLabels]
    return allLabels

def SetupDirNames(BasePath): 
    """
    Inputs: 
    BasePath is the base path where Images are saved without "/" at the end
    Outputs:
    Writes a file ./TxtFiles/DirNames.txt with full path to all image files without extension
    """
    DirNamesTrain1 = ReadDirNames('./TxtFiles/data1.txt')        
    DirNamesTrain2 = ReadDirNames('./TxtFiles/data2.txt')
    return DirNamesTrain1, DirNamesTrain2

def ReadDirNames(ReadPath):
    """
    Inputs: 
    ReadPath is the path of the file you want to read
    Outputs:
    DirNames is the data loaded from ./TxtFiles/DirNames.txt which has full path to all image files without extension
    """
    # Read text files
    DirNames = open(ReadPath, 'r')
    DirNames = DirNames.read()
    DirNames = DirNames.split()
    return DirNames

def SetupValidDirNames(BasePath): 
    """
    Inputs: 
    BasePath is the base path where Images are saved without "/" at the end
    Outputs:
    Writes a file ./TxtFiles/DirNames.txt with full path to all image files without extension
    """
    DirNamesValid1 = ReadDirNames('./TxtFiles/val_data1.txt')        
    DirNamesValid2 = ReadDirNames('./TxtFiles/val_data2.txt')
    return DirNamesValid1, DirNamesValid2

