#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 17:18:02 2019

@author: shailesh
"""

import numpy as np
import os
import random

partNos = {}

		
def loadNextPart(dataFolder):
	global partNos
	if dataFolder not in partNos.keys():
		partNos[dataFolder] = 0
	if partNos[dataFolder] >= len(os.listdir(dataFolder)):
		partNos.pop(dataFolder)
		return 0
		
	part = np.load(dataFolder + '/part_'+str(partNos[dataFolder]) + '.npy')
	partNos[dataFolder]+=1
	
	return part


def getPartsBatch(bsize,XFolder,YFolder):
	n_parts = len(os.listdir(XFolder))
	partInd = random.choice(range(n_parts))
#	print(XFolder+'/part_'+str(partInd)+'.npy')
	X_train = np.load(XFolder+'/part_'+str(partInd)+'.npy')
	Y_train = np.load(YFolder+'/part_'+str(partInd)+'.npy')
#	print(X_train.size)
	seed = np.random.choice(X_train.shape[0], size = bsize, replace = False)
	xb = X_train[seed,:]
	yb = Y_train[seed,:]
	return xb,yb

def getSeqBatch(x_train,bs,batchInd):
	return x_train[bs*batchInd:bs*(batchInd+1)]


