#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 23:23:40 2019

@author: shailesh
"""
import numpy as np
import random as random

X = np.load('Xtrain.npy')
Y = np.load('Ytrain.npy')

Xpath = 'parsimonius_X'
Ypath = 'parsimonius_Y'


nParts = 6
for i in range(1,nParts+1):
	rowList = random.sample(range(len(X)),20000)
	Xpar = X[rowList]
	Ypar = Y[rowList]
	np.save(Xpath+'_' +str(i)+'.npy',Xpar)
	np.save(Ypath+'_'+str(i) + '.npy',Ypar)
	


