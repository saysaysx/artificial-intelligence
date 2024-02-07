#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 17:18:22 2018

@author: say
"""

import sys
print(sys.path)


import cv2
import numpy as np
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,Flatten
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D,GlobalMaxPooling2D , MaxPooling2D, AveragePooling2D, LSTM
from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras import optimizers
#from tensorflow.keras.layers.merge import concatenate
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

#import theano
#print(theano.config.device)
from tensorflow.keras.utils import plot_model
from random import random
from math import *

from tensorflow.python.client import device_lib

import machine_learning as ml



import tensorflow as tf

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)

#session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

from tensorflow.keras.backend import set_session

#from tensorflow.compat.v1 import ConfigProto
#from tensorflow.compat.v1 import InteractiveSession

#config = ConfigProto()
#config.gpu_options.allow_growth = True
#sess = InteractiveSession(config=config)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)
#config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.5
session = tf.Session(config=config)
set_session(session)

print(device_lib.list_local_devices())

#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
#config.log_device_placement = True  # to log device placement (on which device the operation ran)
#config.gpu_options.per_process_gpu_memory_fraction = 0.7
#sess = tf.Session(config=config)
#set_session(sess)  # set this TensorFlow session as the default session for Keras

import keras.backend as K
#cfg = K.tf.ConfigProto(gpu_options={'allow_growth': True, 'per_process_gpu_memory_fraction': 0.5})

#config = K.tf.ConfigProto()
#config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.2
#sess = K.tf.Session(config=config)
#K.set_session(sess)


masmin = [(245,0,0),(0,245,0),(0,0,245),(245,0,245),(245,245,0)]
masmax = [(255,10,10),(10,255,10),(10,10,255),(255,10,255),(255,255,10)]

#sizeh = 120
#sizew = 160

sizeh = 45
sizew = 60

#sizeh = 30
#sizew = 40


nrows = cv2.getOptimalDFTSize(sizeh)
ncols = cv2.getOptimalDFTSize(sizew)

#print(nrows, ncols)


winSize = (sizew,sizeh)
blockSize = (10,10)
blockStride = (10,10)
cellSize = (5,5)
nbins = 9
derivAperture = 1
winSigma = 4.
histogramNormType = 0
L2HysThreshold = 2.0000000000000001e-01
gammaCorrection = True
nlevels = 64
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels)



sizehist = 1
nexample = 2831
nclone = 600
iexample = -1
nlearn = nexample*nclone

xtrain = [[]]*nlearn
#xtrain1 = [[]]*nlearn
xtrain2 = []
xtrain3 = []
ytrain = [[]]*nlearn
#2831


for i in range(nexample):
	print(i)
	img1 = cv2.imread("./ris/exampl/a" + str(i)+'.jpg')
	img2 = cv2.imread("./ris/exampl/b" + str(i)+'.jpg')
	#for k in range(3):
		#outflip = [0,0,1]
		#if(k==1): 
		#	img1 = cv2.flip(img1,1)
		#	img2 = cv2.flip(img2,1)
		#	outflip = [0,1,0]
		#if(k==2): 
		#	img1 = cv2.flip(img1,0)
		#	img2 = cv2.flip(img2,0)
		#	outflip = [1,0,0]
		
	gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
	gheight, gwidth = gray.shape[:2]


	ytr = []
	
	for j in range(5):
		h_min = np.array(masmin[j], np.uint8)
		h_max = np.array(masmax[j], np.uint8)
	
		col = cv2.inRange(img1, h_min, h_max)
		moments = cv2.moments(col, 2)
		dM01 = moments['m01']
		dM10 = moments['m10']
		dArea = moments['m00']
		x = int(dM10 / dArea)
		y = int(dM01 / dArea)
		x = x*float(sizew)/gwidth
		y = y*float(sizeh)/gheight
		ytr = ytr+[[x,y]]

	yn = [[0,0],[0,0],[0,0],[0,0],[0,0]]
	for k in range(nclone):
		iexample = iexample+1
		al = (random()-0.5)*2.5
		scale = (random()-0.01)*0.3+1
		dx = (random()-0.5)*200*float(sizew)/gwidth
		dy = (random()-0.5)*200*float(sizeh)/gheight
		dx0 = sizew/2.0
		dy0 = sizeh/2.0
		dxx = -cos(al)*dx0+sin(al)*dy0+dx+dx0
		dyy = -sin(al)*dx0-cos(al)*dy0+dy+dy0
		mpov = np.array([[cos(al)*scale,-sin(al)*scale, dxx],[sin(al)*scale,cos(al)*scale,dyy]])
		gray1 = cv2.resize(gray, (sizew, sizeh))
		gray1 = cv2.Canny(gray1, 10, 100)
		xc,yc = int(random()*sizew),int(random()*sizeh)
		small = cv2.warpAffine(gray1,mpov,(sizew,sizeh),borderValue=0)
		small1 = small.copy()
		sh2 = int(sizeh/2)
		sw2 = int(sizew/2)
		sh4 = int(sizeh/4)
		sw4 = int(sizew/4)
		
		
		#winStride = (8,8)
		#padding = (8,8)
		#locations = ((10,20),)
		#hist = hog.compute(small)#,winStride,padding,locations)
		#sizehist = len(hist)
		#corn = cv2.goodFeaturesToTrack(small, 200, 0.0001,10)
		#print(len(corn))
		#print(hist)

		
		#spec1 = small[0:sh2,0:sw2]
		#spec2 = small[sh2:sizeh,0:sw2]
		#spec3 = small[0:sh2,sw2:sizew]
		#spec4 = small[sh2:sizeh,sw2:sizew]
		#spec = np.array([spec1,spec2,spec3,spec4])
		#spec=spec.swapaxes(0,2)
		#spec = spec.swapaxes(0,1)
		#print(spec.shape)
		#spec1 = small[0:sh4,0:sizew]
		#spec2 = small[sh4:2*sh4,0:sizew]
		#spec3 = small[2*sh4:3*sh4,0:sizew]
		#spec4 = small[3*sh4:sizeh,0:sizew]
		#speca = np.array([spec1,spec2,spec3,spec4])
		#speca=speca.swapaxes(0,2)
		#speca = speca.swapaxes(0,1)

		#spec1 = small[0:sizeh,0:sw4]
		#spec2 = small[0:sizeh,sw4:sw4*2]
		#spec3 = small[0:sizeh,sw4*2:sw4*3]
		#spec4 = small[0:sizeh,sw4*3:sizew]
		#specb = np.array([spec1,spec2,spec3,spec4])
		#specb=specb.swapaxes(0,2)
		#specb = specb.swapaxes(0,1)
		#print(speca.shape)
		#print(specb.shape)
		#spectrum = np.expand_dims(spec, axis=2)
		
		ytr1 = []
		for l in range(5):
			yn[l][0] = mpov[0][0]*ytr[l][0]+mpov[0][1]*ytr[l][1]+mpov[0][2]
			yn[l][1] = mpov[1][0]*ytr[l][0]+mpov[1][1]*ytr[l][1]+mpov[1][2]
			x = yn[l][0]/sizew
			y = yn[l][1]/sizeh
			cv2.putText(small1,str(l+1), (int(x*sizew),int(y*sizeh)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,0,0),1)
			ytr1 = ytr1 + [yn[l][0],yn[l][1]]
		#for l1 in range(5):
		#	for l2 in range(l1+1,5):
		#		sq = sqrt((ytr[l1][0]-ytr[l2][0])**2+(ytr[l1][1]-ytr[l2][1])**2)
		#		ytr1 = ytr1+[sq]
		
		#ytr = ytr + outflip
		#dft1= cv2.dft(np.float32(small),flags=cv2.DFT_COMPLEX_OUTPUT)
		#print(len(dft1),len(dft1[0]),len(dft1[0][0]))
		#small = np.array(small, dtype=np.float32)/255.0
		#dft1 = np.insert(dft1,2,small,axis=2)
		#dft_shift = np.fft.fftshift(dft1)
		#spectr = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1])+1.0)
		#phase = cv2.phase(dft_shift[:,:,0],dft_shift[:,:,1])
		#spectrum = np.insert(spectrum,1,phase,axis=2)
		#edg = cv2.Canny(small, 10, 100)
		#spectrum = np.insert(spectrum,1,edg,axis=2)
		#spectrum = np.insert(spectrum,2,spectr,axis=2)
		#spectrum = np.insert(spectrum,3,phase,axis=2)
		#print(len(spectrum),len(spectrum[0]),len(spectrum[0][0]))
		#dft1 = np.insert(dft1,0,small,axis=2)
		

		#al = pi/4
		#mpov = np.array([[cos(al),-sin(al), 0],[sin(al),cos(al),0]])
		#small1 = cv2.warpAffine(small,mpov,(sizew,sizeh),borderValue=int(random()*220))
		#al = -pi/4
		#mpov = np.array([[cos(al),-sin(al), 0],[sin(al),cos(al),0]])
		#small2 = cv2.warpAffine(small,mpov,(sizew,sizeh),borderValue=int(random()*220))

		spectrum = np.expand_dims(small, axis=2)
		#spectrum1 = np.expand_dims(small1, axis=2)
		#spectrum2 = np.expand_dims(small2, axis=2)

		
		xtrain[iexample] = spectrum
		#xtrain1[iexample] = dft1
		#xtrain2 = xtrain2 + [spectrum2]
		#xtrain3 = xtrain3 + [specb]
		ytrain[iexample] = ytr1
		#small[:,:] = spectrum[:,:,0]
		#print(small)
		cv2.imshow('img1',small1)
		key = cv2.waitKey(1)
		if(key==27): break

cap = cv2.VideoCapture("./ris/output4.mp4")
#1140
for i in range(0):
	print(i)
	ret,img2 = cap.read()
	for k in range(3):
		if(k==1): 
			img2 = cv2.flip(img2,1)
		if(k==2): 
			img2 = cv2.flip(img2,0)

		gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
		small = cv2.resize(gray, (sizew, sizeh))
		ytr = []
		for j in range(5):
			ytr = ytr+[1e-9,1e-9]
		ytr = ytr+[1e-9,0.99999999]
		small = np.array(small, dtype=np.float32)/255.0
		xtrain = xtrain + [small]
		ytrain = ytrain+[ytr]
		cv2.imshow('img1',small)
		key = cv2.waitKey(1)
		if(key==27): break

szout = len(ytrain[0])
xtrain = np.array(xtrain)
#xtrain1 = np.array(xtrain1)
#xtrain2 = np.array(xtrain2)
#xtrain3 = np.array(xtrain3)
ytrain = np.array(ytrain)
#minyt,maxyt, ytrain = ml.norm_learntrain(ytrain)



#xtrain = np.expand_dims(xtrain, axis=3)





def addwithpool(x,fsz,pool):
	f1 = BatchNormalization()(x)
	fpoolx = MaxPooling2D(pool_size=(pool, pool))(f1)
	f1 = Conv2D(fsz, (1, 1), activation='relu', padding='same') (fpoolx)
	f1 = BatchNormalization()(f1)
	f1 = Conv2D(fsz, (3, 3), activation='relu', padding='same') (f1)
	f1 = BatchNormalization()(f1)
	f1 = Conv2D(fsz, (3, 3), activation='relu', padding='same') (f1)
	#merge = concatenate([f1,fpoolx])
	merge = keras.layers.add([f1,fpoolx])
	return merge 

def addwithoutpool(x,fsz,pool):
	ff = BatchNormalization()(x)
	ff = Conv2D(4, (1, 1), activation='relu', padding='same') (ff)
	f1 = BatchNormalization()(ff)
	f1 = Conv2D(fsz, (3, 3), activation='relu', padding='same') (f1)
	f1 = BatchNormalization()(f1)
	f1 = Conv2D(fsz, (3, 3), activation='relu', padding='same') (f1)
	#merge = concatenate([f1,fpoolx])
	merge = keras.layers.add([ff,f1])
	return merge 

def addmanywithpool(x,fsz,pool):
	f1 = BatchNormalization()(x)
	fpoolx = MaxPooling2D(pool_size=(pool, pool))(f1)
	f1 = Conv2D(2, (1, 1), activation='relu', padding='same') (fpoolx)
	f2 = Conv2D(fsz, (3, 3), activation='relu', padding='same') (fpoolx)
	f3 = Conv2D(fsz, (5, 5), activation='relu', padding='same') (fpoolx)
	#merge = concatenate([f1,fpoolx])
	merge = concatenate([f1,f2,f3,fpoolx])
	return merge 

def addmany(x,fsz,pool):
	f1 = Conv2D(4, (1, 1), activation='relu', padding='same') (x)
	f2 = Conv2D(fsz, (3, 3), activation='relu', padding='same') (x)
	f3 = Conv2D(fsz, (5, 5), activation='relu', padding='same') (x)
	#merge = concatenate([f1,fpoolx])
	merge = concatenate([f1,f2,f3])
	return merge 

def addwp(x,fpred,fsz,pool):
	f1 = BatchNormalization()(x)
	fpoolx = MaxPooling2D(pool_size=(pool, pool))(f1)
	f1 = Conv2D(fpred, (1, 1), activation='relu', padding='same') (fpoolx)
	f2 = Conv2D(fsz, (3, 3), activation='relu', padding='same') (fpoolx)
	f3 = Conv2D(fsz, (5, 5), activation='relu', padding='same') (fpoolx)
	#merge = concatenate([f1,fpoolx])
	merge = concatenate([f1,f2,f3])
	return merge 
def addw(x,fpred,fsz,pool):
	f1 = Conv2D(fpred, (1, 1), activation='relu', padding='same') (x)
	f2 = Conv2D(fsz, (3, 3), activation='relu', padding='same') (x)
	f3 = Conv2D(fsz, (5, 5), activation='relu', padding='same') (x)
	#merge = concatenate([f1,fpoolx])
	merge = concatenate([f1,f2,f3])
	return merge 


def addwp2(x,fsz,pool,filt1,filt2):
	#f1 = BatchNormalization()(x)
	fpoolx = MaxPooling2D(pool_size=(pool, pool))(x)
	f2 = Conv2D(fsz, (filt1, filt1), activation='relu', padding='same') (fpoolx)
	f3 = Conv2D(fsz, (filt2, filt2), activation='relu', padding='same') (fpoolx)
	#merge = concatenate([f1,fpoolx])
	merge = tensorflow.keras.layers.add([f2,f3])
	return merge 
def addw2(x,fsz,filt1,filt2):
	#f1 = BatchNormalization()(x)
	f2 = Conv2D(fsz, (filt1, filt1), activation='relu', padding='same') (x)
	f3 = Conv2D(fsz, (filt2, filt2), activation='relu', padding='same') (x)
	#merge = concatenate([f1,fpoolx])
	merge = tensorflow.keras.layers.add([f2,f3])
	return merge 


def addwp4(x,fsz,pool,filt1,filt2):
	f1 = Conv2D(fsz, (1,1), activation='relu', padding='same') (x)
	f2 = Conv2D(fsz, (filt1, filt1), activation='relu', padding='same') (x)
	f3 = Conv2D(fsz, (filt2, filt2), activation='relu', padding='same') (x)
	merge = tensorflow.keras.layers.add([f2,f3,f1])
	fpoolx = MaxPooling2D(pool_size=(pool, pool))(merge)
	return fpoolx
def addw4(x,fsz,filt1,filt2):
	f2 = Conv2D(fsz, (filt1, filt1), activation='relu', padding='same') (x)
	f3 = Conv2D(fsz, (filt2, filt2), activation='relu', padding='same') (x)
	merge = tensorflow.keras.layers.add([f2,f3,x])
	return merge 


def addwp3(x,fsz,pool,filt1):
	#f1 = BatchNormalization()(x)
	f2 = Conv2D(fsz, (filt1, filt1), activation='relu', padding='same')(x)
	fpoolx = MaxPooling2D(pool_size=(pool, pool))(f2)
	return fpoolx





visible1 = Input(shape=(sizeh, sizew, 1))
lay = Conv2D(20, (5, 5), activation='relu', padding='same') (visible1)
lay = MaxPooling2D(pool_size=(2, 2))(lay)
lay = Conv2D(40, (7, 7), activation='relu', padding='same') (lay)
lay = MaxPooling2D(pool_size=(2, 2))(lay)
lay = Conv2D(40, (5, 5), activation='relu', padding='same') (lay)
lay = MaxPooling2D(pool_size=(2, 2))(lay)
lay = Conv2D(20, (3, 3), activation='relu', padding='same') (lay)
#lay = MaxPooling2D(pool_size=(2, 2))(lay)
lay = Flatten()(lay)
lay = BatchNormalization()(lay)
lay = Dense(512, activation='relu')(lay)
lay = Dense(512, activation='relu')(lay)
laya = Dense(524, activation='relu')(lay)


#visible2 = Input(shape=(sizeh, sizew, 2))
#lay = Conv2D(20, (5, 5), activation='relu', padding='same') (visible2)
#lay = MaxPooling2D(pool_size=(2, 2))(lay)
#lay = Conv2D(40, (7, 7), activation='relu', padding='same') (lay)
#lay = MaxPooling2D(pool_size=(2, 2))(lay)
#lay = Conv2D(40, (5, 5), activation='relu', padding='same') (lay)
#lay = MaxPooling2D(pool_size=(2, 2))(lay)
#lay = Conv2D(20, (3, 3), activation='relu', padding='same') (lay)
#lay = MaxPooling2D(pool_size=(2, 2))(lay)
#lay = Flatten()(lay)
#lay = BatchNormalization()(lay)
#lay = Dense(512, activation='relu')(lay)
#lay = Dense(512, activation='relu')(lay)
#layb = Dense(312, activation='relu')(lay)





#merge = tensorflow.keras.layers.concatenate([laya,layb])
dens = Dense(624, activation='relu')(laya)
dens = Dense(624, activation='relu')(dens)
dens = Dense(328, activation='relu')(dens)
dens = Dense(50, activation='relu')(dens)
densout = Dense(szout, activation='linear')(dens)


model = Model(inputs = [visible1],outputs = densout)


plot_model(model, to_file='./out/model1.png', show_shapes=True)


meth = ["Adam","Adam","Adam"]
bs = [5000]
#model.load_weights("model22.h5")
#opt = optimizers.SGD(lr=0.0000005, decay=1e-12, momentum=0.0000001, nesterov=True)
opt = optimizers.Adam(lr=0.0001)
#opt = optimizers.Adam(lr=0.001)



for i in range(200):
	num1 = int(random()*len(meth))
	eps = int(180)
	numb = int(random()*len(bs))
	print(meth[num1])
	#if(i>0):
	model.load_weights("model61.h5")
	
	momentum = 0.9*random()*random()
	f = False
	if(random()>0.5):
		f=True
	
	model.compile(loss='mean_squared_error',
			optimizer=opt,
			metrics=['accuracy'])
	
	model.fit( [xtrain], ytrain, batch_size=bs[numb], epochs=50)
	
	model.save_weights("model61.h5")
result = model.predict([xtrain])
#result = ml.unnorm_train(result,minyt,maxyt)

#val = result - ytrain
#print(val)



for i in range(50):
	print(i)
	img2 = cv2.imread("./ris/exampl/b" + str(i)+'.jpg')
	print("-----")
	for j in range(5):
		x = int(result[i][j*2]*float(gwidth)/sizew)
		y = int(result[i][j*2+1]*float(gheight)/sizeh)
		if(x<0):
			x=0
		if(y<0):
			y=0
		if(x>639):
			x=639
		if(y>639):
			y=479
		print("rx,ry = ", x,y)
		print("tx,ty = ", int(ytrain[i][j*2]*float(gwidth)/sizew),int(ytrain[i][j*2+1]*float(gheight)/sizeh))

		#if(result[i][10]>0.5):
		cv2.putText(img2,str(j+1), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,0,0),1)
	cv2.imshow('img2',xtrain[i])
	key = cv2.waitKey(50)


