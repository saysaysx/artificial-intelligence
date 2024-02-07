#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 13:28:42 2019

@author: say
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 17:18:22 2018

@author: say
"""
import cv2
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout,Flatten
from keras.layers import Embedding
from keras.layers import LeakyReLU
from keras.layers import Conv2D, GlobalAveragePooling2D, MaxPooling2D, AveragePooling2D
from keras.layers import BatchNormalization


from keras.layers.merge import concatenate
from keras.layers import Input
from keras.models import Model

#import theano
#print(theano.config.device)
from keras.utils.vis_utils import plot_model
from random import random





masmin = [(245,0,0),(0,245,0),(0,0,245),(245,0,245),(245,245,0)]
masmax = [(255,10,10),(10,255,10),(10,10,255),(255,10,255),(255,255,10)]

sizeh = 120
sizew = 160

xtrain = []
ytrain = []
#2831
for i in range(2831):

	print(i)
	img1 = cv2.imread("./hands/exampl/a" + str(i)+'.jpg')
	img2 = cv2.imread("./hands/exampl/b" + str(i)+'.jpg')
	for k in range(3):
		if(k==1): 
			img1 = cv2.flip(img1,1)
			img2 = cv2.flip(img2,1)
		if(k==2): 
			img1 = cv2.flip(img1,0)
			img2 = cv2.flip(img2,0)
			
		gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
		small = cv2.resize(gray, (sizew, sizeh))
		ytr = []
		small1 = small.copy()
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
			x = x/640.0
			y = y/480.0
			#ytr = ytr+[x,y]

			
			cv2.putText(small1,str(j+1), (int(x*sizew),int(y*sizeh)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,0,0),1)

		ytr = ytr + [1, 0]
		small = np.array(small, dtype=np.float32)/255.0
		xtrain = xtrain + [small]
		ytrain = ytrain+[ytr]

		cv2.imshow('img1',small1)
		key = cv2.waitKey(1)
		if(key==27): break

cap = cv2.VideoCapture("./hands/output.mp4")
#1140
for i in range(1140):
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
		#for j in range(5):
		#	ytr = ytr+[1e-9,1e-9]
		ytr = ytr+[0,1]
		small = np.array(small, dtype=np.float32)/255.0
		xtrain = xtrain + [small]
		ytrain = ytrain+[ytr]
		cv2.imshow('img1',small)
		key = cv2.waitKey(1)
		if(key==27): break

szout = len(ytrain[0])
xtrain = np.array(xtrain)
ytrain = np.array(ytrain)


print(xtrain)

xtrain = np.expand_dims(xtrain, axis=3)

model =  Sequential()

model.add(Conv2D(4, (3, 3), padding='same', input_shape=(sizeh, sizew, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(5, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(5, (4, 4), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(szout, activation='softmax'))



plot_model(model, to_file='./out/model1.png', show_shapes=True)

model_json = model.to_json()
json_file = open("./out/mclasshand.json", "w")
json_file.write(model_json)
json_file.close()

meth = ["Adam","Adam","Adam","sgd","rmsprop","AdaDelta"]
bs = [100,300,500]

for i in range(8):
	num1 = int(random()*len(meth))
	eps = int(50+random()*30)
	numb = int(random()*len(bs))
	print(meth[num1])
	if(i>0):
		model.load_weights("mclasshand1.h5")
	model.compile(loss='categorical_crossentropy',
			optimizer=meth[num1],
			metrics=['accuracy'])

	model.fit(xtrain, ytrain, batch_size=bs[numb], epochs=eps)
	model.save_weights("mclasshand1.h5")
result = model.predict(xtrain)


#val = result - ytrain
#print(val)



for i in range(1590):
	print(i)
	img2 = cv2.imread("./ris/exampl/b" + str(i)+'.jpg')
	for j in range(5):
		x = int(result[i][j*2]*640)
		y = int(result[i][j*2+1]*480)
		if(result[i][0]>0.5):
			cv2.putText(img2,"hand", (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,0,0),1)
		else:
			cv2.putText(img2,"not hand", (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,0,0),1)
	cv2.imshow('img2',img2)
	key = cv2.waitKey(5)


