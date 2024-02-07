#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 22:05:40 2018

@author: say
"""

import cv2
import numpy as np
mas = [(255,0,0),(0,255,0),(0,0,255),(255,0,255),(255,255,0)]

class click:
	index = 0
	img =[]
	nimg = 1000
	def click_and_crop(event, x, y, flags, param):
		if event == cv2.EVENT_LBUTTONDOWN:
			print("event",x,y)
			if(click.index==0):
				click.img = param.copy()
			cv2.circle(param, (x,y), 3, mas[click.index], -1)
			click.index = click.index+1
			cv2.imshow("image", image)
			if(click.index>4):
				click.index = 0
				cv2.imwrite("./hands/exampl/a" + str(click.nimg)+'.jpg',image)
				cv2.imwrite("./hands/exampl/b" + str(click.nimg)+'.jpg',click.img)
				click.nimg =click.nimg+1



cap = cv2.VideoCapture("./hands/output.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')


index = 0
# keep looping until the 'esc' key is pressed
# или пока не нажат enter для ввода мышкой коодинаты пальца начиная с большого
# чтобы ввести следующую картинку нажать Enter
while True:
	ret,image = cap.read()
	cv2.imshow("image", image)
	cv2.setMouseCallback("image", click.click_and_crop, image)

	key = cv2.waitKey(40)
	print(key)
	if key == 27: 
		break

	if key == 13:
		while True:
			k = cv2.waitKey(40)
			if k == 13:
				break
			
 
cap.release()
cv2.destroyAllWindows()
