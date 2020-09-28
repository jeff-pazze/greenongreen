import cv2


import numpy as np
import os
import glob


class imgRoi():
	def __init__(self):
		self.image_roi = 0
		self.image_draw = 0
		self.image_left = 0
		self.image_right = 0
		self.image_calib = 0

	def calc(self, img, x, y, z):
                
		l = img.shape[1]
		a = img.shape[0]

		x = x/100.0
		y = y/100.0
		z = z/100.0

		if x > 1:
			x = 1
		if y > 1:
			y = 1
		if z > 1-y/2.0:
			z = 1-y/2.0

		if x < 0:
			x = 0.01
		if y < 0:
			y = 0.01
		if z < y/2.0:
			z = 0+y/2.0

		a0 = int(z*a-(y*a/2))
		a1 = int(z*a+(y*a/2))
		l0 = int((l/2)-(x*l/2))
		l1 = int((l/2)+(x*l/2))

		self.image_roi = img.copy()[a0:a1, l0:l1]
	
	def getLeftImage(self):
		w = self.image_roi.shape[1]
		h = self.image_roi.shape[0]        
		self.image_left = self.image_roi[0:h, 0:(int(w/2))]
		return self.image_left
		
	def getRightImage(self):
		w = self.image_roi.shape[1]
		h = self.image_roi.shape[0] 
		self.image_right = self.image_roi[0:h, (int(w/2)):w]
		return self.image_right

	def getCalibImage(self):
		w = self.image_roi.shape[1]
		h = self.image_roi.shape[0] 
		self.image_calib = self.image_roi[0:h, 0:(int(w))]
		return self.image_calib

	def getRoi(self):    
		return self.image_roi

	def getDraw(self):
		return self.image_draw
