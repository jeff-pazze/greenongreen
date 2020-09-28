from numpy import *
from greenDetection import LeafDetection, greenDetection
from imgRoi import imgRoi
from math import exp
from imgRoi import imgRoi
from skimage.feature import hessian_matrix, hessian_matrix_eigvals

import random
import cv2
import numpy as np
import os
import glob
import time
import csv
import datetime as dt
import matplotlib.pyplot as plt
import sys

video = False
systemCalibration = True
leafdetection = True

file = {}
param = {}

param['verde'] = 0.38
param['tamanho_planta'] = 0.02

file['x'] = 80
file['y'] = 40
file['z'] = 60

totalarea = 0
auxarea = 0

imageRoi = imgRoi()

if leafdetection == True:
        
        greenNozzle1 = LeafDetection()
        greenNozzle2 = LeafDetection()
else:
        greenNozzle1 = greenDetection()
        greenNozzle2 = greenDetection()
        

def readCalibration():

        global totalarea
        separador = ','
        with open('calibration.csv', 'r') as csv_file:
                for line_number, content in enumerate(csv_file):
                        if line_number:
                                colunas = content.strip().split(separador)
                                totalarea = float(colunas[1])
        print("Calibration:", totalarea)
        csv_file.close()
                                
	
pathcalibration='C:/Users/jeff_/Documents/EIRENE/CALIBRACAO_IMAGEM/Referencias/120.jpg'

def areaCalibration():

	global totalarea
	

	print("Initializing area calibration...")
	print("press any button to capture the image and calculate its area")
	input()
	
	imagecalib = cv2.imread(pathcalibration)

	print("successful capture")
	print("imagecalib.shape", imagecalib.shape)

	imageRoi.calc(imagecalib, 80, 40, 60)
	icalib = imageRoi.getCalibImage()
	cv2.imshow('calib', icalib)

	print("imagecalib.shape after", icalib.shape)
	
	hsvcal = cv2.cvtColor(icalib, cv2.COLOR_BGR2HSV)
	lowercal = np.array([80, 65, 60], dtype="uint8")
	uppercal = np.array([125, 240,150], dtype="uint8")
	maskcal = cv2.inRange(hsvcal, lowercal, uppercal)

	element = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
	maskcal = cv2.erode(maskcal, element, iterations = 1)
	maskcal = cv2.dilate(maskcal, element, iterations = 2)
	maskcal = cv2.erode(maskcal, element)

	ret, thresh = cv2.threshold(maskcal,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	_, contours, hierarchy = cv2.findContours(maskcal,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

	total   = 98304
	refarea = 102
	a = 45805
	b = 29

	for i in contours :
		cnt = cv2.contourArea(i)
		(x, y, w, h) = cv2.boundingRect(i)
		M = cv2.moments(thresh)

		percentarea = (1-(total-cnt)/total)
		calc = percentarea*refarea
		totalarea = refarea/percentarea
		objectarea = totalarea*percentarea
		heigthobject = -(1000*np.log(cnt/a))/b

		global auxarea

		auxarea = calc
		
		print('\nCalibration percentarea {:0.1f} %'.format(percentarea*100))
		print('Calibration cnt {:0.1f} pixels'.format(cnt))
		print('Calibration calc {:0.1f} % total'.format(calc))
		print('Calibration totalarea {:0.1f} pixels correction'.format(totalarea))
		print('Calibration objectarea {:0.1f} cm^2'.format(objectarea))
		print('Calibration heigthobject {:0.5f} cm^2\n'.format(heigthobject))

		cv2.putText(icalib, "area: {:.3f}cm^2".format(objectarea), (x-5,y-5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
		cv2.putText(icalib, "Area Correction: {:.3f}%".format(calc), (x-25,y-25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
		cv2.rectangle(imagecalib, (x, y), (x + w, y + h), (0,255,255), 1)

		cv2.putText(maskcal, "area: {:.3f}cm^2".format(objectarea), (x-5,y-5),cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
		cv2.putText(maskcal, "Area Correction: {:.3f}%".format(calc), (x-25,y-25),cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
		cv2.rectangle(maskcal, (x, y), (x + w, y + h), (255,255,255), 1)

		start_pointL = (66, 188) 
##		end_pointL   = (322, 380)
##		start_pointR = (322, 188) 
		end_pointL   = (578, 380)
			
		cv2.rectangle(imagecalib, start_pointL, end_pointL, (0,255,255), 1)
##		cv2.rectangle(imagecalib, start_pointR, end_pointR, (0,255,255), 1)

		cv2.imshow('imagecalib', icalib)
		cv2.imshow('maskcal', maskcal)

		with open("calibration.csv" , 'a', newline = '\n') as csv_file:
			time = dt.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
			escrever = csv.writer(csv_file)
			escrever.writerow([time, float(totalarea), int(totalarea)])
		csv_file.close()
						
if __name__ == "__main__":

	path='C:/Users/jeff_/Documents/EIRENE/CALIBRACAO_IMAGEM/Referencias'
##	path='C:/Users/jeff_/Documents/EIRENE/CALIBRACAO_IMAGEM'

	piclist = list()

	WIDTH = 640
	HEIGHT = 48 
	i = 0
	font = cv2.FONT_HERSHEY_SIMPLEX

	if systemCalibration == True:
		#areaCalibration(imagecalib)
		areaCalibration()
		print("calib")
	else:
		readCalibration()

	if video == False:

		for infile in glob.glob(os.path.join(path,'*.jpg')):

			image = cv2.imread(infile)

			#areaCalibration(image)
			
			hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
			lower = np.array([100, 65, 60], dtype="uint8")
			upper = np.array([125, 240,150], dtype="uint8")
##			lower = np.array([36, 25, 25], dtype="uint8")
##			upper = np.array([90, 255,255], dtype="uint8")

			mask = cv2.inRange(hsv, lower, upper)
			
			element = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
			mask = cv2.erode(mask, element, iterations = 1)
			mask = cv2.dilate(mask, element, iterations = 2)
			mask = cv2.erode(mask, element)

			Afdilate = cv2.bitwise_and(image, image, mask = mask)
	
			imageRoi.calc(Afdilate, file['x'], file['y'], file['z'])
			image_left = imageRoi.getLeftImage()
			image_right = imageRoi.getRightImage()

			e1 = cv2.getTickCount()
			
			if leafdetection == True:
				nozzle1_detection =  greenNozzle1.selectiveDetector(image_left, totalarea, auxarea)
				nozzle2_detection =  greenNozzle2.selectiveDetector(image_right, totalarea, auxarea)
				
			else:
				nozzle1_detection =  greenNozzle1.detect(image_left, param['verde'], param['tamanho_planta'])
				nozzle2_detection =  greenNozzle2.detect(image_right, param['verde'], param['tamanho_planta'])		
			
			cv2.imshow('image_left', image_left)
			cv2.imshow('image_right', image_right)

			print("nozzle1_detection" , nozzle1_detection, "nozzle2_detection" , nozzle2_detection)
		
			e2 = cv2.getTickCount() 
			t = (e2 - e1)/cv2.getTickFrequency()
			print("time" , t)

			start_pointL = (66, 188) 
			end_pointL   = (322, 380)
			start_pointR = (322, 188) 
			end_pointR   = (578, 380)
			
			cv2.rectangle(Afdilate, start_pointL, end_pointL, (0,255,255), 1)
			cv2.rectangle(Afdilate, start_pointR, end_pointR, (0,255,255), 1)

			cv2.rectangle(image, start_pointL, end_pointL, (0,255,255), 1)
			cv2.rectangle(image, start_pointR, end_pointR, (0,255,255), 1)

##			cv2.imshow('Image', image)
##			cv2.imshow('Afdilate', Afdilate)

			print('imagem:', infile)	
			input()
							
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

	path='C:/Users/jeff_/Documents/EIRENE/BUVA/ImagensEditadas'

	
	if video == True:
		for infile in glob.glob(os.path.join(path,'*.mp4')):
			cap = cv2.VideoCapture(infile)
			print('video:', infile)
			
			if (cap.isOpened()== False):
				print("Error opening video stream or file")

			while(cap.isOpened()):
				ret, frame = cap.read()
				if ret == True:
					
					frame = cv2.resize(frame, (640,480))
					hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
					lower = np.array([36, 25, 25], dtype="uint8")
					upper = np.array([90, 255,255], dtype="uint8")
					mask = cv2.inRange(hsv, lower, upper)
		
					kernal = np.ones((1,1),"uint8")
					red = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernal)
					dilate = cv2.dilate(red,kernal,iterations=1)

					Afdilate = cv2.bitwise_and(frame, frame, mask = dilate)

					imageRoi.calc(Afdilate, file['x'], file['y'], file['z'])
					image_left = imageRoi.getLeftImage()
					image_right = imageRoi.getRightImage()

					if leafdetection == True:
						nozzle1_detection =  greenNozzle1.selectiveDetector(image_left)
						nozzle2_detection =  greenNozzle2.selectiveDetector(image_right)
					else:
						nozzle1_detection =  greenNozzle1.detect(image_left, param['verde'], param['tamanho_planta'])
						nozzle2_detection =  greenNozzle2.detect(image_right, param['verde'], param['tamanho_planta'])		
			
					cv2.imshow('image_left', image_left)
					cv2.imshow('image_right', image_right)

					start_pointL = (66, 188) 
					end_pointL   = (322, 380)
					start_pointR = (322, 188) 
					end_pointR   = (578, 380)
					
					cv2.rectangle(Afdilate, start_pointL, end_pointL, (0,255,255), 1)
					cv2.rectangle(Afdilate, start_pointR, end_pointR, (0,255,255), 1)

					cv2.rectangle(frame, start_pointL, end_pointL, (0,255,255), 1)
					cv2.rectangle(frame, start_pointR, end_pointR, (0,255,255), 1)

					cv2.imshow('Afdilate', Afdilate)
					cv2.imshow('Frame',frame)

					
					if cv2.waitKey(25) & 0xFF == ord('q'):
						break
				else:
					break

			cap.release()
			cv2.destroyAllWindows()
