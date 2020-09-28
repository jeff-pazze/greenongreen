import cv2
import numpy as np
from numpy import *

#totalarea = 1914.6

class greenDetection():
    def __init__(self):
	    self.mask = 0
	    self.binary = 0
	    self.binary_no_noise = 0
	    self.number_of_pixels = 0
		
    def sigmoid(self, x):
	    sig = (1/(1+np.exp(-x)))
	    return sig
	
    def isolateGreen(self, img):
	    img = np.array(img).astype(np.float32)
	    b = img[:,:,0]/255
	    g = img[:,:,1]/255
	    r = img[:,:,2]/255
	    weights = array([-1.97584457e+00,-10.04155935e+03, 8.39547821e+03,-8.01239533e+03])
	    y = (dot(array([1, r, g, b]), weights))
	    self.mask = self.sigmoid(y)

    def convertToBinary(self, green_th):
	    img = self.mask
	    img[img <= green_th] = 0
	    img[img > green_th] = 1
	    self.binary = img

    def removeNoise(self):
	    kernel = np.ones((3,3),np.uint8)
	    self.binary_no_noise = cv2.erode(self.binary, kernel)
	
    def numberOfPixels(self):
	    self.number_of_pixels = float(cv2.countNonZero(self.binary_no_noise))/(100.0)

    def isPlantRecognized(self, th):
	    if self.number_of_pixels > th:
		    return True
	    else:
		    return False
	
    def detect(self, img, green_th, pixel_th):
	    self.isolateGreen(img)
	    self.convertToBinary(green_th)
	    self.removeNoise()
	    self.numberOfPixels()
	    return self.isPlantRecognized(pixel_th)
    
    def getBinaryImg(self):
	    return self.binary
    
    def getBinaryImgNoNoise(self):
	    return self.binary_no_noise
    
    def getNumberOfGreenPixels(self):
	    return self.number_of_pixels

class LeafDetection():
    def __init__(self):
	    self.number_of_pixels = 0
	    
    def selectiveDetector(self, image, totalarea, auxarea):
	    
	    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	    blur_gray = cv2.GaussianBlur(gray,(3, 3),3)
	    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	    _, contours, hierarchy = cv2.findContours(gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

	    total   = 49152
	    refarea = 102

##	    auxarea = 102 cm^2
##	    percentarea = objectarea
	    
	    for i in contours :
		    cnt = cv2.contourArea(i)
		    (x, y, w, h) = cv2.boundingRect(i)
		    M = cv2.moments(thresh)

		    percentarea = (1-(total-cnt)/total)
		    objectarea = (((percentarea/2)*102)/auxarea)*100
		
		    print("auxarea", auxarea)
		    print('\npercentarea {:0.1f} %'.format(percentarea*100))
		    print('objectarea {:0.1f} cm^2'.format(objectarea))

		    if objectarea <10:
			    cv2.putText(image, "area: {:.3f} cm^2".format(objectarea), (x-5,y-5),cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
			    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

		    elif objectarea >10 and objectarea <20:
			    cv2.putText(image, "area: {:.3f} cm^2".format(objectarea), (x-5,y-5),cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
			    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

		    elif objectarea >20 and objectarea <50:
			    cv2.putText(image, "area: {:.3f} cm^2".format(objectarea), (x-5,y-5),cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
			    cv2.rectangle(image, (x, y), (x + w, y + h), (0,255,0), 2)

		    elif objectarea >50:
			    cv2.putText(image, "area: {:.3f} cm^2".format(objectarea), (x-5,y-5),cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
			    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
			    return True
  
