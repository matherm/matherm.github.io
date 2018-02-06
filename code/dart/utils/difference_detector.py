import cv2
import numpy as np
from utils.gui import GUI 

#print(cv2.__version__)



class Difference_Detector:

    ENV = {
        'BLUR' : (5,5),
        'BINARY_THRESHOLD_MIN' : 75,
        'BINARY_THRESHOLD_MAX' : 255,
        'CLAHE_CLIP_LIMIT' : 5,
        'CLAHE_TILE_SIZE' : (10,10),

        'ARROW_BLUR' : (5,5),
        'ARROW_BINARY_THRESHOLD_MIN' : 50,
        'ARROW_BINARY_THRESHOLD_MAX' : 255,
        'ARROW_CLAHE_CLIP_LIMIT' : 20,
        'ARROW_CLAHE_TILE_SIZE' : (10,10)
    }
    #def __init__(self):

    def computeDifference(self,grey1,grey2):
        # blur
        blur = Difference_Detector.ENV['BLUR']
        grey2 = cv2.blur(grey2,blur)
        grey1 = cv2.blur(grey1,blur)
        #normalize
        grey1 = cv2.equalizeHist(grey1)
        grey2 = cv2.equalizeHist(grey2)
        clahe = cv2.createCLAHE(Difference_Detector.ENV['CLAHE_CLIP_LIMIT'], Difference_Detector.ENV['CLAHE_TILE_SIZE'])
        #clahe
        grey1 = clahe.apply(grey1)
        grey2 = clahe.apply(grey2)
        #diff
        diff = cv2.subtract(grey2,grey1) + cv2.subtract(grey1,grey2)
        ret2,dif_thred = cv2.threshold(diff,Difference_Detector.ENV['BINARY_THRESHOLD_MIN'],Difference_Detector.ENV['BINARY_THRESHOLD_MAX'],cv2.THRESH_BINARY)
        return dif_thred,grey1,grey2,diff


    def computeDifferenceHighRes(self,grey1,grey2):
        # blur
        blur = Difference_Detector.ENV['BLUR']
        grey2 = cv2.blur(grey2,blur)
        grey1 = cv2.blur(grey1,blur)
        #normalize
        grey1 = cv2.equalizeHist(grey1)
        grey2 = cv2.equalizeHist(grey2)
        clahe = cv2.createCLAHE(Difference_Detector.ENV['ARROW_CLAHE_CLIP_LIMIT'], Difference_Detector.ENV['ARROW_CLAHE_TILE_SIZE'])
        #clahe
        grey1 = clahe.apply(grey1)
        grey2 = clahe.apply(grey2)
        #diff
        diff = cv2.subtract(grey2,grey1) + cv2.subtract(grey1,grey2)
        ret2,dif_thred = cv2.threshold(diff,Difference_Detector.ENV['ARROW_BINARY_THRESHOLD_MIN'],Difference_Detector.ENV['ARROW_BINARY_THRESHOLD_MAX'],cv2.THRESH_BINARY)
        return dif_thred