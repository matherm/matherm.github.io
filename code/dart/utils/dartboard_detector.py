import cv2
import numpy as np
from utils.gui import GUI 
from utils.image_tools import Image_Tools

print(cv2.__version__)



class Dartboard_Detector:

    ENV = {
        'DARTBOARD_SHAPE' : (1000,1000),

        'DETECTION_BLUR' : (5,5),
        'DETECTION_GREEN_LOW' : 90,
        'DETECTION_GREEN_HIGH' : 95,
        'DETECTION_RED_LOW' : 0,
        'DETECTION_RED_HIGH' : 20,
        'DETECTION_STRUCTURING_ELEMENT' : (100,100),
        'DETECTION_BINARY_THRESHOLD_MIN' : 127,
        'DETECTION_BINARY_THRESHOLD_MAX' : 255,
        'DETECTION_OFFSET' : 200,

        'ORIENTATION_BLUR' : (5,5),
        'ORIENTATION_COLOR_LOW' : 45,
        'ORIENTATION_COLOR_HIGH': 60,
        'ORIENTATION_KERNEL' : (100,100),
        'ORIENTATION_ELEMENT_SIZE_MIN' : 350,
        'ORIENTATION_ELEMENT_SIZE_MAX' : 600,

        'ORIENTATION_TEMPLATES' : ['shape_top.png','shape_bottom.png','shape_left.png','shape_right.png']

        }

    def scaleROI(self,IM):
        if(IM.ndim == 3):
            IM_normal = np.zeros((self.ENV['DARTBOARD_SHAPE'][0],self.ENV['DARTBOARD_SHAPE'][1],IM.shape[2]),"uint8")
        else:
            IM_normal = np.zeros((self.ENV['DARTBOARD_SHAPE'][0],self.ENV['DARTBOARD_SHAPE'][1]),"uint8")
        scale = 1
        if IM.shape[0] > IM.shape[1]:
            #higher than width
            scale = IM_normal.shape[0] / IM.shape[0]
        else:
            #widther than high
            scale = IM_normal.shape[1] / IM.shape[1]
        new_y =  int(IM.shape[0] * scale)
        new_x =  int(IM.shape[1] * scale)
        offset_y = int((IM_normal.shape[0] - new_y)/2) 
        offset_x = int((IM_normal.shape[1] - new_x)/2)
        IM_resized = cv2.resize(IM, (new_x,new_y),cv2.INTER_AREA)
        if(IM.ndim == 3):
            IM_normal[offset_y:offset_y+new_y,offset_x:offset_x+new_x,:] = IM_resized
        else:
            IM_normal[offset_y:offset_y+new_y,offset_x:offset_x+new_x] = IM_resized
        return IM_normal

    def detectDartboard(self,IM):
        IM_blur = cv2.blur(IM,Dartboard_Detector.ENV['DETECTION_BLUR'])
        #convert to HSV
        base_frame_hsv = cv2.cvtColor(IM_blur, cv2.COLOR_BGR2HSV)
        # Extract Green
        green_thres_low = int(Dartboard_Detector.ENV['DETECTION_GREEN_LOW'] /255. * 180)
        green_thres_high = int(Dartboard_Detector.ENV['DETECTION_GREEN_HIGH'] /255. * 180)
        green_min = np.array([green_thres_low, 100, 100],np.uint8)
        green_max = np.array([green_thres_high, 255, 255],np.uint8)
        frame_threshed_green = cv2.inRange(base_frame_hsv, green_min, green_max)
        #Extract Red
        red_thres_low = int(Dartboard_Detector.ENV['DETECTION_RED_LOW'] /255. * 180)
        red_thres_high = int(Dartboard_Detector.ENV['DETECTION_RED_HIGH'] /255. * 180)
        red_min = np.array([red_thres_low, 100, 100],np.uint8)
        red_max = np.array([red_thres_high, 255, 255],np.uint8)
        frame_threshed_red = cv2.inRange(base_frame_hsv, red_min, red_max)
        #Combine
        combined = frame_threshed_red + frame_threshed_green
        #Close
        kernel = np.ones(Dartboard_Detector.ENV['DETECTION_STRUCTURING_ELEMENT'],np.uint8)
        closing = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        #GUI.show(closing, "Dart_Detector")
        #find contours
        ret,thresh = cv2.threshold(combined,Dartboard_Detector.ENV['DETECTION_BINARY_THRESHOLD_MIN'],Dartboard_Detector.ENV['DETECTION_BINARY_THRESHOLD_MAX'],0)
        im2, contours, hierarchy = cv2.findContours(closing.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        max_cont = -1   
        max_idx = 0
        for i in range(len(contours)):
            length = cv2.arcLength(contours[i], True)
            if  length > max_cont:
                max_idx = i
                max_cont = length
        x,y,w,h = cv2.boundingRect(contours[max_idx])
        x = x-Dartboard_Detector.ENV['DETECTION_OFFSET']
        y = y-Dartboard_Detector.ENV['DETECTION_OFFSET']
        w = w+int(2*Dartboard_Detector.ENV['DETECTION_OFFSET'])
        h = h+int(2*Dartboard_Detector.ENV['DETECTION_OFFSET'])
        return x,y,w,h,closing,frame_threshed_green,frame_threshed_red


    def getOrientation(self,IM_ROI,IM_ROI_board):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,Dartboard_Detector.ENV['ORIENTATION_KERNEL'])
        #Segment zones
        IM_ROI_blur = cv2.blur(IM_ROI,Dartboard_Detector.ENV['ORIENTATION_BLUR'])
        #convert to HSV
        IM_ROI_HSV = cv2.cvtColor(IM_ROI_blur, cv2.COLOR_BGR2HSV)
        purple_thres_low = int(Dartboard_Detector.ENV['ORIENTATION_COLOR_LOW'] /255. * 180)
        purple_thres_high = int(Dartboard_Detector.ENV['ORIENTATION_COLOR_HIGH'] /255. * 180)
        purple_min = np.array([purple_thres_low, 100, 100],np.uint8)
        purple_max = np.array([purple_thres_high, 255, 255],np.uint8)
        frame_thres_color = cv2.inRange(IM_ROI_HSV, purple_min, purple_max)
        #Mask
        frame_thres_color = cv2.subtract(frame_thres_color,IM_ROI_board)
        frame_thres_color_closed = cv2.morphologyEx(frame_thres_color, cv2.MORPH_CLOSE, kernel)
        
        #Compute contours
        im2, contours, hierarchy = cv2.findContours(frame_thres_color_closed.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contour_lengths = []
        contours_structure = []
        for i in range(len(contours)):
            length = cv2.arcLength(contours[i],True)
            contour_lengths.append(length)
            if length > Dartboard_Detector.ENV['ORIENTATION_ELEMENT_SIZE_MIN'] and length < Dartboard_Detector.ENV['ORIENTATION_ELEMENT_SIZE_MAX']:
                contours_structure.append(contours[i])
        #debug histogramm
        #print(len(point_contours))
        #plt.hist(contour_lengths, bins=20, range=(50,1000), normed=False, weights=None, cumulative=False, bottom=None, histtype='bar', align='mid', orientation='vertical', rwidth=None, log=False, color=None, label=None, stacked=False, hold=None, data=None)
        #plt.show()
        return frame_thres_color,frame_thres_color_closed,contours_structure


    def getOrientationCorr(self,IM_ROI,base_dir):
        kernel_l = cv2.imread(base_dir + self.ENV['ORIENTATION_TEMPLATES'][2])
        kernel_r = cv2.imread(base_dir + self.ENV['ORIENTATION_TEMPLATES'][3])
        kernel_t = cv2.imread(base_dir + self.ENV['ORIENTATION_TEMPLATES'][0])
        kernel_b = cv2.imread(base_dir + self.ENV['ORIENTATION_TEMPLATES'][1])
        h = kernel_l.shape[0]
        w = kernel_l.shape[1]
       
        #right
        res = cv2.matchTemplate(IM_ROI,kernel_r,cv2.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        right_top_left = max_loc
        right = (right_top_left[0] + w, right_top_left[1] + h//2)
        #GUI.imShow(kernel_r)
        

        #left
        res = cv2.matchTemplate(IM_ROI,kernel_l,cv2.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        left_top_left = max_loc
        left = (left_top_left[0], left_top_left[1] + h//2)
        #GUI.imShow(kernel_l)
        
        h = kernel_t.shape[0]
        w = kernel_t.shape[1]
        #top
        res = cv2.matchTemplate(IM_ROI,kernel_t,cv2.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_top_left = max_loc
        top = (top_top_left[0] + w//2, top_top_left[1])
        #GUI.imShow(kernel_t)
        #GUI.imShow(res)
        #print(max_loc)

        #bottom
        res = cv2.matchTemplate(IM_ROI,kernel_b,cv2.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        bottom_top_left = max_loc
        bottom = (bottom_top_left[0] + w//2, bottom_top_left[1] + h)
        #GUI.imShow(kernel_b)
        
        return top_top_left,bottom_top_left,left_top_left,right_top_left,top,bottom,left,right

      

