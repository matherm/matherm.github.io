import cv2
import numpy as np
import time
from utils.gui import GUI 
from utils.image_tools import Image_Tools

#print(cv2.__version__)
class Arrow_Detector:

    ENV = {
        'DETECTION_KERNEL_SIZE' : (100,100),
        'DETECTION_RADIAL_STEP' : 10,
        'DETECTION_KERNEL_THICKNESS' : 1,
        'DETECTION_APEX_OFFSET' : 20, #20
        'DETECTION_APEX_LINE_THICKNESS' : 20, #20
        'DETECTION_APEX_LINE_THICKNESS_PEAK' : 10, #20

        'APEX_CLIPPING_OFFSET' : 50, 
        'APEX_MARK_SIZE' : 10
    }

    def detectArrowState(self,IM_arrow):
        lu = IM_arrow[0:IM_arrow.shape[0]//2,0:IM_arrow.shape[1]//2]
        ru = IM_arrow[0:IM_arrow.shape[0]//2,IM_arrow.shape[1]//2:IM_arrow.shape[1]]
        lb = IM_arrow[IM_arrow.shape[0]//2:IM_arrow.shape[0],0:IM_arrow.shape[1]//2]
        rb = IM_arrow[IM_arrow.shape[0]//2:IM_arrow.shape[0],IM_arrow.shape[1]//2:IM_arrow.shape[1]]
        verbs = [('l','u'),('r','u'),('l','b'),('r','b')]
        stack = [lu,ru,lb,rb]
        max = -1
        maxIdx = 0
        for i in range(len(stack)):
            if np.sum(stack[i]) > max:
                max = np.sum(stack[i])
                maxIdx = i
        #print(verbs[maxIdx])
        return verbs[maxIdx]

    def computeArrowOrientation(self,IM,arange,kernel):
                    max_contour_length = 0
                    max_angle = 0
                    max_contour = 0
                    max_img = 0
                    for i in arange:
                        kernel_rot = Image_Tools.rotateImage(kernel,i)
                        closed = cv2.morphologyEx(IM, cv2.MORPH_CLOSE, kernel_rot)
                        im2, contours, hierarchy = cv2.findContours(closed.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
                        for j in range(len(contours)):
                            length = cv2.arcLength(contours[j],True)
                            if length > max_contour_length:
                                max_contour_length = length
                                max_angle = i
                                max_contour = contours[j]
                                max_img = closed
                    return max_contour_length,max_angle,max_contour,max_img

    def _detectArrowLine(self,IM_closed,max_contour,xx,yy,ww,hh):
        # Improve with fitting line
        line_image = np.zeros(IM_closed.shape,"uint8")
        line_image_peak = np.zeros(IM_closed.shape,"uint8")
        
        # then apply fitline() function
        [vx,vy,x,y] = cv2.fitLine(max_contour,cv2.DIST_L2,0,0.01,0.01)

        # Now find two extreme points on the line to draw line
        righty = int((-x*vy/vx) + y)
        lefty = int(((line_image.shape[1]-x)*vy/vx)+y)

        #Finally draw the line
        cv2.line(line_image,(line_image.shape[1]-1,lefty),(0,righty),255,Arrow_Detector.ENV['DETECTION_APEX_LINE_THICKNESS'])
        cv2.line(line_image_peak,(line_image.shape[1]-1,lefty),(0,righty),255,Arrow_Detector.ENV['DETECTION_APEX_LINE_THICKNESS_PEAK'])
        
        #compute orientation
        (h,v) = self.detectArrowState(Image_Tools.getROI(IM_closed,xx,yy,ww,hh))
        if h == 'l':
            if v == 'u':
                arrow_x1 = xx+ww
                arrow_y1 = yy+hh
            else:
                arrow_x1 = xx+ww
                arrow_y1 = yy
        else:
            if v == 'u':
                arrow_x1 = xx
                arrow_y1 = yy+hh
            else:
                arrow_x1 = xx
                arrow_y1 = yy  
        return arrow_x1,arrow_y1,line_image_peak,h,v

    def _detectApex(self,IM_ROI2_grey,line_image_peak,arrow_x1,arrow_y1,h,v):
        # Isolate the apex
        offset = Arrow_Detector.ENV['DETECTION_APEX_OFFSET']
        IM_ROI_APEX = IM_ROI2_grey[arrow_y1-offset:arrow_y1+offset,arrow_x1-offset:arrow_x1+offset]
        IM_ROI_LINE = line_image_peak[arrow_y1-offset:arrow_y1+offset,arrow_x1-offset:arrow_x1+offset] 
        IM_ROI_APEX_edges = cv2.Canny(IM_ROI_APEX,50,100)
        IM_ROI_APEX_masekd = cv2.multiply(IM_ROI_LINE,IM_ROI_APEX_edges)
        
        #GUI.imShow(IM_ROI_APEX)
        #GUI.imShow(IM_ROI_APEX_edges)

        im2_line, contours_line, hierarchy_line = cv2.findContours(IM_ROI_APEX_masekd.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours_line) == 0:
            return None, None, None, None,None,None,None,None,None,None
        
        max_contour_idx = Image_Tools.getMaxContourIdx(contours_line)
        xxx,yyy,www,hhh = cv2.boundingRect(contours_line[max_contour_idx])

        #GUI.imShow(Image_Tools.debugRectangle(IM_ROI_APEX_masekd,xxx,yyy,www,hhh))
        #GUI.imShow(IM_ROI_APEX_masekd)

        IM_ROI_APEX_clipped = np.zeros(IM_ROI_APEX_masekd.shape, "uint8")
        IM_ROI_APEX_clipped[yyy:yyy+hhh,xxx:xxx+www] = IM_ROI_APEX_masekd[yyy:yyy+hhh,xxx:xxx+www] 

        IM_ROI_APEX_masekd = IM_ROI_APEX_clipped
        #GUI.imShow(IM_ROI_APEX_clipped)

        # respect orientation
        y,x = np.where(IM_ROI_APEX_masekd > 1)
        np.sort(y)
        #print(h)
        #print(v)
        if h == 'l':
            if v == 'u':
                arrow_y2 = y[y.shape[0]-1]
                x = np.where(IM_ROI_APEX_masekd[arrow_y2,:] > 1)[0]
                np.sort(x)
                arrow_x2 = x[x.shape[0]-1]
            else:
                arrow_y2 = y[0]
                x = np.where(IM_ROI_APEX_masekd[arrow_y2,:] > 1)[0]
                arrow_x2 = x[x.shape[0]-1]
        else:
            if v == 'u':
                arrow_y2 = y[y.shape[0]-1]
                x = np.where(IM_ROI_APEX_masekd[arrow_y2,:] > 1)[0]
                np.sort(x)
                arrow_x2 = x[0]
                #arrow_y2 = yyy
                #arrow_x2 = xxx
            else:
                arrow_y2 = y[0]
                x = np.where(IM_ROI_APEX_masekd[arrow_y2,:] > 1)[0]
                np.sort(x)
                arrow_x2 = x[0]   
        
        # transform to original space
        arrow_y1 = (arrow_y1 - offset) + arrow_y2
        arrow_x1 = (arrow_x1 - offset) + arrow_x2

        return arrow_x1,arrow_y1,IM_ROI_APEX



    def detectArrow(self,diff_image,IM_ROI2_grey):
        kernel_size = Arrow_Detector.ENV['DETECTION_KERNEL_SIZE']
        kernel = np.zeros(kernel_size,np.uint8)
        kernel_thickness = Arrow_Detector.ENV['DETECTION_KERNEL_THICKNESS']
        kernel[:,(kernel.shape[1]//2)-kernel_thickness:(kernel.shape[1]//2)+kernel_thickness] = 1
        max_contour_length,max_angle,max_contour,max_img = self.computeArrowOrientation(diff_image,range(0,180,Arrow_Detector.ENV['DETECTION_RADIAL_STEP']),kernel)
        
        if len(max_contour) == 0:
            return None, None, None, None,None,None,None,None,None,None
                
        xx,yy,ww,hh = cv2.boundingRect(max_contour)

        #Detect line of arrow
        arrow_x1,arrow_y1,line_image_peak,h,v = self._detectArrowLine(max_img,max_contour,xx,yy,ww,hh) 
        
        #Detect apex of arrow
        arrow_x1,arrow_y1,IM_apex = self._detectApex(IM_ROI2_grey,line_image_peak,arrow_x1,arrow_y1,h,v)

        return max_img,arrow_x1,arrow_y1,xx,yy,ww,hh,line_image_peak,IM_apex

    def debugApex(self,IM,arrow_x,arrow_y,color):
        IM = IM.copy()
        clipping_offset = Arrow_Detector.ENV['APEX_CLIPPING_OFFSET']
        cv2.rectangle(IM,(arrow_x,arrow_y),(arrow_x,arrow_y),color,2)
        IM_arrow_roi = IM[arrow_y-clipping_offset:arrow_y+clipping_offset,arrow_x-clipping_offset:arrow_x+clipping_offset]
        #show(IM_arrow_roi)
        return IM_arrow_roi

    def markApex(self,IM_ROI,arrow_x,arrow_y):
        IM_ROI = (IM_ROI.copy() - 2)
        cv2.rectangle(IM_ROI,(arrow_x,arrow_y),(arrow_x,arrow_y),(255,255,255),Arrow_Detector.ENV["APEX_MARK_SIZE"])
        return IM_ROI

    def getMetricOfArrow(self,IM_ROI_ROTATED):
        ret2,thred = cv2.threshold(IM_ROI_ROTATED,254,255,cv2.THRESH_BINARY)
        im2, contours, hierarchy = cv2.findContours(thred.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return None,None,None,None,None,None,None

        IM_dot = thred.copy()
        cnt = contours[0]
        MO = cv2.moments(cnt)
        cx = int(MO['m10']/MO['m00'])
        cy = int(MO['m01']/MO['m00'])
        cv2.line(thred,(thred.shape[0]//2,thred.shape[1]//2),(cx,cy),(255,255,255),2)
        IM_line = thred
        dx = cx - (thred.shape[0]//2)  
        dy = cy - (thred.shape[1]//2)
        length = np.sqrt((dx*dx) + (dy*dy))
        nx = 0
        ny = -1
        angle = np.arccos ( ((nx * dx) + (ny*dy)) / length)
        cross = (dx * ny) - (nx * dy)
        if cross > 0:
            angle = (np.pi * 2) - angle
        
        angle = np.rad2deg(angle)
        return cx,cy,angle,length,cross,IM_dot,IM_line
