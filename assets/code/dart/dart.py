import cv2
import time
import numpy as np
from darty.dartboard_detector import Dartboard_Detector
from darty.difference_detector import Difference_Detector
from darty.arrow_detector import Arrow_Detector
from darty.dartboard import Dartboard
from darty.image_tools import Image_Tools
from darty.gui import GUI

class Darty:

    ENV = {
        'VIDEO_INPUT_PATH': "./images/video.mov", 
        'FPS_LIMIT': 20,
        'FPS_SKIP': 10,
        'RESOLUTION_WIDTH' :  1080,
        'RESOLUTION_HEIGHT' :  1920,

        'DIFFERENCE_THRES' : 30000,
        'DIFFERENCE_THRES_RESET' : 1000000  
        }
  
    def __init__(self):
        self.dartboard_detector = Dartboard_Detector()
        self.difference_detector = Difference_Detector()
        self.dartboard = Dartboard()
        self.arrow_detector = Arrow_Detector()

        self.reset()
        self.capture()
    
    def reset(self):
        print("-- SYSTEM RESET--")
        self.BASE_IM = None
        self.BASE_IM_GRAY = None
        self.BASE_IM_green = None  
        self.BASE_IM_red = None 
        self.BASE_IM_board = None  
        self.ROI_x = None
        self.ROI_y = None
        self.ROI_w = None
        self.ROI_h = None
        self.M_corrected = None
        self.M = None
        self.PREVIOUS_DIFFERENCE = None

    def computeBaseFrame(self,IM,IM_GRAY):
        print("--Computing New Baseframe--")
        x,y,w,h,BOARD,GREEN,RED = self.dartboard_detector.detectDartboard(IM)
        self.BASE_IM =  self.dartboard_detector.scaleROI(Image_Tools.getROI(IM,x,y,w,h))
        self.BASE_IM_GRAY =  self.dartboard_detector.scaleROI(Image_Tools.getROI(IM_GRAY,x,y,w,h))  
        self.BASE_IM_green =  self.dartboard_detector.scaleROI(Image_Tools.getROI(GREEN,x,y,w,h))  
        self.BASE_IM_red =  self.dartboard_detector.scaleROI(Image_Tools.getROI(RED,x,y,w,h))  
        self.BASE_IM_board =  self.dartboard_detector.scaleROI(Image_Tools.getROI(BOARD,x,y,w,h))
        self.ROI_x = x
        self.ROI_y = y
        self.ROI_w = w
        self.ROI_h = h
        IM_ROI_thres_color,IM_ROI_thres_color_closed,contours_structure = self.dartboard_detector.getOrientation(self.BASE_IM,self.BASE_IM_board)
        #GUI.show(Image_Tools.debugContours(IM_ROI_thres_color,contours_structure),"ORIENTATION")
        self.M, src_points = self.dartboard.computePerspectiveTransformation(contours_structure,self.BASE_IM_GRAY,self.BASE_IM_red)

        if src_points is  None:
            self.reset()
            return

        px,py = Image_Tools.getIntersection(src_points)
        if py is None:
            self.reset()
            return
        self.M_corrected = self.dartboard.computePerspectiveTransformationCorrection(src_points)
        self.PREVIOUS_DIFFERENCE = None

    def capture(self):
        video = cv2.VideoCapture()
        if Darty.ENV['VIDEO_INPUT_PATH']:
            video = cv2.VideoCapture(Darty.ENV['VIDEO_INPUT_PATH'])

        # Fps stuff
        loop_delta = 1./Darty.ENV['FPS_LIMIT']
        current_time = target_time = time.clock()
        frame_counter = 0

        while (True):
            # Sleep management. Limit fps.
            target_time += loop_delta
            sleep_time = target_time - time.clock()
            if sleep_time > 0:
                time.sleep(sleep_time)
                print("--FPS sleep--")
            
            # Loop frequency evaluation, prints actual fps
            previous_time, current_time = current_time, time.clock()
            time_delta = current_time - previous_time
            current_fps = 1. / time_delta

            #
            # Read Frame
            #  
            ret, new_frame = video.read()    
            #video.set(cv2.CAP_PROP_POS_FRAMES, frame_counter)
            if new_frame is None:
                print("END OF VIDEO, BREAKING")
                break # no more frames to read
            
            #
            # Subsample over time
            #
            if frame_counter % Darty.ENV['FPS_SKIP'] == 0:
                print('frequency: ' + str(current_fps) + " fps_skip: " + str(Darty.ENV['FPS_SKIP']))      
                IM, IM_GRAY = Image_Tools.prepareImage(new_frame, (Darty.ENV['RESOLUTION_WIDTH'], Darty.ENV['RESOLUTION_HEIGHT']))
                
                #
                # Compute Base Image for the first time
                #
                #print("frame: " + str(frame_counter))
                if self.BASE_IM is None:
                    self.computeBaseFrame(IM,IM_GRAY)
                    continue;          
                
                #
                # Get ROI
                #
                IM_ROI = self.dartboard_detector.scaleROI(Image_Tools.getROI(IM,self.ROI_x,self.ROI_y,self.ROI_w,self.ROI_h))
                IM_ROI_grey = self.dartboard_detector.scaleROI(Image_Tools.getROI(IM_GRAY,self.ROI_x,self.ROI_y,self.ROI_w,self.ROI_h)) 
                GUI.show(IM, "FEED")    

                #
                # Difference detection
                #
                IM_ROI_difference, IM_ROI_GRAY_NORM,IM_ROI_GRAY2_NORM,IM_ROI_GRAY_NORM_DIFF = self.difference_detector.computeDifference(self.BASE_IM_GRAY,IM_ROI_grey)                
                difference_sum = np.sum(IM_ROI_difference)
                if difference_sum > Darty.ENV['DIFFERENCE_THRES_RESET']:
                    self.computeBaseFrame(IM,IM_GRAY)
                    continue; 


                #print("Difference: " + str(difference_sum) + "px")
                if difference_sum > Darty.ENV['DIFFERENCE_THRES']:
                    #
                    # Arrow Detection
                    #
                    IM_arrow_closed,arrow_x1,arrow_y1,xx,yy,ww,hh,line_image,apex_image = self.arrow_detector.detectArrow(IM_ROI_difference,IM_ROI_grey)

                    if arrow_x1 is not None:                
                        IM_arrow_roi1 = self.arrow_detector.debugApex(IM_ROI,arrow_x1,arrow_y1,(0,255,0))
                        IM_ROI_grey = self.arrow_detector.markApex(IM_ROI_grey,arrow_x1,arrow_y1)

                        #
                        # Draw Dartboard
                        #                   
                        IM_ROI_ROTATED,IM_ROI_NORMAL = self.dartboard.warpWithRotation(IM_ROI_grey,self.M_corrected)
                        cx,cy,angle,length,cross,IM_dot,IM_line = self.arrow_detector.getMetricOfArrow(IM_ROI_ROTATED)

                        if cx is not None:
                            score = self.dartboard.calcScore(angle,length)
                            IM_dartboard, IM_mask = self.dartboard.drawDartboard()
                            IM_dartboard_geo,IM_dartboard_error = self.dartboard.drawFinalDartboard(IM_dartboard,IM_dot,IM_ROI_ROTATED)
                            IM_score = self.dartboard.drawScore(IM_dartboard_error,score)
                            GUI.show(IM_score, "DARTBOARD")
                            print("--Arrow Detected: Score was: " + str(score))
                            self.computeBaseFrame(IM,IM_GRAY)

                else:
                    if self.PREVIOUS_DIFFERENCE is None:
                        self.PREVIOUS_DIFFERENCE = IM_ROI_difference
                    else:
                        IM_ROI_difference = cv2.add(self.PREVIOUS_DIFFERENCE,IM_ROI_difference)
                        self.PREVIOUS_DIFFERENCE = IM_ROI_difference 
                   

                     
                
            cv2.waitKey(1)
            frame_counter += 1
            
        cv2.destroyAllWindows()

if __name__ == "__main__":
    darty = Darty()