import cv2
import numpy as np
from utils.gui import GUI 
from utils.image_tools import Image_Tools



class Dartboard:
    ENV = {
        'BULL_CENTER_DETECTION_SIZE' : 50,

        'DART_BOARD_TARGET_DIMENSION' : (1000,1000),
        'DART_BOARD_TARGET_ROTATION' : 45, 
        'DART_BOARD_TARGET_OFFSET' : 500, #in mm-1 500=5cm

        'TEXT_SIZE' : 3,
        'TEXT_THICKNESS' : 10
        }

    
    def _correctCenterOfBull(self,IM_ROI_grey,IM_ROI_red,px,py,src_points):
        src_points = src_points.copy()
        offset = Dartboard.ENV["BULL_CENTER_DETECTION_SIZE"]
        ROI_center = IM_ROI_grey[px-offset:px+offset,py-offset:py+offset] 
        IM_ROI_red_center = IM_ROI_red[px-offset:px+offset,py-offset:py+offset]
        
        ROI_bull = ROI_center * IM_ROI_red_center
        
        im2, contours, hierarchy = cv2.findContours(ROI_bull.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return None , None ,None 

        idx = Image_Tools.getMaxContourIdx(contours)
        bull_contour = contours[idx]
        
        M = cv2.moments(bull_contour)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        
        #scale back
        cx = (px - offset) + cx
        cy = (py - offset) + cy
        correction_offset_x = cx - px 
        correction_offset_y = cy - py 
        src_points[0,0] += correction_offset_x
        src_points[3,0] += correction_offset_x
        src_points[1,1] += correction_offset_y
        src_points[2,1] += correction_offset_y
        return src_points, ROI_bull, ROI_center

    def computePerspectiveTransformation(self,contours_structure,BASE_IM_GRAY,BASE_IM_red):
        target_dim = Dartboard.ENV['DART_BOARD_TARGET_DIMENSION']
        pts1 = np.zeros((4,2),'int')
        for i in range(len(contours_structure)):
            cnt = contours_structure[i]
            M = cv2.moments(cnt)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            #print(str(i) + " x: " + str(cx) + " y: " +str(cy))
            pts1[i,0] = cx
            pts1[i,1] = cy
        #print(pts1)
        sorted_points_x = pts1[pts1[:,0].argsort()]
        #print(sorted_points_x)
        left_column = sorted_points_x[0:2,:]
        right_column = sorted_points_x[2:,:]
        sorted_points_y_left = left_column[left_column[:,1].argsort()]
        sorted_points_y_right = right_column[right_column[:,1].argsort()]

        pts_src = np.asarray(np.concatenate((sorted_points_y_left, sorted_points_y_right), axis=0),"float32")
        #print(pts_src)
        pts_dest = np.float32([[0,0],[0,target_dim[1]],[target_dim[1],0],[target_dim[1],target_dim[1]]])
        M = cv2.getPerspectiveTransform(pts_src,pts_dest)

        px,py = Image_Tools.getIntersection(pts_src)
        pts_src_corrected, ROI_bull, ROI_center = self._correctCenterOfBull(BASE_IM_GRAY,BASE_IM_red,px,py,pts_src)
        return M, pts_src_corrected


    def computePerspectiveTransformationPts(self,PTS,BASE_IM_GRAY,BASE_IM_red):
        #PTS: top,bottom,left,right
        target_dim = Dartboard.ENV['DART_BOARD_TARGET_DIMENSION']
        pts_src = np.asarray(([PTS[0][0],PTS[0][1]],[PTS[3][0],PTS[3][1]],[PTS[2][0],PTS[2][1]],[PTS[1][0],PTS[1][1]]),"float32")
 
        diag = np.sqrt(((target_dim[0]/2) * (target_dim[0]/2)) + ((target_dim[1]/2) * (target_dim[1]/2)))
        r = target_dim[0]/2
        offset_diag = diag - r
        offset = np.sqrt((offset_diag*offset_diag)/2)

        #top, right,left,bottom
        pts_dest = np.float32([[0+offset,0+offset],[0+offset,target_dim[1]-offset],[target_dim[1]-offset,0+offset],[target_dim[1]-offset,target_dim[1]]-offset])
        M = cv2.getPerspectiveTransform(pts_src,pts_dest)

        px,py = Image_Tools.getIntersection(pts_src)
        pts_src_corrected, ROI_bull, ROI_center = self._correctCenterOfBull(BASE_IM_GRAY,BASE_IM_red,px,py,pts_src)
        return M, pts_src_corrected



    def computePerspectiveTransformationCorrection(self,pts_src):
        target_dim = Dartboard.ENV['DART_BOARD_TARGET_DIMENSION']
        pts_dest = np.float32([[0,0],[0,target_dim[1]],[target_dim[1],0],[target_dim[1],target_dim[1]]])
        M = cv2.getPerspectiveTransform(pts_src,pts_dest)
        return M


    def warpWithRotation(self,IM_ROI_grey,M):
        target_dim = Dartboard.ENV['DART_BOARD_TARGET_DIMENSION']
        IM_ROI_NORMAL = cv2.warpPerspective(IM_ROI_grey,M,(target_dim[1],target_dim[1]))
        IM_ROI_ROTATED = Image_Tools.rotateImage(IM_ROI_NORMAL,-1*Dartboard.ENV['DART_BOARD_TARGET_ROTATION'])
        return IM_ROI_ROTATED,IM_ROI_NORMAL


    def drawDartboard(self):
        IM = np.zeros(Dartboard.ENV['DART_BOARD_TARGET_DIMENSION'],"uint8")
        offset = Dartboard.ENV['DART_BOARD_TARGET_OFFSET']
        center = (IM.shape[0] // 2,IM.shape[1] // 2)
        size_dartboard = 3400 + offset
        scale = IM.shape[0] / size_dartboard
        rad_board = int(3400 / 2 * scale)
        rad_bull = int(127 / 2 * scale)
        rad_ring = int(318 / 2 * scale)
        rad_double = int((3400 - 160) / 2 * scale)
        rad_triple = int((2140 - 160) / 2 * scale)
        width_rings = int(80 * scale)
        line_thickness = int(12 * scale)
        angle = 360 // 20 
        angle_offset = 9

        #rings
        cv2.circle(IM, center, rad_bull, (255,255,255),line_thickness)
        cv2.circle(IM, center, rad_ring, (255,255,255),line_thickness)
        cv2.circle(IM, center, rad_double, (255,255,255),line_thickness)
        cv2.circle(IM, center, rad_double + width_rings, (255,255,255),line_thickness)
        cv2.circle(IM, center, rad_triple, (255,255,255),line_thickness)
        cv2.circle(IM, center, rad_triple + width_rings, (255,255,255),line_thickness)
        
        #lines  
        line_shape = np.zeros(IM.shape,"uint8")
        line_shape[:,(line_shape.shape[1]//2)-line_thickness:(line_shape.shape[1]//2)+line_thickness] = 255
        IM_temp = np.zeros(IM.shape,"uint8")
        for i in range(0,360,angle):
            line_shape_rot = Image_Tools.rotateImage(line_shape,i + angle_offset)   
            IM_temp = IM_temp + line_shape_rot
        
        #restore bull
        IM_mask = np.zeros(IM.shape,"uint8")
        cv2.circle(IM_mask, center, rad_board, (255,255,255),-1)
        cv2.circle(IM_mask, center, rad_ring, (0,0,0),-1)
        IM = IM + (IM_temp * IM_mask)
        
        #create Mask
        IM_mask = np.zeros(IM.shape,"uint8")
        cv2.circle(IM_mask, center, rad_board, (255,255,255),-1)
        
        #make color
        IM_color = np.repeat(IM[:, :, np.newaxis], 3, axis=2)    
        return IM_color, IM_mask

    def drawFinalDartboard(self,IM_dartboard,IM_dot,IM_dartboard_rotated):
        return IM_dartboard[:,:,0] + IM_dot,IM_dartboard_rotated + IM_dartboard[:,:,0]

    def drawScore(self,IM,score):
        IM = IM.copy()
        cv2.putText(IM,str(score), (0+10,IM.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, Dartboard.ENV['TEXT_SIZE'], 255,Dartboard.ENV['TEXT_THICKNESS'])
        return IM

    def calcScore(self,arrow_angle,length):
        offset = Dartboard.ENV['DART_BOARD_TARGET_OFFSET']
        IM = np.zeros(Dartboard.ENV['DART_BOARD_TARGET_DIMENSION'],"uint8")
        center = (IM.shape[0] // 2,IM.shape[1] // 2)
        size_dartboard = 3400 + offset
        scale = IM.shape[0] / size_dartboard
        rad_board = int(3400 / 2 * scale)
        rad_bull = int(127 / 2 * scale)
        rad_ring = int(318 / 2 * scale)
        rad_double = int((3400 - 160) / 2 * scale)
        rad_triple = int((2140 - 160) / 2 * scale)
        width_rings = int(80 * scale)
        line_thickness = int(12 * scale)
        angle = 360 // 20 
        angle_offset = 9

        NUMBERS = [20,1,18,4,13,6,10,15,2,17,3,19,7,16,8,11,14,9,12,5]

        #is out
        if length > rad_board:
            return -1
        
        #is double
        if length > rad_double and lenght < rad_double + width_rings:
            NUMBERS[:] = [x*2 for x in NUMBERS]
            
        #is triple
        if length > rad_triple and lenght < rad_triple + width_rings:
            NUMBERS[:] = [x*3 for x in NUMBERS]
        
        #is bull
        if length > rad_bull and length < rad_ring:
            return 25
        
        #is bull
        if length < rad_bull:
            return 50 
        
        #calc numbers
        for i in range(len(NUMBERS)):
            if arrow_angle < (i * angle) + angle_offset:
                return NUMBERS[i]
        
        #was 20
        return NUMBERS[0]