import cv2
import numpy as np

class Image_Tools:

    @staticmethod
    def getMaxContourIdx(contours):
        max_contour_length = 0
        max_contour = None
        for j in range(len(contours)):
            length = cv2.arcLength(contours[j],True)
            if length > max_contour_length:
                max_contour_length = length
                max_contour = j
        return max_contour

    @staticmethod
    def readFrame(time_in_millis, size):
        cap.set(cv2.CAP_PROP_POS_MSEC, time_in_millis)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, size) 
            return frame

    @staticmethod
    def readAndSafeFrames(path):
        cap = cv2.VideoCapture(path)
        frames = [] 
        frames.append(readFrame(100)) 
        frames.append(readFrame(2000)) 
        frames.append(readFrame(3000)) 
        frames.append(readFrame(5000)) 
        cap.release() 
        for i in range(len(frames)): 
            cv2.imwrite("Vid" + str(i) + ".png", frames[i])    
    

    @staticmethod
    def normalizeHist(arr):
        minval = arr[:,:].min()
        maxval = arr[:,:].max()
        print(minval)
        print(maxval)
        if minval != maxval:
            arr -= minval
            arr *= int((255.0 / (maxval - minval)))
        return arr

    @staticmethod
    def rotateImage(image, angle):
        image_center = tuple(np.array(image.shape)/2)
        rot_mat = cv2.getRotationMatrix2D(image_center,angle,1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape,flags=cv2.INTER_LINEAR)
        return result

    @staticmethod
    def sift(GrayIM):
        sift = cv2.xfeatures2d.SIFT_create()
        kp = sift.detect(GrayIM,None)
        out = np.array(GrayIM.shape)
        out = cv2.drawKeypoints(GrayIM,kp,out)
        return out, kp

    @staticmethod
    def debugRectangle(IM,x,y,w,h):
        IM_copy = IM.copy()
        cv2.rectangle(IM_copy,(x,y),(x+w,y+h),(255,255,255),1)
        return IM_copy

    @staticmethod
    def debugContours(IM,contours):
        SAMPLE = np.zeros(IM.shape,"uint8")
        cv2.drawContours(SAMPLE, contours, -1, (255,255,255), 10)
        return SAMPLE

    @staticmethod
    def getROI(IM,x,y,w,h):
        if(IM.ndim == 2):
            IM_ROI = IM[y:y+h,x:x+w]
        else:
            IM_ROI = IM[y:y+h,x:x+w,:]
        return IM_ROI


    @staticmethod
    def readImage(path, dimension=None):
        IM = cv2.imread(path)
        if dimension is not None:
            IM = cv2.resize(IM, dimension)
        if IM.ndim == 3: 
            base_frame_gray = cv2.cvtColor(IM, cv2.COLOR_BGR2GRAY)
        #print(IM.shape)
        #print(IM.dtype)
        #show(IM)
        return IM, base_frame_gray

    @staticmethod
    def prepareImage(IM, dimension):
        IM = cv2.resize(IM, dimension) 
        base_frame_gray = cv2.cvtColor(IM, cv2.COLOR_BGR2GRAY)
        #print(IM.shape)
        #print(IM.dtype)
        #show(IM)
        return IM, base_frame_gray

    @staticmethod
    def getIntersection(src_points):
        if src_points.shape != (4,2):
            return None, None
        #interesect lines
        x1 = src_points[0,0] 
        y1 = src_points[0,1]
        x2 = src_points[3,0]
        y2 = src_points[3,1]
        x3 = src_points[1,0] 
        y3 = src_points[1,1]
        x4 = src_points[2,0]
        y4 = src_points[2,1]
        py = ((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4)) / ((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))
        px =  ((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4)) / ((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))
        #show(IM)
        #swap output
        return int(px),int(py)

    @staticmethod    
    def debugIntersection(IM,src_points):
        IM = IM.copy()
        x1 = src_points[0,0] 
        y1 = src_points[0,1]
        x2 = src_points[3,0]
        y2 = src_points[3,1]
        x3 = src_points[1,0] 
        y3 = src_points[1,1]
        x4 = src_points[2,0]
        y4 = src_points[2,1]
        cv2.line(IM,(x1,y1),(x2,y2),255,5)
        cv2.line(IM,(x3,y3),(x4,y4),255,5)
        return IM
