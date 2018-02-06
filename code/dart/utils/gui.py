import cv2
import numpy as np
import matplotlib.pyplot as plt


class GUI:

    ENV = {
        'GUI_RESOLUTION_SCALE' : 0.5,
        'SHOW_GUI' : ["FEED","DIFFERENCE", "ARROW","DARTBOARD", "APEX", "ORIENTATION"]#['ORIENTATION','FEED',"ARROW1","ARROW2" ,"DIFFERENCE", "ROTATED", "DARTBOARD"]
    }

    @staticmethod
    def show(frame, window='feed'):
        if window in GUI.ENV['SHOW_GUI']:
            cv2.imshow(window, cv2.resize(frame, (int(frame.shape[1] * GUI.ENV['GUI_RESOLUTION_SCALE']), int(frame.shape[0] * GUI.ENV['GUI_RESOLUTION_SCALE']))))

    @staticmethod
    def imShow(frame):
        if(frame.ndim == 3):
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(frame,cmap='Greys_r')
        plt.show()