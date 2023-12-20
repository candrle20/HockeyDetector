###Conor Andrle
###Hockey Player Object Detection

import os
import numpy as np
import pandas as pd
import cv2
from ultralytics import YOLO


class HockeyDetector:
    '''YOLOv8 Hockey Detector Class
    
    Creates a HockeyDetector object that detects skaters, refs, and goalies from input video
    
    Args:
        model_path (str): Path to model file
        class_path (str, optional): Path to file containing class text. Defaults to 'labels.txt'.
    
    Attributes:
        model: YOLOv8_hockey.pt model
        classes (list): List of classes
    
    Methods:
        detect_video: Outputs video with bounding boxes around detected skaters, refs, and goalies
            ARGS- video_path (str): Path to video file, 
                conf_threshold (float, optional): Confidence threshold. Defaults to 0.5.
                show (bool, optional): Show video. Defaults to True.
                save (bool, optional): Save video. Defaults to True.
    '''

    def __init__(self, model_path, class_path='labels.txt'):

        self.model = YOLO(model_path)
        self.classes= self.obj_classes(class_path)
    

    def obj_classes(self, class_path):
        
        with open(class_path, 'r') as f:
            classes = f.read().splitlines()
    
        return classes


    def detect_video(self, video_path, conf_threshold=0.5, show=True, save=True):
    
        cap = cv2.VideoCapture(video_path)
        
        #Frame Attributes
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps= cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        current_frame = 0

        base_name = os.path.basename(video_path)
        vid_name, ext = os.path.splitext(base_name)

        if save:
            record = cv2.VideoWriter(f'{vid_name}_final.mp4', cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height))

        while True:
            current_frame += 1
            print(f"Processing: {round(current_frame/total_frames * 100, 2)}% complete")
            stream, frame = cap.read()

            if not stream:
                break

            results = self.model.predict(frame, conf=conf_threshold)       
            boxed_frame = results[0].plot()

            if show:
                cv2.imshow("YOLOv8 Hockey Player Detector", boxed_frame)
            if save:
                record.write(boxed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        record.release()
        cap.release()
        cv2.destroyAllWindows()
        

######ARGS######
model_path = 'yolov8_hockey.pt'
video_path = 'videos/HawksPan.mp4'
conf_threshold = 0.5


if __name__ == '__main__':
    detector = HockeyDetector(model_path)
    detector.detect_video(video_path, conf_threshold, show=False, save=True)